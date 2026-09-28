"""Turn a raw scene graph (one entry per detection cluster, often hundreds) into a small, trustworthy one.

Pure python / numpy, no ROS or torch: it runs on a saved `scene_graph.json` without the rosbag or the detector.

Raw clusters have five problems (measured against the simulator's real objects, see `evaluation/`):

  1. one real object is listed several times (a bed seen from five places becomes five clusters);
  2. many clusters are the same physical thing under different class names (desk / vending machine / trash can);
  3. detection score cannot tell a real object from a look-alike: what does help is how much evidence there is, whether
     it is *consistent* (nearby detections of the same class) and whether the place is plausible on the saved map;
  4. structure (walls) and other things nobody navigates to fill the list and the prompt;
  5. a pile of identical things becomes `box_1 ... box_20` instead of "many boxes here".

The steps, in order (each is counted in the returned Report):

  structure   classes with weight 0 (walls) are not navigation targets and are dropped
  merge       same-class entries within the class's `merge_radius` are one object (agglomerative, capped in size)
  arbitrate   entries of different classes on (almost) the same spot compete; the one with the most evidence stays
  group       same-class objects within `group_gap` of each other that add up to `min_group` or more detection
              clusters (and are at least two distinct objects) become ONE group entry with a size ("a group of beds
              here"), never chained beyond `max_group_size`. `members` counts the detection clusters merged in: a lower
              bound on how many objects there are
  rank        importance = class weight x evidence x plausibility (height, map state, distance to mapped obstacles)
  select      entries below `min_importance` are dropped; at most `max_objects` remain. A class that was detected at
              all keeps its single best entry when that is at least half as important as the cut-off, so "go to
              the wheelchair" is answered with the best guess instead of "unknown object"
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass, field, replace

import numpy as np

from .scene_graph import SCENE_GRAPH_VERSION, SceneGraph, SceneObject


@dataclass(frozen=True)
class ClassProfile:
    weight: float = 0.7             # how much the robot cares about this class as a destination (0 = structure)
    merge_radius: float = 1.2       # metres: same-class entries closer than this are one object
    footprint: float = 0.8          # typical width (m); used to size merged entries
    group_gap: float = 2.5          # metres: same-class objects this close chain into a group (None = never group)
    solid: bool = True              # tall enough that the lidar should see it (expects a mapped obstacle nearby)


# Typical values for the classes used in the two scenes. Anything else falls back to ClassProfile().
CLASS_PROFILES = {
    # hospital
    'bed': ClassProfile(1.0, 1.6, 2.0, 3.5),
    'wheelchair': ClassProfile(1.0, 1.0, 0.8, 2.5),
    'vending machine': ClassProfile(1.2, 1.5, 1.0, None),
    'door': ClassProfile(1.0, 1.2, 1.0, None),
    'cart': ClassProfile(0.8, 1.2, 0.8, 2.5),
    'desk': ClassProfile(0.7, 1.2, 1.2, 2.5),
    'chair': ClassProfile(0.6, 1.0, 0.6, 2.0),
    'trash can': ClassProfile(0.5, 0.9, 0.4, 2.0),
    'computer': ClassProfile(0.4, 0.8, 0.4, 2.0),
    'wall': ClassProfile(0.0),
    # warehouse
    'forklift': ClassProfile(1.2, 2.0, 2.0, None),
    'shelf': ClassProfile(1.0, 2.0, 2.0, 3.5),
    'pallet': ClassProfile(0.8, 1.5, 1.2, 2.5),
    'box': ClassProfile(0.8, 0.8, 0.5, 2.0),
    'crate': ClassProfile(0.7, 0.8, 0.5, 2.0),
    'container': ClassProfile(0.9, 1.5, 2.0, None),
    'ladder': ClassProfile(0.7, 1.0, 0.5, None),
    'cone': ClassProfile(0.4, 0.6, 0.3, 2.0),
}


@dataclass
class Settings:
    profiles: dict = field(default_factory=lambda: dict(CLASS_PROFILES))
    conflict_radius: float = 0.7        # different-class entries closer than this compete (0 = off)
    min_group: int = 3                  # detection clusters needed to call something a group
    max_group_size: float = 6.0         # metres: widest a group may become (no chains across a whole floor)
    support_scale: float = 6.0          # evidence at which confidence reaches ~63 %
    neighbour_credit: float = 0.5       # share of nearby same-class evidence that supports an entry
    min_importance: float = 0.2
    keep_best_per_class: bool = True
    max_objects: int = 60
    z_range: tuple = (0.15, 2.2)        # heights (m) of the detected surface point outside which it is suspect
    clearance_near: float = 0.6         # a solid object farther than this from every mapped obstacle is suspicious...
    clearance_far: float = 1.5          # ...and at this distance the plausibility bottoms out
    off_map_factor: float = 0.4         # plausibility of a detection in unknown / unmapped space
    floating_factor: float = 0.5
    bad_height_factor: float = 0.5

    def profile(self, label):
        return self.profiles.get(label, ClassProfile())


@dataclass
class Report:
    n_input: int = 0
    n_output: int = 0
    structure: list = field(default_factory=list)
    merged: int = 0
    arbitrated: list = field(default_factory=list)      # (dropped id, label, kept id, label)
    grouped: list = field(default_factory=list)         # (group id, members)
    low_importance: list = field(default_factory=list)
    over_limit: list = field(default_factory=list)

    def summary(self):
        lines = [f'{self.n_input} raw entries -> {self.n_output} kept',
                 f'  dropped as structure (not destinations): {len(self.structure)}',
                 f'  merged duplicates of the same object:    {self.merged}',
                 f'  lost a different-class conflict:         {len(self.arbitrated)}',
                 f'  collapsed into {len(self.grouped)} group(s):            {sum(m for _, m in self.grouped)} entries',
                 f'  dropped for low importance:              {len(self.low_importance)}',
                 f'  dropped over the size limit:             {len(self.over_limit)}']
        return '\n'.join(lines)


# ----------------------------------------------------------------------------- building blocks

@dataclass
class _Blob:
    """Same-class entries believed to be one object (or one group)."""
    label: str
    members: list       # SceneObject (raw entries or merged instances)

    def support(self):
        return sum(m.count * m.score for m in self.members)

    def centre(self):
        w = np.array([max(m.count * m.score, 1e-3) for m in self.members])
        p = np.array([[m.x, m.y, m.z] for m in self.members])
        return (p * w[:, None]).sum(axis=0) / w.sum()

    def spread(self):
        """Largest distance between two members (0 for a single one)."""
        p = np.array([[m.x, m.y] for m in self.members])
        if len(p) < 2:
            return 0.0
        return float(np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)).max())


def _agglomerate(blobs, link, max_spread):
    """Merge the closest pair of blobs (single link <= `link`) while the merged blob stays <= `max_spread` wide."""
    blobs = list(blobs)
    while True:
        best = None
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                a, b = blobs[i], blobs[j]
                d = min(math.hypot(p.x - q.x, p.y - q.y) for p in a.members for q in b.members)
                if d <= link and (best is None or d < best[0]):
                    if _Blob(a.label, a.members + b.members).spread() <= max_spread:
                        best = (d, i, j)
        if best is None:
            return blobs
        _, i, j = best
        blobs[i] = _Blob(blobs[i].label, blobs[i].members + blobs[j].members)
        del blobs[j]


def _agglomerate_blobs(blobs, link, max_spread):
    """Like _agglomerate, but the units are already-merged blobs; the result is a list of lists of those blobs."""
    clusters = [[b] for b in blobs]

    def flat(c):
        return _Blob(c[0].label, [m for b in c for m in b.members])

    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = min(math.hypot(p.x - q.x, p.y - q.y)
                        for a in clusters[i] for p in a.members for b in clusters[j] for q in b.members)
                if d <= link and (best is None or d < best[0]) and flat(clusters[i] + clusters[j]).spread() <= max_spread:
                    best = (d, i, j)
        if best is None:
            return clusters
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]


def _to_object(blob, profile, kind, obj_id=''):
    x, y, z = blob.centre()
    count = sum(m.count for m in blob.members)
    score = sum(m.count * m.score for m in blob.members) / max(count, 1)
    spread = blob.spread()
    members = sum(m.members for m in blob.members)
    size = 0.0 if len(blob.members) == 1 and blob.members[0].size == 0.0 else max(
        spread + profile.footprint, max(m.size for m in blob.members))
    return SceneObject(
        id=obj_id, label=blob.label, x=round(float(x), 3), y=round(float(y), 3), z=round(float(z), 3),
        count=count, score=round(float(score), 3), kind=kind, members=members, size=round(float(size), 2),
        z_min=round(float(min(m.z_min for m in blob.members)), 3),
        z_max=round(float(max(m.z_max for m in blob.members)), 3))


def _plausibility(obj, profile, grid, s):
    """Soft 0..1 factor: is this a believable place for such an object?"""
    factor = 1.0
    if not (s.z_range[0] <= obj.z <= s.z_range[1]):
        factor *= s.bad_height_factor           # a "desk" at floor level or a "chair" at ceiling height
    if grid is not None:
        from .map_grid import FREE, OCCUPIED
        state = grid.state(obj.x, obj.y)
        if state == OCCUPIED:
            return factor
        if state != FREE:
            factor *= s.off_map_factor          # unknown / outside the map: nothing confirms the place exists
        elif profile.solid:
            c = grid.clearance(obj.x, obj.y)
            if c > s.clearance_near:            # solid things are obstacles: a detection floating in open floor is odd
                t = min(1.0, (c - s.clearance_near) / max(s.clearance_far - s.clearance_near, 1e-6))
                factor *= 1.0 - t * (1.0 - s.floating_factor)
    return factor


# ----------------------------------------------------------------------------- the pipeline

def consolidate(objects, grid=None, settings=None):
    """Return (consolidated SceneObjects, Report). `grid` is an optional MapGrid of the saved map."""
    s = settings or Settings()
    rep = Report(n_input=len(objects))

    # structure -------------------------------------------------------------
    work = []
    for o in sorted(objects, key=lambda o: (o.label, o.x, o.y, o.z, o.count, o.id)):      # same output for any input order
        if s.profile(o.label).weight <= 0:
            rep.structure.append(o.id)
        elif all(math.isfinite(v) for v in (o.x, o.y, o.z)):
            work.append(o)

    # merge duplicates of one object ---------------------------------------
    by_label = {}
    for o in work:
        by_label.setdefault(o.label, []).append(o)
    instances = []
    for label, entries in by_label.items():
        prof = s.profile(label)
        blobs = _agglomerate([_Blob(label, [e]) for e in entries], prof.merge_radius, 2 * prof.merge_radius)
        rep.merged += len(entries) - len(blobs)
        instances += blobs

    # different classes on the same spot ------------------------------------
    if s.conflict_radius > 0:
        dead = set()

        def strength(i):
            return -instances[i].support(), instances[i].label, tuple(instances[i].centre()[:2])

        order = sorted(range(len(instances)), key=strength)
        for a in range(len(order)):
            i = order[a]
            if i in dead:
                continue
            ci = instances[i].centre()
            for b in range(a + 1, len(order)):
                j = order[b]
                if j in dead or instances[j].label == instances[i].label:
                    continue
                cj = instances[j].centre()
                if math.hypot(ci[0] - cj[0], ci[1] - cj[1]) <= s.conflict_radius:
                    dead.add(j)
                    rep.arbitrated.append((instances[j].members[0].id, instances[j].label,
                                           instances[i].members[0].id, instances[i].label))
        instances = [b for k, b in enumerate(instances) if k not in dead]

    # groups of identical things -------------------------------------------
    entries = []
    for label in sorted({b.label for b in instances}):
        prof = s.profile(label)
        mine = [b for b in instances if b.label == label]
        if prof.group_gap is None or len(mine) < 2:
            entries += [(_to_object(b, prof, 'instance'), prof) for b in mine]
            continue
        # chain instances whose nearest detections are within the gap, but never beyond max_group_size
        clusters = _agglomerate_blobs(mine, prof.group_gap, s.max_group_size)
        for cluster in clusters:
            fragments = sum(m.members for b in cluster for m in b.members)
            if len(cluster) >= 2 and fragments >= s.min_group:
                merged = _Blob(label, [m for b in cluster for m in b.members])
                entries.append((_to_object(merged, prof, 'group'), prof))
            else:
                entries += [(_to_object(b, prof, 'instance'), prof) for b in cluster]

    # rank and select --------------------------------------------------------
    scored = []
    for o, prof in entries:
        support = o.count * o.score
        nearby = sum(q.count * q.score for q, _ in entries
                     if q is not o and q.label == o.label and math.hypot(q.x - o.x, q.y - o.y) <= 3.0 + o.radius)
        evidence = 1.0 - math.exp(-(support + s.neighbour_credit * nearby) / s.support_scale)
        o.importance = round(prof.weight * evidence * _plausibility(o, prof, grid, s), 3)
        scored.append(o)
    scored.sort(key=lambda o: (-o.importance, o.label, o.x, o.y))
    kept, seen_labels = [], set()
    best_of = {}
    for o in scored:
        best_of.setdefault(o.label, o)          # `scored` is sorted, so the first one seen is the class's best
    for o in scored:
        rescued = (s.keep_best_per_class and best_of[o.label] is o and o.label not in seen_labels
                   and o.importance >= s.min_importance / 2)
        if o.importance < s.min_importance and not rescued:
            rep.low_importance.append(o.id or o.label)
        elif len(kept) >= s.max_objects and o.label in seen_labels:
            rep.over_limit.append(o.label)
        else:
            kept.append(o)
            seen_labels.add(o.label)

    counters = {}
    for o in kept:
        counters[o.label] = counters.get(o.label, 0) + 1
        o.id = f"{o.label.replace(' ', '_')}_{counters[o.label]}"
    for c in kept:
        if c.kind == 'group':
            rep.grouped.append((c.id, c.members))
    rep.n_output = len(kept)
    return kept, rep


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description='Consolidate a raw scene_graph.json into a compact, trustworthy one.')
    p.add_argument('--input', default=os.path.join(os.path.expanduser('~'), 'scene_graph', 'scene_graph.json'))
    p.add_argument('--output', default=None, help='default: <input>.consolidated.json')
    p.add_argument('--map', default=os.path.join(os.path.expanduser('~'), 'my_map.yaml'),
                   help='saved Nav2 map (yaml) used to judge plausibility; "none" to skip')
    p.add_argument('--min-importance', type=float, default=Settings.min_importance)
    p.add_argument('--max-objects', type=int, default=Settings.max_objects)
    p.add_argument('--conflict-radius', type=float, default=Settings.conflict_radius)
    p.add_argument('--no-groups', action='store_true', help='never collapse repeated objects into groups')
    args = p.parse_args(argv)

    graph = SceneGraph.load(args.input)
    grid = None
    if args.map and args.map.lower() != 'none' and os.path.isfile(args.map):
        from .map_grid import MapGrid
        grid = MapGrid.from_yaml(args.map)
    elif args.map and args.map.lower() != 'none':
        print(f'Map {args.map} not found: plausibility is judged without it.', file=sys.stderr)
    settings = Settings(min_importance=args.min_importance, max_objects=args.max_objects,
                        conflict_radius=args.conflict_radius)
    if args.no_groups:
        settings.profiles = {k: replace(v, group_gap=None) for k, v in settings.profiles.items()}
    objects, report = consolidate(graph.objects, grid, settings)
    out = args.output or os.path.splitext(args.input)[0] + '.consolidated.json'
    SceneGraph(objects, graph.frame_id, consolidated=True).save(out)
    print(report.summary())
    print(f'Wrote {out} (version {SCENE_GRAPH_VERSION})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
