"""Semantic scene graph: per-object map-frame positions built from detections.

Pure python / numpy (no ROS, no torch) so it can be tested anywhere.
"""
import json
import math
import re
from dataclasses import dataclass, field, fields

import numpy as np

SCENE_GRAPH_VERSION = 3
# Version 2 (plain clusters) still loads: the version-3 fields all have defaults.
READABLE_VERSIONS = (2, 3)

DEFAULT_CLASSES = ['box', 'shelf', 'pallet', 'forklift', 'door', 'wall',
                   'crate', 'container', 'ladder', 'cone']


# Big objects are seen from many sides and distances, so their surface points spread out: cluster them with a
# larger radius than the 1 m default so one forklift / shelf / wall is not reported as several instances.
DEFAULT_CLASS_RADIUS = {'forklift': 2.0, 'shelf': 2.5, 'wall': 4.0, 'container': 1.5, 'pallet': 1.5}


def build_caption(classes):
    """GroundingDINO text prompt: class names separated by ' . '."""
    return ' . '.join(classes)


def normalize_name(name):
    """Canonical form of an object name so that 'Vending_Machine-3', 'vending machine 3' and 'vending machine_3' agree."""
    return ' '.join(re.split(r'[\s_\-]+', name.strip().lower().strip(' \t"\'`.,;:!?()[]<>*'))).strip()


_ARTICLES = ('the ', 'a ', 'an ', 'that ', 'this ', 'nearest ', 'closest ')


def _name_variants(name):
    """Forms a user/model may write for the same thing: without an article, singular for a plural ('beds', 'boxes')."""
    n = normalize_name(name)
    stripped = True
    while stripped:                                   # 'the closest trash cans' -> 'trash cans'
        stripped = False
        for art in _ARTICLES:
            if n.startswith(art):
                n, stripped = n[len(art):], True
    variants = [n]
    if n.endswith('ies') and len(n) > 4:
        variants.append(n[:-3] + 'y')
    if n.endswith('es') and len(n) > 3:
        variants.append(n[:-2])
    if n.endswith('s') and len(n) > 2:
        variants.append(n[:-1])
    return variants


def match_phrase_to_class(phrase, classes):
    """Map a GroundingDINO output phrase back to exactly one entry of `classes`.

    GroundingDINO returns free text: empty strings, plurals, or several class
    names merged into one phrase ('box shelf'). Only phrases that resolve to a
    single class are accepted; anything else returns None so the detection is dropped.
    """
    words = re.findall(r'[a-z]+', phrase.lower())
    if not words:
        return None
    hits = set()
    for cls in classes:
        cls_words = cls.lower().split()
        n = len(cls_words)
        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            if all(w == c or w == c + 's' or w == c + 'es' for w, c in zip(window, cls_words)):
                hits.add(cls)
    return hits.pop() if len(hits) == 1 else None


@dataclass
class Observation:
    """One detection, already located in the map frame.

    `frame` identifies the camera frame it came from and `range` the distance (m) from the camera at which its depth
    was measured. Both are optional: with them, evidence is counted in distinct *views* (several boxes on one object in
    one frame are one sighting) and far observations, whose stereo depth is the least accurate, weigh less.
    """
    label: str
    score: float
    x: float
    y: float
    z: float = 0.0
    frame: int = None
    range: float = None


# Stereo depth error grows with the square of the range (disparity = fx * baseline / z), so an observation at RANGE_HALF_WEIGHT
# metres counts a quarter as much as one made from close by.
RANGE_HALF_WEIGHT = 6.0


def observation_weight(o):
    """Weight of an observation in a cluster centroid: detection score, discounted with range when it is known."""
    w = max(o.score, 1e-3)
    if o.range is not None and o.range > 0:
        w /= 1.0 + (o.range / RANGE_HALF_WEIGHT) ** 2
    return w


@dataclass
class _Cluster:
    label: str
    obs: list = field(default_factory=list)

    def centroid(self):
        w = np.array([observation_weight(o) for o in self.obs])
        pts = np.array([[o.x, o.y, o.z] for o in self.obs])
        return (pts * w[:, None]).sum(axis=0) / w.sum()

    def views(self):
        """Independent sightings: distinct camera frames when known, else one per observation."""
        if any(o.frame is None for o in self.obs):
            return len(self.obs)
        return len({o.frame for o in self.obs})


def cluster_observations(observations, radius=1.0, min_observations=2, class_radius=None):
    """Group observations into distinct object instances.

    Observations of the same label within `radius` metres of an existing
    cluster's centroid join it; otherwise they start a new one. Clusters are
    then merged if their centroids drifted within `radius` of each other, and
    clusters seen in fewer than `min_observations` views are dropped as noise (a view = one camera frame when the
    observations carry frame ids, otherwise one observation).
    `class_radius` ({label: metres}) overrides `radius` for individual labels.
    Returns a list of SceneObject sorted by label, then by descending count.
    """
    class_radius = class_radius or {}
    clusters = []
    for o in sorted(observations, key=lambda o: -o.score):
        best, best_d = None, class_radius.get(o.label, radius)
        for c in clusters:
            if c.label != o.label:
                continue
            cx, cy, _ = c.centroid()
            d = math.hypot(cx - o.x, cy - o.y)
            if d <= best_d:
                best, best_d = c, d
        if best is None:
            best = _Cluster(o.label)
            clusters.append(best)
        best.obs.append(o)

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if a.label != b.label:
                    continue
                ax, ay, _ = a.centroid()
                bx, by, _ = b.centroid()
                if math.hypot(ax - bx, ay - by) <= class_radius.get(a.label, radius):
                    a.obs.extend(b.obs)
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break

    kept = [c for c in clusters if c.views() >= min_observations]
    kept.sort(key=lambda c: (c.label, -c.views(), c.centroid()[0], c.centroid()[1]))
    objects, per_label = [], {}
    for c in kept:
        per_label[c.label] = per_label.get(c.label, 0) + 1
        x, y, z = c.centroid()
        objects.append(SceneObject(
            id=f'{c.label}_{per_label[c.label]}', label=c.label,
            x=round(float(x), 3), y=round(float(y), 3), z=round(float(z), 3),
            count=c.views(), score=round(float(np.mean([o.score for o in c.obs])), 3)))
    return objects


@dataclass
class SceneObject:
    """One thing the robot can be sent to: a single object, or a `group` of many of the same class in one place.

    x, y, z: centre in the map frame (a group's centroid). count: how many detections support it, score: their mean
    confidence. The rest is filled in by the consolidation step (consolidate.py) and defaults to "a plain instance".
    """
    id: str
    label: str
    x: float
    y: float
    z: float
    count: int
    score: float
    kind: str = 'instance'      # 'instance' | 'group'
    members: int = 1            # instances merged into this entry (a group of stacked boxes has many)
    size: float = 0.0           # footprint diameter in metres (0 = unknown, i.e. a single surface point)
    z_min: float = None         # height range of the merged detections (a stack of boxes spans several)
    z_max: float = None
    importance: float = 0.0     # ranking used to decide what is worth keeping / listing

    def __post_init__(self):
        if self.z_min is None:
            self.z_min = self.z
        if self.z_max is None:
            self.z_max = self.z

    @property
    def xy(self):
        return self.x, self.y

    @property
    def radius(self):
        return self.size / 2.0


def dump_observations(observations, path, frame_id='map'):
    """Save raw observations so the clustering / consolidation can be redone without the bag or the detector."""
    with open(path, 'w') as f:
        json.dump({'version': 1, 'frame_id': frame_id, 'observations': [o.__dict__ for o in observations]}, f)


def load_observations(path):
    with open(path) as f:
        data = json.load(f)
    known = {f.name for f in fields(Observation)}
    return [Observation(**{k: v for k, v in o.items() if k in known}) for o in data['observations']]


class SceneGraph:
    def __init__(self, objects, frame_id='map', consolidated=False):
        self.objects = list(objects)
        self.frame_id = frame_id
        self.consolidated = consolidated      # True once consolidate.py has merged / ranked / grouped the entries
        self.dropped_on_load = 0

    def __len__(self):
        return len(self.objects)

    def labels(self):
        return sorted({o.label for o in self.objects})

    def names(self):
        """Every name the robot can be sent to: instance ids and bare labels."""
        return sorted({o.id for o in self.objects} | {o.label for o in self.objects})

    def _match(self, name):
        """Objects `name` can mean: the one whose id it is, else every object of that label ([] when unknown)."""
        for variant in _name_variants(name):
            for o in self.objects:
                if normalize_name(o.id) == variant:
                    return [o]
            found = [o for o in self.objects if normalize_name(o.label) == variant]
            if found:
                return found
        return []

    def resolve(self, name, robot_xy=None):
        """Find the object for an instance id ('forklift_2') or bare label ('forklift').

        Case, '_' vs ' ' and simple plurals do not matter ('Trash_Can_1', 'the beds'). A bare label with several
        instances resolves to the one nearest `robot_xy` (the first one if the robot position is unknown).
        Returns None for unknown names.
        """
        found = self.resolve_all(name, robot_xy)
        return found[0] if found else None

    def resolve_all(self, name, robot_xy=None):
        """Every object `name` can mean, nearest to `robot_xy` first: one for an id, all instances for a bare label."""
        found = self._match(name)
        if robot_xy is not None and len(found) > 1:
            found.sort(key=lambda o: math.hypot(o.x - robot_xy[0], o.y - robot_xy[1]))
        return found

    def to_dict(self):
        return {
            'version': SCENE_GRAPH_VERSION,
            'frame_id': self.frame_id,
            'consolidated': self.consolidated,
            'objects': [o.__dict__ for o in self.objects],
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get('version') not in READABLE_VERSIONS:
            raise ValueError(
                f'{path} is not a version-{"/".join(map(str, READABLE_VERSIONS))} scene graph (old graphs stored the '
                "robot's odom position, not the object's map position). Rebuild it with build_scene_graph.")
        known = {f.name for f in fields(SceneObject)}
        objects, dropped = [], 0
        for raw in data.get('objects', []):
            try:
                obj = SceneObject(**{k: v for k, v in raw.items() if k in known})
                if not all(math.isfinite(v) for v in (obj.x, obj.y, obj.z)):
                    raise ValueError('non-finite position')
            except (TypeError, ValueError):
                dropped += 1          # one malformed entry must not make the whole map unusable
                continue
            objects.append(obj)
        graph = cls(objects, data.get('frame_id', 'map'), bool(data.get('consolidated', False)))
        graph.dropped_on_load = dropped
        return graph
