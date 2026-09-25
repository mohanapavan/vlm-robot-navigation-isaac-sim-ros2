"""Semantic scene graph: per-object map-frame positions built from detections.

Pure python / numpy (no ROS, no torch) so it can be tested anywhere.
"""
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

SCENE_GRAPH_VERSION = 2

DEFAULT_CLASSES = ['box', 'shelf', 'pallet', 'forklift', 'door', 'wall',
                   'crate', 'container', 'ladder', 'cone']


# Big objects are seen from many sides and distances, so their surface points spread out: cluster them with a
# larger radius than the 1 m default so one forklift / shelf / wall is not reported as several instances.
DEFAULT_CLASS_RADIUS = {'forklift': 2.0, 'shelf': 2.5, 'wall': 4.0, 'container': 1.5, 'pallet': 1.5}


def build_caption(classes):
    """GroundingDINO text prompt: class names separated by ' . '."""
    return ' . '.join(classes)


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
    """One detection, already located in the map frame."""
    label: str
    score: float
    x: float
    y: float
    z: float = 0.0


@dataclass
class _Cluster:
    label: str
    obs: list = field(default_factory=list)

    def centroid(self):
        w = np.array([max(o.score, 1e-3) for o in self.obs])
        pts = np.array([[o.x, o.y, o.z] for o in self.obs])
        return (pts * w[:, None]).sum(axis=0) / w.sum()


def cluster_observations(observations, radius=1.0, min_observations=2, class_radius=None):
    """Group observations into distinct object instances.

    Observations of the same label within `radius` metres of an existing
    cluster's centroid join it; otherwise they start a new one. Clusters are
    then merged if their centroids drifted within `radius` of each other, and
    clusters seen fewer than `min_observations` times are dropped as noise.
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

    kept = [c for c in clusters if len(c.obs) >= min_observations]
    kept.sort(key=lambda c: (c.label, -len(c.obs), c.centroid()[0], c.centroid()[1]))
    objects, per_label = [], {}
    for c in kept:
        per_label[c.label] = per_label.get(c.label, 0) + 1
        x, y, z = c.centroid()
        objects.append(SceneObject(
            id=f'{c.label}_{per_label[c.label]}', label=c.label,
            x=round(float(x), 3), y=round(float(y), 3), z=round(float(z), 3),
            count=len(c.obs), score=round(float(np.mean([o.score for o in c.obs])), 3)))
    return objects


@dataclass
class SceneObject:
    id: str
    label: str
    x: float
    y: float
    z: float
    count: int
    score: float

    @property
    def xy(self):
        return self.x, self.y


class SceneGraph:
    def __init__(self, objects, frame_id='map'):
        self.objects = list(objects)
        self.frame_id = frame_id

    def __len__(self):
        return len(self.objects)

    def labels(self):
        return sorted({o.label for o in self.objects})

    def names(self):
        """Every name the robot can be sent to: instance ids and bare labels."""
        return sorted({o.id for o in self.objects} | {o.label for o in self.objects})

    def resolve(self, name, robot_xy=None):
        """Find the object for an instance id ('forklift_2') or bare label ('forklift').

        A bare label with several instances resolves to the one nearest `robot_xy`
        (the first one if the robot position is unknown). Returns None for unknown names.
        """
        name = name.strip().lower()
        for o in self.objects:
            if o.id == name:
                return o
        candidates = [o for o in self.objects if o.label == name]
        if not candidates:
            return None
        if robot_xy is None:
            return candidates[0]
        return min(candidates, key=lambda o: math.hypot(o.x - robot_xy[0], o.y - robot_xy[1]))

    def resolve_all(self, name, robot_xy=None):
        """Every object `name` can mean, nearest to `robot_xy` first: one for an id, all instances for a bare label."""
        name = name.strip().lower()
        for o in self.objects:
            if o.id == name:
                return [o]
        found = [o for o in self.objects if o.label == name]
        if robot_xy is not None:
            found.sort(key=lambda o: math.hypot(o.x - robot_xy[0], o.y - robot_xy[1]))
        return found

    def to_dict(self):
        return {
            'version': SCENE_GRAPH_VERSION,
            'frame_id': self.frame_id,
            'objects': [o.__dict__ for o in self.objects],
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get('version') != SCENE_GRAPH_VERSION:
            raise ValueError(
                f'{path} is not a version-{SCENE_GRAPH_VERSION} scene graph (old graphs stored the '
                "robot's odom position, not the object's map position). Rebuild it with build_scene_graph.")
        return cls([SceneObject(**o) for o in data['objects']], data.get('frame_id', 'map'))
