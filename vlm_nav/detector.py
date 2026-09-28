"""GroundingDINO open-vocabulary detector wrapper (torch / groundingdino are imported lazily)."""
import logging
from dataclasses import dataclass

from .scene_graph import build_caption, match_phrase_to_class

log = logging.getLogger('vlm_nav.detector')


@dataclass
class Detection:
    label: str      # one of the configured classes
    score: float    # GroundingDINO confidence in [0, 1]
    box: tuple      # (x0, y0, x1, y1) in pixels of the image passed to detect()


def box_iou(a, b):
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _contained_fraction(inner, outer):
    """Share of `inner`'s area that lies inside `outer`."""
    iw = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    ih = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return iw * ih / area if area > 0 else 0.0


def filter_detections(detections, image_shape, iou_same=0.5, iou_cross=0.7, contained=0.9,
                      max_box_fraction=0.6, min_box_px=12):
    """Clean up one frame's detections before they become observations.

    GroundingDINO returns several overlapping boxes for one object and never suppresses them, so counting boxes counted
    the same sighting many times (two boxes in one frame passed the "seen twice" test). This keeps the best box per
    object: a box is dropped when it overlaps a higher-scoring one of the same class (IoU > iou_same, or mostly inside
    it) or of another class (IoU > iou_cross: one object cannot be two things). Boxes covering most of the image (their
    centre says nothing about where the object is) and slivers are dropped too.
    """
    h, w = image_shape[:2]
    kept = []
    for d in sorted(detections, key=lambda d: -d.score):
        x0, y0, x1, y1 = d.box
        if x1 - x0 < min_box_px or y1 - y0 < min_box_px:
            continue
        if (x1 - x0) * (y1 - y0) > max_box_fraction * w * h:
            continue
        dup = False
        for k in kept:
            if k.label == d.label:
                dup = box_iou(k.box, d.box) > iou_same or _contained_fraction(d.box, k.box) > contained
            else:
                dup = box_iou(k.box, d.box) > iou_cross
            if dup:
                break
        if not dup:
            kept.append(d)
    return kept


def _use_pytorch_deformable_attention():
    """Route GroundingDINO's CUDA deformable-attention call to its pure-PyTorch implementation."""
    import groundingdino.models.GroundingDINO.ms_deform_attn as msda

    class _PyTorchOp:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return msda.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

    msda.MultiScaleDeformableAttnFunction = _PyTorchOp


class GroundingDinoDetector:
    def __init__(self, config_path, weights_path, classes, box_threshold=0.30, text_threshold=0.25,
                 device=None):
        import torch
        from groundingdino.util.inference import load_model
        import groundingdino.datasets.transforms as T

        # Workaround kept from the original script: these SDP kernels misbehave on some new GPUs.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if device.startswith('cuda'):
            try:
                from groundingdino import _C  # noqa: F401  (custom deformable-attention CUDA op)
            except ImportError:
                # The compiled op needs nvcc at install time. GroundingDINO ships the same operation in pure
                # PyTorch (its CPU path), which also runs on the GPU, so use that instead of falling back to CPU.
                log.warning('GroundingDINO CUDA op (_C) is not built; using the pure-PyTorch deformable attention on the GPU.')
                _use_pytorch_deformable_attention()
        self.device = device
        self.classes = list(classes)
        self.caption = build_caption(self.classes)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._torch = torch
        # Same preprocessing as groundingdino.util.inference.load_image: resize + ImageNet normalisation.
        self._transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        log.info('Loading GroundingDINO on %s...', device)
        self.model = load_model(config_path, weights_path, device=device)

    def detect(self, rgb):
        """Detect objects in an (H, W, 3) uint8 RGB image. Returns a list of Detection."""
        from groundingdino.util.inference import predict
        from PIL import Image

        h, w = rgb.shape[:2]
        image, _ = self._transform(Image.fromarray(rgb), None)
        boxes, logits, phrases = predict(
            model=self.model, image=image, caption=self.caption,
            box_threshold=self.box_threshold, text_threshold=self.text_threshold, device=self.device)

        detections = []
        for (cx, cy, bw, bh), score, phrase in zip(boxes.cpu().numpy(), logits.cpu().numpy(), phrases):
            label = match_phrase_to_class(phrase, self.classes)
            if label is None:
                log.debug('dropping unmatched phrase %r', phrase)
                continue
            box = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            detections.append(Detection(label, float(score), tuple(float(v) for v in box)))
        return detections
