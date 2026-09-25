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
