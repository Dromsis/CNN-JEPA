# YOLO26-SEA backbone wrapper for CNN-JEPA pretraining.
#
# Exposes the Ultralytics YOLO26-SEA backbone (layers 0-11 of yolo26-sea.yaml) as a
# plain nn.Module compatible with the CNN-JEPA / SparK machinery, SPLIT into:
#   - trunk = layers 0-8  (pure Conv + C3k2)  -> runs SPARSE during pretraining
#   - tail  = layers 9-11 (SESA, SPPF, C2PSA) -> runs DENSE  (after densify)
#
# The split point is AFTER layer 8 because SESA/SPPF/C2PSA mix information across ALL
# spatial positions (SE global pool, SimAM mean/var, PSA self-attention). Running them
# sparse would leak masked information; instead the training loop densifies (fills masked
# positions with a learned mask token) right after the trunk, then runs the tail dense.
#
# Contracts consumed by pretrain/train_ijepa_yolo.py:
#   - get_downsample_ratio() -> 32   (stride of the layer-8 feature map @ 640 input -> 20x20)
#   - trunk_channels : channels of the layer-8 output  (mask-token dim, densify point)
#   - num_features   : channels of the layer-11 output (predictor / loss dim)
#   - forward_trunk(x), forward_tail(x), forward(x)

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from timm.models.registry import register_model


# Number of the LAST layer kept in the JEPA encoder (inclusive). C2PSA in yolo26-sea.yaml.
ENCODER_LAST_LAYER = 11
# Index of the last layer of the SPARSE trunk (inclusive). Last pure-conv C3k2.
TRUNK_LAST_LAYER = 8


def _ensure_ultralytics_importable(yolo_repo_path: Optional[str]) -> None:
    """Make the vendored Ultralytics package (in the yolo_sea_homemade repo) importable."""
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    candidate = yolo_repo_path or os.environ.get("YOLO_SEA_REPO")
    if candidate and candidate not in sys.path:
        sys.path.insert(0, candidate)
    import ultralytics  # noqa: F401  (raises a clear ImportError if still not found)


class YOLO26SEABackbone(nn.Module):
    """Layers 0-11 of yolo26-sea.yaml, split into a sparse trunk and a dense tail."""

    def __init__(
        self,
        yaml_path: str,
        scale: str = "n",
        yolo_repo_path: Optional[str] = None,
        in_chans: int = 3,
        **_ignored,
    ):
        super().__init__()
        _ensure_ultralytics_importable(yolo_repo_path)
        from ultralytics.nn.tasks import DetectionModel

        # Build the full detection model just to get correctly-parsed, scale-applied layers,
        # then keep only the backbone (0..ENCODER_LAST_LAYER). The head is discarded.
        det = DetectionModel(cfg=self._scaled_cfg(yaml_path, scale), ch=in_chans, verbose=False)
        full_layers = det.model  # nn.Sequential(backbone + head)

        # Backbone layers 0..11 in yolo26-sea.yaml are all `from: -1` (sequential), so we can
        # run them as a plain chain without the Ultralytics routing/save machinery.
        self._assert_sequential(full_layers, ENCODER_LAST_LAYER)

        self.trunk = nn.Sequential(*[full_layers[i] for i in range(0, TRUNK_LAST_LAYER + 1)])
        self.tail = nn.Sequential(
            *[full_layers[i] for i in range(TRUNK_LAST_LAYER + 1, ENCODER_LAST_LAYER + 1)]
        )

        # Infer channel dims with a dummy dense forward (CPU, no grad).
        self.trunk_channels, self.num_features = self._infer_channels(in_chans)

    @staticmethod
    def _scaled_cfg(yaml_path: str, scale: str):
        """Return a cfg dict for the requested scale (Ultralytics keys scales by letter)."""
        import yaml as _yaml

        with open(yaml_path, "r") as f:
            cfg = _yaml.safe_load(f)
        cfg["scale"] = scale
        return cfg

    @staticmethod
    def _assert_sequential(layers, last_idx: int) -> None:
        for i in range(0, last_idx + 1):
            f = getattr(layers[i], "f", -1)
            if f != -1:
                raise ValueError(
                    f"Backbone layer {i} has from={f} (non-sequential). The plain-chain "
                    f"forward in YOLO26SEABackbone only supports from=-1 in layers 0..{last_idx}."
                )

    @torch.no_grad()
    def _infer_channels(self, in_chans: int):
        was_training = self.training
        self.eval()
        dummy = torch.zeros(1, in_chans, 640, 640)
        t = self.trunk(dummy)
        out = self.tail(t)
        if was_training:
            self.train()
        return t.shape[1], out.shape[1]

    def get_downsample_ratio(self) -> int:
        return 32  # layer-8 (and layer-11) feature map is stride 32 @ 640 input

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def forward_tail(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full dense backbone (used by the EMA target encoder)."""
        return self.tail(self.trunk(x))

    # The split design converts only the trunk to sparse (in the training model), so the
    # backbone-level `sparse` flag the base IJEPA_CNN toggles is a no-op here.
    @property
    def sparse(self):
        return getattr(self, "_sparse", False)

    @sparse.setter
    def sparse(self, value):
        self._sparse = value


@register_model
def yolo26_sea_backbone(pretrained=False, **kwargs):
    """timm entry point. Pass yaml_path / scale / yolo_repo_path via cfg.backbone.kwargs."""
    for k in ("pretrained_cfg", "pretrained_cfg_overlay", "num_classes", "global_pool"):
        kwargs.pop(k, None)
    return YOLO26SEABackbone(**kwargs)
