# CNN-JEPA pretraining for the YOLO26-SEA backbone.
#
# Variant of pretrain/train_ijepacnn.IJEPA_CNN. The ONLY architectural difference is WHERE
# the densify (fill-in mask tokens) happens:
#
#   IJEPA_CNN (ResNet/ConvNeXt):  masked -> [sparse backbone ENTIER] -> densify -> predictor
#   IJEPA_YOLO (this file):       masked -> [sparse trunk 0-8] -> densify -> [dense tail 9-11] -> predictor
#
# The tail (SESA, SPPF, C2PSA) mixes information globally, so it must run on a DENSE feature
# map. We therefore densify right after the trunk and let the tail run normally. Masking,
# EMA target encoder, loss and the whole training loop are inherited unchanged.

import copy

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from lightly.models.utils import deactivate_requires_grad

from pretrain.trainer_common import LightlyModelMomentum, main_pretrain
from pretrain.train_ijepacnn import IJEPA_CNN
import models.sparse_encoder as sparse_encoder

# Import to register `yolo26_sea_backbone` in the timm model registry (used by the base
# LightlyModel via timm.create_model(cfg.backbone.name, ...)).
import models.yolo_backbone  # noqa: F401


class IJEPA_YOLO(IJEPA_CNN):
    def __init__(self, cfg: DictConfig):
        # Build self.backbone (YOLO26SEABackbone) + self.backbone_momentum (dense deepcopy).
        # We deliberately bypass IJEPA_CNN.__init__ because it sparse-converts the WHOLE
        # backbone and sizes the mask token at the final feature dim — both wrong for the
        # split (trunk-sparse / tail-dense) design.
        LightlyModelMomentum.__init__(self, cfg)

        # Sparse-convert ONLY the trunk (layers 0-8). The tail (9-11) stays dense.
        self.backbone.trunk = sparse_encoder.dense_model_to_sparse(self.backbone.trunk)

        # The sparse backbone can't be used for online eval; use the dense momentum copy.
        self.backbone_for_online_eval = self.backbone_momentum

        # Densify token lives at the TRUNK output (the densify point, after layer 8).
        self.mask_token = nn.Parameter(torch.zeros(1, self.backbone.trunk_channels, 1, 1))
        trunc_normal_(self.mask_token, mean=0, std=0.02, a=-0.02, b=0.02)

        # Predictor operates at the encoder OUTPUT dim (tail / layer-11 = num_features).
        norm_cls = nn.BatchNorm2d
        c = self.backbone.num_features

        # Optional projection head (student) + its EMA copy (teacher). Off by default.
        if self.cfg.get("use_projection_head", False):
            proj_layers = []
            proj_depth = self.cfg.get("projection_head_depth", 2)
            for i in range(proj_depth):
                proj_layers.append(nn.Conv2d(c, c, kernel_size=1, padding="same"))
                if i < proj_depth - 1:
                    proj_layers.append(norm_cls(c))
                    proj_layers.append(nn.ReLU(inplace=True))
            self.projection_head = nn.Sequential(*proj_layers)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)
            deactivate_requires_grad(self.projection_head_momentum)
        else:
            self.projection_head = None
            self.projection_head_momentum = None

        pred_layers = []
        n_layers = self.cfg.predictor.n_layers
        for i in range(n_layers):
            if self.cfg.predictor.get("dw_sep_conv", False):
                pred_layers.append(nn.Conv2d(c, c, self.cfg.predictor.kernel_size, padding="same", groups=c))
                pred_layers.append(nn.Conv2d(c, c, kernel_size=1, padding="same"))
            else:
                pred_layers.append(nn.Conv2d(c, c, self.cfg.predictor.kernel_size, padding="same"))
            # No norm/ReLU on the LAST layer: the predictor must output raw (possibly negative)
            # features to match the target encoder's embeddings. A final ReLU would clamp the
            # prediction to >=0 and cripple the JEPA loss.
            if i < n_layers - 1:
                pred_layers.append(norm_cls(c))
                pred_layers.append(nn.ReLU(inplace=True))
        self.predictor = nn.Sequential(*pred_layers)

        self.criterion = F.smooth_l1_loss

    def forward(self, x):
        inp_bchw = x
        # step 1. Mask (coarse, at stride-32 patch level)
        context_mask_b1ff, target_mask_b1ff = self.mask(inp_bchw.shape[0], inp_bchw.device)
        sparse_encoder._cur_active = context_mask_b1ff  # (B, 1, f, f)
        active_b1hw = context_mask_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(
            self.downsample_raito, 3
        )
        masked_bchw = inp_bchw * active_b1hw

        # step 2. Encode the SPARSE trunk (0-8). Masked positions stay zeroed throughout.
        trunk_feat = self.backbone.forward_trunk(masked_bchw)  # (B, trunk_ch, f, f)

        # step 3. DENSIFY: fill masked (non-active) positions with the learned mask token.
        mask_tokens = self.mask_token.expand_as(trunk_feat)
        trunk_dense = torch.where(
            context_mask_b1ff.expand_as(trunk_feat), trunk_feat, mask_tokens.to(trunk_feat.dtype)
        )

        # step 4. DENSE tail (SESA -> SPPF -> C2PSA). No leakage now: the map is fully filled.
        feat = self.backbone.forward_tail(trunk_dense)  # (B, num_features, f, f)

        # step 4.5. Project (student projection head, if enabled).
        if self.projection_head is not None:
            feat = self.projection_head(feat)

        # step 5. Predict masked-region embeddings.
        z = self.predictor(feat)
        return z, context_mask_b1ff, target_mask_b1ff

    def forward_momentum(self, x):
        # EMA target encoder = full DENSE backbone (trunk + tail) on the unmasked image.
        z = self.backbone_momentum(x)
        if self.projection_head_momentum is not None:
            z = self.projection_head_momentum(z)
        return z.detach()


@hydra.main(version_base="1.2", config_path="configs/", config_name="ijepacnn_yolo_maritime.yaml")
def pretrain_yolo(cfg: DictConfig):
    main_pretrain(cfg, IJEPA_YOLO)


if __name__ == "__main__":
    pretrain_yolo()
