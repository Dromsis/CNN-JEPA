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
import math

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from pretrain.trainer_common import LightlyModel, LightlyModelMomentum, main_pretrain
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

        # Final-level predictor (encoder output / layer-11 dim).
        self.predictor = self._build_predictor(c, norm_cls)

        # Deep Self-Supervision (V-JEPA 2.1): also supervise the TRUNK output (layer-8) level so
        # local information is pushed toward the final layers. Aux predictor at trunk_channels;
        # its target is the EMA encoder's trunk output. Off unless cfg enables it.
        self.deep_supervision = bool(self.cfg.get("deep_supervision", {}).get("enabled", False)) \
            if self.cfg.get("deep_supervision", None) is not None else False
        if self.deep_supervision:
            self.predictor_trunk = self._build_predictor(self.backbone.trunk_channels, norm_cls)

        self.criterion = F.smooth_l1_loss

        # Cached (ema_params, src_params) lists for the fused EMA update; built lazily at the
        # first training step (i.e. after Lightning has moved the module to its device).
        self._ema_params = None

        self._apply_perf_options()

    def _apply_perf_options(self):
        """cfg.perf: channels_last + torch.compile. Both off when the block is absent."""
        perf = self.cfg.get("perf", None)
        perf = dict(perf) if perf is not None else {}

        self._channels_last = bool(perf.get("channels_last", False))
        if self._channels_last:
            modules = [self.backbone, self.backbone_momentum, self.predictor]
            if self.deep_supervision:
                modules.append(self.predictor_trunk)
            if self.projection_head is not None:
                modules += [self.projection_head, self.projection_head_momentum]
            for mod in modules:
                mod.to(memory_format=torch.channels_last)

        # compile: false | "dense" | "all"/true
        #   "dense" -> teacher + tail + predictors (no global state, always safe)
        #   "all"   -> also the sparse student trunk. Its conv/BN wrappers READ the mask cache,
        #              which forward() prefills per step, so compiled regions never mutate it.
        #              On torch 2.0.x check TORCH_LOGS=recompiles once; fall back to "dense"
        #              if graphs churn.
        compile_opt = perf.get("compile", False)
        if compile_opt and not hasattr(torch, "compile"):
            print("perf.compile requested but torch.compile unavailable (torch<2.0); skipping.", flush=True)
            return
        if compile_opt:
            import torch._dynamo
            # A dynamo/inductor failure must not kill a multi-day run: log + fall back to eager.
            torch._dynamo.config.suppress_errors = True
            self.backbone.forward_tail = torch.compile(self.backbone.forward_tail)
            self.backbone_momentum.forward_trunk = torch.compile(self.backbone_momentum.forward_trunk)
            self.backbone_momentum.forward_tail = torch.compile(self.backbone_momentum.forward_tail)
            self.predictor.forward = torch.compile(self.predictor.forward)
            if self.deep_supervision:
                self.predictor_trunk.forward = torch.compile(self.predictor_trunk.forward)
            if self.projection_head is not None:
                self.projection_head.forward = torch.compile(self.projection_head.forward)
                self.projection_head_momentum.forward = torch.compile(self.projection_head_momentum.forward)
            if compile_opt is True or compile_opt == "all":
                self.backbone.forward_trunk = torch.compile(self.backbone.forward_trunk)

    def _setup_masking(self, input_size: int) -> None:
        super()._setup_masking(input_size)
        # Every resolution the sparse trunk produces (stride 2..32 -> fmap*16..fmap*1), plus
        # the input resolution (fmap*32) used to mask the image itself. forward() prefills the
        # sparse-mask cache with these once per step.
        self._prefill_sizes = tuple(
            self.fmap_h * (2 ** i) for i in range(int(math.log2(self.downsample_raito)) + 1)
        )

    @torch.no_grad()
    def _ema_update(self, m: float):
        """Fused EMA of the teacher: ema = ema*m + src*(1-m), same math as lightly's
        update_momentum but batched with torch._foreach_* (2 kernel launches instead of 2 per
        parameter, i.e. several hundred per step)."""
        if self._ema_params is None:
            self._ema_params = (
                list(self.backbone_momentum.parameters()),
                list(self.backbone.parameters()),
            )
        ema, src = self._ema_params
        m = float(m)  # cosine_schedule returns a numpy scalar; _foreach_* wants a Python number
        torch._foreach_mul_(ema, m)
        torch._foreach_add_(ema, src, alpha=1.0 - m)

    def training_step(self, batch, batch_idx):
        # Replicates LightlyModelMomentum.training_step with the fused EMA (skip over the
        # parent on purpose; keep in sync with trainer_common if that method changes).
        momentum = cosine_schedule(self.current_epoch, self.cfg.trainer.max_epochs, 0.996, 1)
        self._ema_update(momentum)
        if self.projection_head_momentum is not None:
            update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        return LightlyModel.training_step(self, batch, batch_idx)

    def _build_predictor(self, c, norm_cls):
        layers = []
        n_layers = self.cfg.predictor.n_layers
        for i in range(n_layers):
            if self.cfg.predictor.get("dw_sep_conv", False):
                layers.append(nn.Conv2d(c, c, self.cfg.predictor.kernel_size, padding="same", groups=c))
                layers.append(nn.Conv2d(c, c, kernel_size=1, padding="same"))
            else:
                layers.append(nn.Conv2d(c, c, self.cfg.predictor.kernel_size, padding="same"))
            # No norm/ReLU on the LAST layer: the predictor must output raw (possibly negative)
            # features to match the target encoder's embeddings. A final ReLU would clamp the
            # prediction to >=0 and cripple the JEPA loss.
            if i < n_layers - 1:
                layers.append(norm_cls(c))
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        inp_bchw = x
        if self._channels_last:
            inp_bchw = inp_bchw.contiguous(memory_format=torch.channels_last)
        # step 1. Mask (coarse, at stride-32 patch level). set_active prefills the expanded
        # masks for every trunk resolution, so all lookups inside the (possibly compiled)
        # sparse trunk are cache hits.
        context_mask_b1ff, target_mask_b1ff = self.mask(inp_bchw.shape[0], inp_bchw.device)
        sparse_encoder.set_active(context_mask_b1ff, prefill_sizes=getattr(self, "_prefill_sizes", ()))
        active_b1hw = sparse_encoder._get_active_ex_or_ii(
            H=inp_bchw.shape[2], W=inp_bchw.shape[3], returning_active_ex=True
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

        if self.deep_supervision:
            # Aux prediction at the trunk (layer-8) level, from the densified trunk features.
            z_trunk = self.predictor_trunk(trunk_dense)
            return {"trunk": z_trunk, "final": z}, context_mask_b1ff, target_mask_b1ff
        return z, context_mask_b1ff, target_mask_b1ff

    def forward_momentum(self, x):
        # EMA target encoder = full DENSE backbone on the unmasked image. Always composed as
        # trunk -> tail so both halves hit the (possibly compiled) method wrappers.
        if self._channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        trunk_out = self.backbone_momentum.forward_trunk(x)
        final = self.backbone_momentum.forward_tail(trunk_out)
        if self.projection_head_momentum is not None:
            final = self.projection_head_momentum(final)
        if self.deep_supervision:
            return {"trunk": trunk_out.detach(), "final": final.detach()}
        return final.detach()

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        # Single-level path (incl. the V-JEPA 2.1 context loss) is handled by the parent.
        if not self.deep_supervision:
            return super().train_val_step(batch, batch_idx, metric_label)

        # Deep Self-Supervision: sum the JEPA loss (masked + context) over all levels.
        x = batch[0]
        p_levels, context_mask_b1ff, target_mask_b1ff = self.forward(x)
        h_levels = self.forward_momentum(x)
        lam = self._lambda_eff()
        # The context distance weight only depends on the masks: compute it once, not per level.
        cl = self.cfg.get("context_loss", None)
        ctx_w = None
        if cl is not None and cl.get("enabled", False):
            ctx_w = self._context_distance_weight(target_mask_b1ff)
        total = 0.0
        for name in p_levels:
            loss_pred, loss_ctx = self._jepa_level_loss(
                p_levels[name], h_levels[name], context_mask_b1ff, target_mask_b1ff, ctx_weight=ctx_w)
            level_loss = loss_pred + (lam * loss_ctx if loss_ctx is not None else 0.0)
            total = total + level_loss
            self.log(f"{metric_label}/ijepa_loss_{name}", loss_pred, on_epoch=True)
            if loss_ctx is not None:
                self.log(f"{metric_label}/ctx_loss_{name}", loss_ctx, on_epoch=True)
        self.log(f"{metric_label}/loss", total, on_epoch=True)
        return total


@hydra.main(version_base="1.2", config_path="configs/", config_name="ijepacnn_yolo_maritime.yaml")
def pretrain_yolo(cfg: DictConfig):
    main_pretrain(cfg, IJEPA_YOLO)


if __name__ == "__main__":
    pretrain_yolo()
