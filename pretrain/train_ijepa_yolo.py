# CNN-JEPA pretraining for the YOLO26-SEA backbone.
#
# Independent implementation of YOLO-JEPA pretraining.
# The trunk (layers 0-8) runs SPARSE, and the tail (9-11: SESA, SPPF, C2PSA) runs DENSE.
# The training loop, masking, and loss are self-contained.

import copy
import math
import os
import sys
from typing import Optional

import hydra
from omegaconf import DictConfig
import torch
import torch._dynamo
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from lightly.transforms.ijepa_transform import IJEPATransform

from pretrain.trainer_common import LightlyModel, LightlyModelMomentum, main_pretrain
import models.sparse_encoder as sparse_encoder
from pretrain.ijepa_mask import MultiBlockMask

# Import to register `yolo26_sea_backbone` in the timm model registry (used by the base
# LightlyModel via timm.create_model(cfg.backbone.name, ...)).
import models.yolo_backbone  # noqa: F401


class IJEPA_YOLO(LightlyModelMomentum):
    def __init__(self, cfg: DictConfig):
        # Build self.backbone (YOLO26SEABackbone) + self.backbone_momentum (dense deepcopy).
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
        self.variance_loss_enabled = bool(self.cfg.get("variance_loss", {}).get("enabled", False))
        self.variance_loss_weight = float(self.cfg.get("variance_loss", {}).get("weight", 1.0))

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
        compile_opt = perf.get("compile", False)
        if compile_opt and not hasattr(torch, "compile"):
            print("perf.compile requested but torch.compile unavailable (torch<2.0); skipping.", flush=True)
            return
        if compile_opt:
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

    def setup_transform(self):
        self.transform = IJEPATransform(self.input_size)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        self._setup_masking(self.input_size)

    def _setup_masking(self, input_size: int) -> None:
        """Derive the mask geometry from the input size."""
        self.input_size = input_size
        self.downsample_raito = self.backbone.get_downsample_ratio()
        self.fmap_h, self.fmap_w = self.input_size // self.downsample_raito, self.input_size // self.downsample_raito
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - self.cfg.mask_ratio))
        self.multi_block_mask = MultiBlockMask(
            input_size=self.input_size,
            patch_size=self.downsample_raito,
            **self.cfg.mask.mutli_block_kwargs
        )
        # Every resolution the sparse trunk produces (stride 2..32 -> fmap*16..fmap*1), plus
        # the input resolution (fmap*32) used to mask the image itself. forward() prefills the
        # sparse-mask cache with these once per step.
        self._prefill_sizes = tuple(
            self.fmap_h * (2 ** i) for i in range(int(math.log2(self.downsample_raito)) + 1)
        )

    def mask(self, x: torch.Tensor, generator=None):
        B = x.shape[0]
        device = x.device
        if self.cfg.mask.strategy == "mixed":
            if torch.rand(1) < self.cfg.mask.mixed_mutli_block_ratio:
                strategy = "multi-block"
            else:
                strategy = "random"
        else:
            strategy = self.cfg.mask.strategy
        if strategy == "random":
            h, w = self.fmap_h, self.fmap_w
            idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
            idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
            context_mask = torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)
            target_mask = context_mask.logical_not()
            return context_mask, target_mask   
        elif strategy == "variance-biased":
            h, w = self.fmap_h, self.fmap_w
            # 1. Convert input x to grayscale
            gray = 0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2] # (B, H, W)
            # 2. Unfold to patches of size 32x32
            patches = gray.unfold(1, self.downsample_raito, self.downsample_raito).unfold(2, self.downsample_raito, self.downsample_raito) # (B, h, w, p, p)
            patches = patches.contiguous().view(B, h, w, -1)
            # 3. Compute variance per patch
            variances = patches.var(dim=-1).view(B, h * w) # (B, h * w)
            # 4. Compute probabilities with bias temperature
            bias_temp = self.cfg.mask.get("variance_bias_temp", 2.0)
            probs = variances ** bias_temp + 1e-4
            probs = probs / probs.sum(dim=-1, keepdim=True)
            # 5. Weighted sampling without replacement
            idx = torch.multinomial(probs, num_samples=self.len_keep, replacement=False) # (B, len_keep)
            
            context_mask = torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)
            target_mask = context_mask.logical_not()
            return context_mask, target_mask
        elif strategy == "multi-block":
            context_mask, target_mask = self.multi_block_mask(B)
            context_mask = context_mask.unsqueeze(1).to(device, dtype=torch.bool)
            target_mask = target_mask.unsqueeze(1).to(device, dtype=torch.bool)
            return context_mask, target_mask

    def get_views_to_log_from_batch(self, batch):
        inp_bchw = batch[0]
        context_mask_b1ff, target_mask_b1ff = self.mask(inp_bchw)  # (B, 1, f, f)
        context_mask_b1hw = context_mask_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        target_mask_b1hw  =  target_mask_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        context_bchw = inp_bchw * context_mask_b1hw
        target_bchw = inp_bchw * target_mask_b1hw
        return [inp_bchw, context_bchw, target_bchw]

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
            if i < n_layers - 1:
                layers.append(norm_cls(c))
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        inp_bchw = x
        if self._channels_last:
            inp_bchw = inp_bchw.contiguous(memory_format=torch.channels_last)
        context_mask_b1ff, target_mask_b1ff = self.mask(inp_bchw)
        sparse_encoder.set_active(context_mask_b1ff, prefill_sizes=getattr(self, "_prefill_sizes", ()))
        active_b1hw = sparse_encoder._get_active_ex_or_ii(
            H=inp_bchw.shape[2], W=inp_bchw.shape[3], returning_active_ex=True
        )
        masked_bchw = inp_bchw * active_b1hw

        trunk_feat = self.backbone.forward_trunk(masked_bchw)  # (B, trunk_ch, f, f)

        mask_tokens = self.mask_token.expand_as(trunk_feat)
        trunk_dense = torch.where(
            context_mask_b1ff.expand_as(trunk_feat), trunk_feat, mask_tokens.to(trunk_feat.dtype)
        )

        feat = self.backbone.forward_tail(trunk_dense)  # (B, num_features, f, f)

        if self.projection_head is not None:
            feat = self.projection_head(feat)

        z = self.predictor(feat)

        if self.deep_supervision:
            z_trunk = self.predictor_trunk(trunk_dense)
            return {"trunk": z_trunk, "final": z}, context_mask_b1ff, target_mask_b1ff
        return z, context_mask_b1ff, target_mask_b1ff

    def forward_momentum(self, x):
        if self._channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        trunk_out = self.backbone_momentum.forward_trunk(x)
        final = self.backbone_momentum.forward_tail(trunk_out)
        if self.projection_head_momentum is not None:
            final = self.projection_head_momentum(final)
        if self.deep_supervision:
            return {"trunk": trunk_out.detach(), "final": final.detach()}
        return final.detach()

    @staticmethod
    @torch.no_grad()
    def _context_distance_weight(target_mask_b1ff):
        m = target_mask_b1ff.to(torch.float32)
        f = m.shape[-1]
        dist = torch.zeros_like(m)
        covered = m.clone()
        cur = m
        for d in range(1, f + 1):
            cur = F.max_pool2d(cur, kernel_size=3, stride=1, padding=1)
            newly = (cur > 0) & (covered == 0)
            dist = dist + d * newly.to(dist.dtype)
            covered = covered + newly.to(covered.dtype)
        dist = torch.where(covered > 0, dist, torch.full_like(dist, float(f)))
        return 1.0 / torch.sqrt(dist.clamp(min=1.0))

    def _lambda_eff(self):
        cl = self.cfg.get("context_loss", None)
        if cl is None or not cl.get("enabled", False):
            return 0.0
        lam = float(cl.get("lambda", 0.5))
        warm = int(cl.get("warmup_epochs", 0))
        return lam * min(1.0, (self.current_epoch + 1) / warm) if warm > 0 else lam

    def _jepa_level_loss(self, p, h, context_mask_b1ff, target_mask_b1ff, ctx_weight=None):
        per_pos = F.smooth_l1_loss(p, h, reduction='none').sum(axis=1, keepdim=True)  # (B,1,f,f)
        tgt = target_mask_b1ff.to(per_pos.dtype)
        loss_pred = per_pos.mul(tgt).sum() / (tgt.sum() + 1e-8)  # masked patches (original JEPA)
        loss_ctx = None
        cl = self.cfg.get("context_loss", None)
        if cl is not None and cl.get("enabled", False):
            w = ctx_weight if ctx_weight is not None else self._context_distance_weight(target_mask_b1ff)
            ctx = context_mask_b1ff.to(per_pos.dtype) * w
            loss_ctx = per_pos.mul(ctx).sum() / (ctx.sum() + 1e-8)
        return loss_pred, loss_ctx

    @torch.no_grad()
    def _log_feature_std(self, h, metric_label):
        feat = h["final"] if isinstance(h, dict) else h  # (B, C, f, f)
        feat = feat.float()
        std = feat.permute(1, 0, 2, 3).reshape(feat.shape[1], -1).std(dim=1).mean()
        self.log(f"{metric_label}/feature_std", std, on_epoch=True)

    def _variance_loss(self, x, eps=1e-4):
        if x.ndim == 4:
            # 1. Batch variance (on spatial mean)
            x_mean = x.mean(dim=(2, 3))  # (B, C)
            std_batch = torch.sqrt(x_mean.var(dim=0, unbiased=False) + eps)  # (C)
            loss_batch = torch.sum((std_batch - 1.0) ** 2)

            # 2. Spatial variance (across H, W for each image)
            std_spatial = torch.sqrt(x.var(dim=(2, 3), unbiased=False) + eps)  # (B, C)
            loss_spatial = torch.sum((std_spatial - 1.0) ** 2) / x.shape[0]  # mean over batch

            return loss_batch + loss_spatial
        else:
            std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
            return torch.sum((std - 1.0) ** 2)

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        if self.deep_supervision:
            # Deep Self-Supervision: sum the JEPA loss (masked + context) over all levels.
            x = batch[0]
            p_levels, context_mask_b1ff, target_mask_b1ff = self.forward(x)
            h_levels = self.forward_momentum(x)
            self._log_feature_std(h_levels, metric_label)
            lam = self._lambda_eff()
            cl = self.cfg.get("context_loss", None)
            ctx_w = None
            if cl is not None and cl.get("enabled", False):
                ctx_w = self._context_distance_weight(target_mask_b1ff)
            total = 0.0
            for name in p_levels:
                loss_pred, loss_ctx = self._jepa_level_loss(
                    p_levels[name], h_levels[name], context_mask_b1ff, target_mask_b1ff, ctx_weight=ctx_w)
                level_loss = loss_pred + (lam * loss_ctx if loss_ctx is not None else 0.0)
                if self.variance_loss_enabled:
                    var_loss = self._variance_loss(p_levels[name])
                    level_loss = level_loss + self.variance_loss_weight * var_loss
                    self.log(f"{metric_label}/var_loss_{name}", var_loss, on_epoch=True)
                total = total + level_loss
                self.log(f"{metric_label}/ijepa_loss_{name}", loss_pred, on_epoch=True)
                if loss_ctx is not None:
                    self.log(f"{metric_label}/ctx_loss_{name}", loss_ctx, on_epoch=True)
            self.log(f"{metric_label}/loss", total, on_epoch=True)
            return total
        else:
            x = batch[0]
            p, context_mask_b1ff, target_mask_b1ff = self.forward(x)
            h = self.forward_momentum(x)
            self._log_feature_std(h, metric_label)
            loss_pred, loss_ctx = self._jepa_level_loss(p, h, context_mask_b1ff, target_mask_b1ff)
            loss = loss_pred
            self.log(f"{metric_label}/ijepa_loss", loss_pred, on_epoch=True)
            if loss_ctx is not None:
                lam = self._lambda_eff()
                loss = loss_pred + lam * loss_ctx
                self.log(f"{metric_label}/ctx_loss", loss_ctx, on_epoch=True)
                self.log(f"{metric_label}/ctx_lambda", lam, on_epoch=True)
            if self.variance_loss_enabled:
                var_loss = self._variance_loss(p)
                loss = loss + self.variance_loss_weight * var_loss
                self.log(f"{metric_label}/var_loss", var_loss, on_epoch=True)
            self.log(f"{metric_label}/loss", loss, on_epoch=True)
            return loss


@hydra.main(version_base="1.2", config_path="configs/", config_name="ijepacnn_yolo_maritime.yaml")
def pretrain_yolo(cfg: DictConfig):
    main_pretrain(cfg, IJEPA_YOLO)


if __name__ == "__main__":
    pretrain_yolo()
