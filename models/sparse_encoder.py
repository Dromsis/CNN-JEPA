# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in under
# https://github.com/keyu-tian/SparK/blob/main/LICENSE ur.

import torch
import torch.nn as nn


_cur_active: torch.Tensor = None            # B1ff
# Per-step cache for the expanded masks: they only depend on (_cur_active, H, W), and every
# layer of a stage asks for the SAME resolution. Without it, the repeat_interleave (+ the
# nonzero of the gather variants) is recomputed for every conv/BN at every step. The cache is
# invalidated whenever `_cur_active` changes identity, so existing code that assigns
# `sparse_encoder._cur_active = mask` directly keeps working.
_active_cache: dict = {}
_active_cache_src: torch.Tensor = None


def set_active(mask: torch.Tensor, prefill_sizes=()):
    """Set the current context mask and optionally prefill the per-resolution cache.

    Prefilling turns every `_get_active_ex_or_ii` call inside the encoder into a cache HIT,
    which keeps torch.compile'd regions free of cache-mutation side effects.
    """
    global _cur_active, _active_cache_src
    _cur_active = mask
    _active_cache.clear()
    _active_cache_src = mask
    for hw in prefill_sizes:
        _get_active_ex_or_ii(H=hw, W=hw, returning_active_ex=True)


def _get_active_ex_or_ii(H, W, returning_active_ex=True):
    global _active_cache_src
    if _active_cache_src is not _cur_active:
        _active_cache.clear()
        _active_cache_src = _cur_active
    key = (H, W, returning_active_ex)
    out = _active_cache.get(key)
    if out is None:
        h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
        active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
        out = active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi
        _active_cache[key] = out
    return out


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)

    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`

    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw


def sp_bn_forward_dense(self, x: torch.Tensor):
    """Masked BatchNorm over the active positions, computed DENSE (no gather/scatter).

    Numerically equivalent to `sp_bn_forward` (gather active pixels into (N, C), BatchNorm1d,
    scatter back into zeros), but without `nonzero()`, whose data-dependent output size forces
    a GPU->CPU sync at every BN of every step. Stats are computed in fp32 (matches autocast's
    handling of batch_norm); masked positions are zeroed in the output exactly like the
    scatter-into-zeros of the gather version. Equivalence (fwd, grads, running stats, eval) is
    checked in tests/test_sparse_bn_equivalence.py.
    """
    active = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)  # (B,1,H,W)
    xf = x.float()
    m = active.to(xf.dtype)

    if self.training or not self.track_running_stats:
        n = m.sum()
        mean = (xf * m).sum(dim=(0, 2, 3)) / n                              # (C,)
        var = ((xf - mean[None, :, None, None]) * m).pow(2).sum(dim=(0, 2, 3)) / n  # biased, like BN
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
            momentum = self.momentum
            if momentum is None:  # cumulative moving average, per _BatchNorm semantics
                momentum = 1.0 / float(self.num_batches_tracked)
            with torch.no_grad():
                var_unbiased = var * (n / (n - 1.0).clamp(min=1.0))
                self.running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
                self.running_var.mul_(1.0 - momentum).add_(var_unbiased, alpha=momentum)
    else:
        mean = self.running_mean.float()
        var = self.running_var.float()

    scale = torch.rsqrt(var + self.eps)
    if self.affine:
        scale = scale * self.weight.float()
        shift = self.bias.float() - mean * scale
    else:
        shift = -mean * scale
    out = xf * scale[None, :, None, None] + shift[None, :, None, None]
    out = out * m
    return out.to(x.dtype)


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward_dense   # dense masked BN: no nonzero/gather/scatter (see above)


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    # Keeps the gather implementation: SyncBatchNorm.forward carries the cross-rank stats
    # reduction (DDP), which the dense rewrite does not reimplement.
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details

class SparseConvNeXtLayerNorm(nn.LayerNorm):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse
    
    def forward(self, x):
        if x.ndim == 4: # BHWC or BCHW
            if self.data_format == "channels_last": # BHWC
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[1], W=x.shape[2], returning_active_ex=False)
                    nc = x[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
    
                    x = torch.zeros_like(x)
                    x[ii] = nc.to(x.dtype)
                    return x
                else:
                    return super(SparseConvNeXtLayerNorm, self).forward(x)
            else:       # channels_first, BCHW
                if self.sparse:
                    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=False)
                    bhwc = x.permute(0, 2, 3, 1)
                    nc = bhwc[ii]
                    nc = super(SparseConvNeXtLayerNorm, self).forward(nc)
                
                    x = torch.zeros_like(bhwc)
                    x[ii] = nc.to(x.dtype)
                    return x.permute(0, 3, 1, 2)
                else:
                    u = x.mean(1, keepdim=True)
                    s = (x - u).pow(2).mean(1, keepdim=True)
                    x = (x - u) / torch.sqrt(s + self.eps)
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                    return x
        else:           # BLC or BC
            if self.sparse:
                raise NotImplementedError
            else:
                return super(SparseConvNeXtLayerNorm, self).forward(x)

    def __repr__(self):
        return super(SparseConvNeXtLayerNorm, self).__repr__()[:-1] + f', ch={self.data_format.split("_")[-1]}, sp={self.sparse})'


class SparseConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, sparse=True, ks=7):
        super().__init__()
        # Lazy import: keeps `sparse_encoder` importable with torch alone (unit tests); timm
        # is only needed when a ConvNeXt block is actually built.
        from timm.models.layers import DropPath
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=ks//2, groups=dim)  # depthwise conv
        self.norm = SparseConvNeXtLayerNorm(dim, eps=1e-6, sparse=sparse)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path: nn.Module = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self._sparse = sparse
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)            # GELU(0) == (0), so there is no need to mask x (no need to `x *= _get_active_ex_or_ii`)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        if self.sparse:
            x *= _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)
        
        x = input + self.drop_path(x)
        return x
    
    def __repr__(self):
        return super(SparseConvNeXtBlock, self).__repr__()[:-1] + f', sp={self.sparse})'
    
    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        self._sparse = value
        self.norm.sparse = value


def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
    oup = m
    if isinstance(m, nn.Conv2d):
        m: nn.Conv2d
        bias = m.bias is not None
        oup = SparseConv2d(
            m.in_channels, m.out_channels,
            kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
            dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
        )
        oup.weight.data.copy_(m.weight.data)
        if bias:
            oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, nn.MaxPool2d):
        m: nn.MaxPool2d
        oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
    elif isinstance(m, nn.AvgPool2d):
        m: nn.AvgPool2d
        oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        m: nn.BatchNorm2d
        oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
        oup.running_mean.data.copy_(m.running_mean.data)
        oup.running_var.data.copy_(m.running_var.data)
        oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
        if hasattr(m, "qconfig"):
            oup.qconfig = m.qconfig
    elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
        m: nn.LayerNorm
        oup = SparseConvNeXtLayerNorm(m.weight.shape[0], eps=m.eps)
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, (nn.Conv1d,)):
        raise NotImplementedError
    
    for name, child in m.named_children():
        oup.add_module(name, dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
    del m
    return oup
