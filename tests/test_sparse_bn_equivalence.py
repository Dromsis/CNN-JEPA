"""Numerical equivalence of the dense masked BatchNorm vs the original gather/scatter one.

The dense rewrite (models/sparse_encoder.sp_bn_forward_dense) replaces SparK's
gather -> BatchNorm1d -> scatter (which costs one GPU->CPU sync per BN via nonzero()).
This test checks, in fp32 on CPU:
  1. training forward output       (vs gather+BN1d reference)
  2. gradients wrt input / weight / bias
  3. running_mean / running_var / num_batches_tracked updates
  4. eval-mode forward output
  5. all-active mask == plain nn.BatchNorm2d

Run:  PYTHONPATH=. python tests/test_sparse_bn_equivalence.py
Only needs torch (sparse_encoder imports timm lazily).
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "sparse_encoder", os.path.join(REPO, "models", "sparse_encoder.py")
)
sparse_encoder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sparse_encoder)

ATOL, RTOL = 1e-5, 1e-4


def reference_gather_bn(bn1d: nn.BatchNorm1d, x: torch.Tensor, active_ex: torch.Tensor):
    """The original SparK sp_bn_forward, verbatim (gather -> BN1d -> scatter into zeros)."""
    ii = active_ex.squeeze(1).nonzero(as_tuple=True)
    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]
    nc = bn1d(nc)
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    return bchw.permute(0, 3, 1, 2)


def make_mask(B, f, keep_ratio=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.rand(B, f * f, generator=g).argsort(dim=1)[:, : max(1, int(f * f * keep_ratio))]
    m = torch.zeros(B, f * f, dtype=torch.bool).scatter_(1, idx, True)
    return m.view(B, 1, f, f)


def check(name, a, b):
    ok = torch.allclose(a, b, atol=ATOL, rtol=RTOL)
    max_diff = (a - b).abs().max().item() if a.shape == b.shape else float("nan")
    print(f"  {'OK ' if ok else 'FAIL'} {name:34s} max|diff|={max_diff:.3e}")
    assert ok, f"{name} differs (max abs diff {max_diff})"


def test_masked_vs_gather():
    torch.manual_seed(42)
    B, C, f, k = 4, 8, 5, 4  # H = W = f*k = 20
    H = f * k
    mask = make_mask(B, f)
    sparse_encoder.set_active(mask, prefill_sizes=(H,))
    active_ex = sparse_encoder._get_active_ex_or_ii(H, H, returning_active_ex=True)

    momentum, eps = 0.03, 1e-3  # ultralytics BN settings
    sp = sparse_encoder.SparseBatchNorm2d(C, eps=eps, momentum=momentum)
    ref = nn.BatchNorm1d(C, eps=eps, momentum=momentum)
    w = torch.randn(C) * 0.5 + 1.0
    b = torch.randn(C) * 0.1
    rm = torch.randn(C) * 0.2
    rv = torch.rand(C) + 0.5
    with torch.no_grad():
        for mdl in (sp, ref):
            mdl.weight.copy_(w)
            mdl.bias.copy_(b)
            mdl.running_mean.copy_(rm)
            mdl.running_var.copy_(rv)
    sp.train(), ref.train()

    x = torch.randn(B, C, H, H)
    x = x * active_ex  # BN input comes from a masked conv: inactive positions are zero
    x_sp = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)

    print("training forward + backward:")
    y_sp = sp(x_sp)
    y_ref = reference_gather_bn(ref, x_ref, active_ex)
    check("forward (train)", y_sp, y_ref)

    up = torch.randn_like(y_sp)
    y_sp.backward(up)
    y_ref.backward(up)
    check("grad x", x_sp.grad, x_ref.grad)
    check("grad weight", sp.weight.grad, ref.weight.grad)
    check("grad bias", sp.bias.grad, ref.bias.grad)
    check("running_mean", sp.running_mean, ref.running_mean)
    check("running_var", sp.running_var, ref.running_var)
    assert int(sp.num_batches_tracked) == int(ref.num_batches_tracked) == 1
    print("  OK  num_batches_tracked")

    print("eval forward:")
    sp.eval(), ref.eval()
    with torch.no_grad():
        check("forward (eval)", sp(x), reference_gather_bn(ref, x, active_ex))


def test_all_active_equals_bn2d():
    torch.manual_seed(7)
    B, C, H = 3, 6, 16
    sparse_encoder.set_active(torch.ones(B, 1, H, H, dtype=torch.bool))

    sp = sparse_encoder.SparseBatchNorm2d(C, eps=1e-3, momentum=0.03)
    ref = nn.BatchNorm2d(C, eps=1e-3, momentum=0.03)
    with torch.no_grad():
        ref.weight.copy_(sp.weight)
        ref.bias.copy_(sp.bias)
    sp.train(), ref.train()

    x = torch.randn(B, C, H, H)
    print("all-active mask vs nn.BatchNorm2d:")
    check("forward (train)", sp(x), ref(x))
    check("running_mean", sp.running_mean, ref.running_mean)
    check("running_var", sp.running_var, ref.running_var)


def test_cache_identity_reset():
    """The lazy cache must reset when _cur_active is swapped by direct assignment."""
    B, f = 2, 4
    m1, m2 = make_mask(B, f, seed=1), make_mask(B, f, seed=2)
    sparse_encoder._cur_active = m1
    a1 = sparse_encoder._get_active_ex_or_ii(f, f)
    assert sparse_encoder._get_active_ex_or_ii(f, f) is a1, "same mask should be a cache hit"
    sparse_encoder._cur_active = m2
    a2 = sparse_encoder._get_active_ex_or_ii(f, f)
    assert a2 is not a1 and torch.equal(a2, m2), "new mask must invalidate the cache"
    print("cache identity reset: OK")


if __name__ == "__main__":
    test_masked_vs_gather()
    test_all_active_equals_bn2d()
    test_cache_identity_reset()
    print("\nALL_OK")
    sys.exit(0)
