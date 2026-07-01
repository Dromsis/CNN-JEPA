"""Smoke validation for the L40S image: GPU kernels + framework + project imports."""
import sys

import torch

print("torch", torch.__version__, "| cuda", torch.version.cuda, flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "CUDA not available in container"
print("device:", torch.cuda.get_device_name(0), flush=True)
print("capability: sm_" + "".join(map(str, torch.cuda.get_device_capability(0))), flush=True)

# Actually execute a kernel on the GPU (this is what fails on an incompatible arch).
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum().item()
torch.cuda.synchronize()
print("GPU matmul OK ->", round(y, 1), flush=True)

import pytorch_lightning as pl
import lightly
import timm

print("pl", pl.__version__, "| lightly", lightly.__version__, "| timm", timm.__version__, flush=True)

# Exercises the patched lightning_lite/lightning_fabric import + lightly/timm + the model module.
import pretrain.train_ijepa_yolo as m

print("project import OK ->", m.IJEPA_YOLO.__name__, flush=True)
print("ALL_OK", flush=True)
sys.exit(0)
