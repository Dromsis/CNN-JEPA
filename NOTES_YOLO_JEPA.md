# CNN-JEPA pretraining for the YOLO26-SEA backbone

Self-supervised pretraining of the **YOLO26-SEA backbone** (long-distance maritime ship
detection) with CNN-JEPA, before supervised detection finetuning in the `yolo_sea_homemade`
repo.

## The idea (1 paragraph)

JEPA pretrains the **backbone only** (layers 0-11 of `yolo26-sea.yaml`) on a large pool of
**unlabeled** drone imagery (~65k labeled + ~300k unlabeled). The backbone learns the visual
statistics of the sea/horizon/hull domain that the ~65k labeled images alone can't cover. The
resulting weights initialize the backbone for supervised finetuning of the full YOLO26-SEA
model (backbone + BiFPN neck + head + direction branch). The neck/head are NOT pretrained
(no SSL signal for the task-specific detection neck); they train from scratch on the labels.

## Architecture: sparse trunk + dense tail

The CNN-JEPA masking uses SparK-style **sparse convolutions**: masked image patches must not
leak into visible ones through the convolutions. This works automatically for any module built
from standard `nn.Conv2d / BatchNorm2d / MaxPool2d` (see `models/sparse_encoder.dense_model_to_sparse`).

The YOLO26-SEA backbone splits cleanly at layer 8:

```
0-8   Conv + C3k2          pure conv  -> SPARSE (dense_model_to_sparse, automatic)
--- densify (fill masked positions with a learned mask token) ---
9     SESA   (SENetV2+SimAM)  global stats  ┐
10    SPPF   (maxpool agg.)   global stats  ├-> DENSE (run normally, no leakage)
11    C2PSA  (self-attention) global mixing ┘
--- predictor -> predict masked-region embeddings of the EMA target encoder ---
12+   BiFPN neck + head     NOT pretrained (supervised finetune only)
```

SESA/SPPF/C2PSA mix information across **all** spatial positions, so they can't run sparse
without leaking. Instead we densify right after the trunk and run them dense. Bonus: this also
matches inference (no mask at finetune time -> SESA always sees a dense map).

## Files added on branch `feat/yolo26-jepa-pretrain`

- `models/yolo_backbone.py` — `YOLO26SEABackbone`: builds layers 0-11 from `yolo26-sea.yaml`
  via Ultralytics, splits into `trunk` (0-8) / `tail` (9-11), exposes `get_downsample_ratio()`,
  `trunk_channels`, `num_features`, `forward_trunk/forward_tail/forward`. Registered in timm as
  `yolo26_sea_backbone`.
- `pretrain/train_ijepa_yolo.py` — `IJEPA_YOLO`: subclass of `IJEPA_CNN`. Sparse-converts only
  the trunk, sizes the mask token at `trunk_channels`, and overrides `forward` to densify
  **between** trunk and tail. Masking / EMA target / loss / training loop are inherited.
- `pretrain/configs/ijepacnn_yolo_maritime.yaml` — config (backbone = yolo26_sea_backbone, 640
  input, scale n).

## Remaining integration before a run (NOT done yet)

1. **Ultralytics import path.** `models/yolo_backbone.py` needs the vendored `ultralytics`
   package importable. Either set `backbone.kwargs.yolo_repo_path` (already in the config) or
   `export YOLO_SEA_REPO=/home/dromsis/Documents/yolo_sea_homemade`.

2. **Maritime dataset wiring.** `pretrain/trainer_common.py::setup()` hardcodes datasets in
   per-name dicts. Add a `"maritime"` entry:
   - `dataset_classes["maritime"] = torchvision.datasets.ImageFolder` (or the HDF5 variant)
   - `train/val_dataset_kwargs["maritime"] = dict(root=<path to unlabeled images>)`
   - `input_sizes["maritime"] = 640`
   - `num_classes["maritime"] = 1` (dummy; data is unlabeled)
   ImageFolder needs at least one class subdir; put all unlabeled images under e.g.
   `images/unlabeled/`.

3. **Online linear-probe benchmark.** It assumes labels and will fail / be meaningless on
   unlabeled data. Guard it: set `on_validation_epoch_end` to skip, or wrap `run_benchmarks`
   (it already catches exceptions and logs `lin_top1=0.0`, so a run won't crash — but the
   metric is noise). For real eval, measure on the DOWNSTREAM detection finetune instead.

4. **Input size plumbing.** `MultiBlockMask` / `len_keep` derive from `input_size` (640) and
   `downsample_ratio` (32) -> a 20x20 mask grid (400 patches). Confirmed consistent once the
   maritime `input_size=640` entry exists.

## Performance (L40S, 640px)

Baseline measured: bs 64 @ ~3.0 it/s (~192 img/s, ~13 effective TFLOPS — GPU busy but
launch/sync-bound, not compute-bound). Changes applied:

- **Dense masked BatchNorm** (`models/sparse_encoder.sp_bn_forward_dense`): the SparK
  gather->BN1d->scatter did a `nonzero()` (= forced GPU->CPU sync) at EVERY BN of every step.
  Rewritten as dense masked mean/var — numerically identical (fwd, grads, running stats):
  `PYTHONPATH=. python tests/test_sparse_bn_equivalence.py`. SyncBN (DDP) keeps the gather path.
- **Per-step mask cache** (`sparse_encoder.set_active` + `_get_active_ex_or_ii`): the expanded
  masks were recomputed for every conv/BN; now computed once per resolution per step.
  `IJEPA_YOLO.forward` prefills all trunk resolutions so compiled regions never mutate the cache.
- **`perf:` config block** (`ijepacnn_yolo_maritime.yaml`): `channels_last` (NHWC tensor-core
  convs) and `compile` (false | "dense" = teacher/tail/predictors | "all" = + sparse trunk).
  `torch._dynamo.config.suppress_errors=True` -> a compile failure falls back to eager instead
  of killing the run. On torch 2.0.x, check `TORCH_LOGS=recompiles` once; if graphs churn, use
  "dense". Compile warmup is a few minutes on the first step.
- **Fused EMA** (`IJEPA_YOLO._ema_update`): `torch._foreach_*` instead of 2 kernel launches per
  parameter; same math as lightly's `update_momentum`.
- **`cudnn.benchmark = True`** (trainer_common): input size is fixed.
- **bs 16 -> 64 default** in the config (bs 128 = the validated ImageNet recipe, same lr).

To measure the model step alone (no dataloader) and get a CUDA op breakdown on the L40S:
```bash
PYTHONPATH=. python scripts/profile_step.py                     # current config
PYTHONPATH=. python scripts/profile_step.py perf.compile=false  # any hydra override
```
Compare its img/s against the real loop to tell model-step vs data-pipeline bottlenecks.

## How to run (after the steps above)

```bash
cd /home/dromsis/Documents/CNN-JEPA
export YOLO_SEA_REPO=/home/dromsis/Documents/yolo_sea_homemade   # if not using yolo_repo_path
PYTHONPATH=. python pretrain/train_ijepa_yolo.py --config-name ijepacnn_yolo_maritime.yaml
```

## Export to the detection model

After pretraining, load `self.backbone` weights (layers 0-11) into the YOLO26-SEA detection
model and finetune the full model on the 65k labeled images. Map the wrapper's
`trunk`/`tail` submodules back to the detection model's `model[0..11]` indices when copying
state_dict. Then compare downstream mAP@0.5 / mAP@0.5:0.95 / direction-MAE against the
existing YOLO26-SEA baseline (ImageNet/COCO init).

## What is verified vs not

- Design / wiring: written to match the existing `IJEPA_CNN` contracts.
- NOT yet run end-to-end (the dataset + the `maritime` setup() entry are still needed). Treat
  the first run as a smoke test: check channel inference in `YOLO26SEABackbone._infer_channels`
  and one forward/backward step before launching the full schedule.
```
