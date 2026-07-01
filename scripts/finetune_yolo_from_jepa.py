"""Finetune YOLO26-SEA detection with a CNN-JEPA pretrained backbone.

The pretrain LightningModule (IJEPA_YOLO) stores the backbone split as:
    backbone.trunk.{0..8}  -> detection model.{0..8}
    backbone.tail.{0..2}   -> detection model.{9,10,11}
plus a DENSE EMA copy under `backbone_momentum.*` with the same sub-structure. We use the
dense EMA weights (their trunk is never sparse-converted, so keys map cleanly), remap them
onto model.{0..11}, load them into a fresh YOLO26-SEA detector (head stays random), then
train. Run from the repo root with the venv python, e.g.:

    YOLO_SEA_REPO=/home/shadeform/yolo_sea PYTHONPATH=. \
      ~/venv-jepa/bin/python scripts/finetune_yolo_from_jepa.py \
        --ckpt ~/CNN-JEPA/artifacts/.../version_0/last.ckpt \
        --yaml /home/shadeform/yolo_sea/ultralytics/cfg/models/26/yolo26-sea.yaml \
        --scale s --data /home/shadeform/maritime_det.yaml --epochs 100 --imgsz 640 --batch 32
"""
import argparse
import os
import sys

import torch

TRUNK_LAST_LAYER = 8  # trunk = model.0..8, tail = model.9..11


def _ensure_ultralytics(yolo_repo_path):
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    cand = yolo_repo_path or os.environ.get("YOLO_SEA_REPO")
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)
    import ultralytics  # noqa: F401


def remap_backbone(state_dict, src_prefix):
    """trunk.i.* -> model.i.* ; tail.j.* -> model.(9+j).*"""
    out = {}
    for k, v in state_dict.items():
        if not k.startswith(src_prefix + "."):
            continue
        sub = k[len(src_prefix) + 1:]
        if sub.startswith("trunk."):
            idx, rest = sub[len("trunk."):].split(".", 1)
            out[f"model.{int(idx)}.{rest}"] = v
        elif sub.startswith("tail."):
            idx, rest = sub[len("tail."):].split(".", 1)
            out[f"model.{TRUNK_LAST_LAYER + 1 + int(idx)}.{rest}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrain last.ckpt")
    ap.add_argument("--yaml", required=True, help="yolo26-sea.yaml")
    ap.add_argument("--scale", default="s")
    ap.add_argument("--data", required=True, help="detection data.yaml (images+labels+names)")
    ap.add_argument("--src-prefix", default="backbone_momentum",
                    help="backbone_momentum (dense EMA, recommended) or backbone")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--yolo-repo", default=None)
    args = ap.parse_args()

    _ensure_ultralytics(args.yolo_repo)
    from ultralytics import YOLO

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    available = sorted({k.split(".")[0] for k in sd})
    print("Top-level groups in ckpt:", available)

    remapped = remap_backbone(sd, args.src_prefix)
    if not remapped:
        raise SystemExit(
            f"No keys matched prefix '{args.src_prefix}'. Pick one of: {available}")

    # Fresh detector at the requested scale; load the pretrained backbone into model.0..11.
    model = YOLO(args.yaml)
    if args.scale:
        # YOLO() infers scale from filename; force it explicitly when loading a bare yaml.
        model.model.yaml["scale"] = args.scale
    missing, unexpected = model.model.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len(unexpected)
    print(f"Backbone keys remapped: {len(remapped)} | loaded: {loaded} | "
          f"unexpected: {len(unexpected)} | (head/other left random: {len(missing)})")
    if loaded == 0:
        raise SystemExit("0 keys loaded -> key mismatch; inspect names before training.")

    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)


if __name__ == "__main__":
    main()
