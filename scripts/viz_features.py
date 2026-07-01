"""Visualize the CNN-JEPA pretrained backbone's feature maps on an input image.

Loads the DENSE EMA backbone (backbone_momentum.*) from a pretrain checkpoint into a fresh
YOLO26SEABackbone, runs one image through it and saves:
  - the mean-activation heatmap of the final feature map (layer 11, stride 32)
  - a PCA->RGB map (top-3 PCA components of the per-position feature vectors) overlaid on the
    image -> objects that the SSL model learned to separate show up as distinct color regions.

Run with the venv python + ultralytics importable, e.g.:
    YOLO_SEA_REPO=/home/shadeform/yolo_sea PYTHONPATH=. \
      ~/venv-jepa/bin/python scripts/viz_features.py \
        --ckpt .../epoch=11-step=22668.ckpt \
        --yaml /home/shadeform/yolo_sea/ultralytics/cfg/models/26/yolo26-sea.yaml \
        --scale s --image /data/combined/images/val/<some>.jpg --out /tmp/feat.png
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


def _ensure_ultralytics(repo):
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    cand = repo or os.environ.get("YOLO_SEA_REPO")
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)
    import ultralytics  # noqa: F401


def pca_rgb(fmap):  # fmap: (C, H, W) tensor
    c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).T.float().cpu().numpy()  # (HW, C)
    x = x - x.mean(0, keepdims=True)
    # top-3 principal components via SVD
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:3].T  # (HW, 3)
    proj = (proj - proj.min(0)) / (proj.ptp(0) + 1e-8)
    return proj.reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--scale", default="s")
    ap.add_argument("--image", required=True)
    ap.add_argument("--src-prefix", default="backbone_momentum")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--layer", default="trunk8",
                    help="which captured layer to visualize, e.g. trunk4/trunk6/trunk8. "
                         "Prefer a TRUNK layer (tail mixes globally). Shallow = higher res.")
    ap.add_argument("--out", default="feat.png")
    ap.add_argument("--yolo-repo", default=None)
    args = ap.parse_args()

    _ensure_ultralytics(args.yolo_repo)
    from models.yolo_backbone import YOLO26SEABackbone

    backbone = YOLO26SEABackbone(yaml_path=args.yaml, scale=args.scale,
                                 yolo_repo_path=args.yolo_repo)
    # checkpoint keys: backbone_momentum.trunk.* / .tail.* -> strip prefix -> trunk.*/tail.*
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    p = args.src_prefix + "."
    remapped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    print(f"loaded {len(remapped) - len(unexpected)}/{len(remapped)} backbone tensors "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")
    backbone.eval()

    # preprocess (match the ImageNet-style normalization used by lightly transforms)
    img = Image.open(args.image).convert("RGB").resize((args.imgsz, args.imgsz))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()

    # Capture every layer output. The TRUNK (0-8) keeps spatial locality; the TAIL
    # (SESA/SPPF/C2PSA, 9-11) mixes globally and destroys per-position correspondence, so
    # for "where is the object" always read a trunk layer (and a shallow one for resolution).
    feats = {}
    with torch.no_grad():
        h = x
        for i, layer in enumerate(backbone.trunk):
            h = layer(h)
            feats[f"trunk{i}"] = h
        t = h
        for j, layer in enumerate(backbone.tail):
            t = layer(t)
            feats[f"tail{j}_layer{9 + j}"] = t

    print("available layers (name: C x H x W, stride):")
    for k, v in feats.items():
        _, c, hh, ww = v.shape
        print(f"  {k}: {c} x {hh} x {ww}  (stride {args.imgsz // hh})")

    if args.layer not in feats:
        raise SystemExit(f"--layer {args.layer!r} not found; pick one of {list(feats)}")
    fmap = feats[args.layer][0]  # (C, H, W)

    # feature L2-norm per position (a "how strongly does the encoder respond here" map)
    norm = fmap.norm(dim=0)  # (H, W)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    norm_up = F.interpolate(norm[None, None], size=(args.imgsz, args.imgsz),
                            mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

    rgb = pca_rgb(fmap)  # (H, W, 3)
    rgb_up = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (args.imgsz, args.imgsz), Image.BILINEAR)) / 255.0

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(arr); ax[0].set_title("input"); ax[0].axis("off")
    ax[1].imshow(rgb_up); ax[1].set_title(f"feature PCA->RGB [{args.layer}]"); ax[1].axis("off")
    ax[2].imshow(arr); ax[2].imshow(norm_up, cmap="jet", alpha=0.5)
    ax[2].set_title(f"feature L2-norm [{args.layer}]"); ax[2].axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print("saved", args.out, "| layer:", args.layer, "| feature map:", tuple(fmap.shape))


if __name__ == "__main__":
    main()
