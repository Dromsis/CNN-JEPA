"""Compare CNN-JEPA features (PCA and L2-norm) of two checkpoints on an input image.

Loads the backbones from two different checkpoints, runs the same image through both,
and generates a side-by-side comparison figure.
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
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    return proj.reshape(h, w, 3)


def get_features(backbone, x, layer_name):
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
            
    if layer_name not in feats:
        raise SystemExit(f"Layer {layer_name!r} not found; pick one of {list(feats.keys())}")
    
    return feats[layer_name][0]  # (C, H, W)


def load_backbone(ckpt_path, yaml_path, scale, yolo_repo, src_prefix):
    from models.yolo_backbone import YOLO26SEABackbone
    backbone = YOLO26SEABackbone(yaml_path=yaml_path, scale=scale, yolo_repo_path=yolo_repo)
    
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    p = src_prefix + "."
    remapped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    print(f"[{os.path.basename(ckpt_path)}] Loaded {len(remapped) - len(unexpected)}/{len(remapped)} backbone tensors "
          f"(missing {len(missing)}, unexpected {len(unexpected)})")
    backbone.eval()
    return backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt1", required=True, help="Path to first checkpoint")
    ap.add_argument("--ckpt2", required=True, help="Path to second checkpoint")
    ap.add_argument("--yaml", required=True, help="Model config YAML")
    ap.add_argument("--scale", default="s")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--src-prefix", default="backbone_momentum")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--layer", default="trunk8",
                    help="which layer to visualize, e.g. trunk8, or 'all-trunks' for trunk1 to trunk8")
    ap.add_argument("--out", default="comparison_feat.png")
    ap.add_argument("--yolo-repo", default=None)
    args = ap.parse_args()

    _ensure_ultralytics(args.yolo_repo)
    
    # 1. Load both backbones
    print("Loading models...")
    backbone1 = load_backbone(args.ckpt1, args.yaml, args.scale, args.yolo_repo, args.src_prefix)
    backbone2 = load_backbone(args.ckpt2, args.yaml, args.scale, args.yolo_repo, args.src_prefix)

    # 2. Preprocess input image
    img = Image.open(args.image).convert("RGB").resize((args.imgsz, args.imgsz))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()

    if args.layer == "all-trunks":
        layers_to_run = [f"trunk{i}" for i in range(1, 9)]
    else:
        layers_to_run = [args.layer]

    for layer_name in layers_to_run:
        print(f"Generating comparison for layer: {layer_name}...")
        # 3. Extract feature maps
        fmap1 = get_features(backbone1, x, layer_name)
        fmap2 = get_features(backbone2, x, layer_name)

        # 4. Process Checkpoint 1
        norm1 = fmap1.norm(dim=0)
        norm1 = (norm1 - norm1.min()) / (norm1.max() - norm1.min() + 1e-8)
        norm_up1 = F.interpolate(norm1[None, None], size=(args.imgsz, args.imgsz),
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        rgb1 = pca_rgb(fmap1)
        rgb_up1 = np.asarray(Image.fromarray((rgb1 * 255).astype(np.uint8)).resize(
            (args.imgsz, args.imgsz), Image.BILINEAR)) / 255.0

        # 5. Process Checkpoint 2
        norm2 = fmap2.norm(dim=0)
        norm2 = (norm2 - norm2.min()) / (norm2.max() - norm2.min() + 1e-8)
        norm_up2 = F.interpolate(norm2[None, None], size=(args.imgsz, args.imgsz),
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        rgb2 = pca_rgb(fmap2)
        rgb_up2 = np.asarray(Image.fromarray((rgb2 * 255).astype(np.uint8)).resize(
            (args.imgsz, args.imgsz), Image.BILINEAR)) / 255.0

        # 6. Plot comparison figure
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        
        name1 = os.path.basename(args.ckpt1)
        name2 = os.path.basename(args.ckpt2)

        # Row 1: Checkpoint 1
        ax[0, 0].imshow(arr)
        ax[0, 0].set_title(f"Input: {os.path.basename(args.image)}")
        ax[0, 0].axis("off")
        
        ax[0, 1].imshow(rgb_up1)
        ax[0, 1].set_title(f"{name1} PCA->RGB [{layer_name}]")
        ax[0, 1].axis("off")
        
        ax[0, 2].imshow(arr)
        ax[0, 2].imshow(norm_up1, cmap="jet", alpha=0.5)
        ax[0, 2].set_title(f"{name1} L2-norm [{layer_name}]")
        ax[0, 2].axis("off")

        # Row 2: Checkpoint 2
        ax[1, 0].imshow(arr)
        ax[1, 0].set_title(f"Input: {os.path.basename(args.image)}")
        ax[1, 0].axis("off")
        
        ax[1, 1].imshow(rgb_up2)
        ax[1, 1].set_title(f"{name2} PCA->RGB [{layer_name}]")
        ax[1, 1].axis("off")
        
        ax[1, 2].imshow(arr)
        ax[1, 2].imshow(norm_up2, cmap="jet", alpha=0.5)
        ax[1, 2].set_title(f"{name2} L2-norm [{layer_name}]")
        ax[1, 2].axis("off")

        plt.tight_layout()
        
        # Determine output file path
        if args.layer == "all-trunks":
            base, ext = os.path.splitext(args.out)
            out_path = f"{base}_{layer_name}{ext}"
        else:
            out_path = args.out
            
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved comparison figure to {out_path}")


if __name__ == "__main__":
    main()
