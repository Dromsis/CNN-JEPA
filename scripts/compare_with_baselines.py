"""Compare CNN-JEPA representations with Random and Supervised COCO baselines.

Loads:
1. CNN-JEPA (Epoch 76) backbone.
2. Randomly initialized YOLO26-SEA backbone.
3. Supervised COCO-trained YOLOv8s backbone.

Runs the same input image through them, extracts features at the final trunk layer
(stride 32), and generates:
- A side-by-side visualization of PCA->RGB and L2-norm.
- A quantitative comparison of SVD rank, spatial similarity, and channel activity.
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


def get_yolov8s_features(model, x):
    # Runs the input x through layers 0 to 8 of yolov8s to get the stride 32 trunk output
    h = x
    with torch.no_grad():
        for i in range(9):  # layers 0 to 8 inclusive
            h = model.model.model[i](h)
    return h[0]  # (512, H, W)


def analyze_features(fmap):
    c, h, w = fmap.shape
    X = fmap.reshape(c, h * w).T.float().cpu()  # (HW, C)
    
    # 1. SVD
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, _ = torch.svd(X_centered)
    singular_values = S.numpy()
    
    # 2. Pairwise Cosine Similarity
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim_matrix = torch.mm(X_norm, X_norm.t())
    n = cos_sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag_cos = cos_sim_matrix[mask].numpy()
    
    # 3. Variance and correlation
    channel_stds = X.std(dim=0).numpy()
    corr_matrix = torch.corrcoef(X.t())
    corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    c_mask = ~torch.eye(c, dtype=torch.bool)
    off_diag_corr = corr_matrix[c_mask].numpy()
    
    effective_rank = (singular_values.sum() ** 2) / ((singular_values ** 2).sum() + 1e-8)
    
    return {
        "singular_values": singular_values,
        "cos_sims": off_diag_cos,
        "channel_stds": channel_stds,
        "correlations": off_diag_corr,
        "effective_rank": effective_rank,
        "mean_cos": np.mean(off_diag_cos),
        "std_cos": np.std(off_diag_cos),
        "mean_abs_corr": np.mean(np.abs(off_diag_corr)),
        "dead_channels": np.sum(channel_stds < 1e-4)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="CNN-JEPA checkpoint path")
    ap.add_argument("--yaml", required=True, help="YOLO26-SEA config YAML")
    ap.add_argument("--scale", default="s")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--src-prefix", default="backbone_momentum")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-viz", default="baseline_comparison_viz.png")
    ap.add_argument("--out-diag", default="baseline_comparison_diag.png")
    ap.add_argument("--yolo-repo", default=None)
    args = ap.parse_args()

    _ensure_ultralytics(args.yolo_repo)
    from models.yolo_backbone import YOLO26SEABackbone
    from ultralytics import YOLO

    # 1. Load the three models
    print("Loading models...")
    # Model A: CNN-JEPA
    backbone_jepa = YOLO26SEABackbone(yaml_path=args.yaml, scale=args.scale, yolo_repo_path=args.yolo_repo)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    p = args.src_prefix + "."
    remapped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    backbone_jepa.load_state_dict(remapped, strict=False)
    backbone_jepa.eval()
    
    # Model B: YOLO26-SEA Random
    backbone_random = YOLO26SEABackbone(yaml_path=args.yaml, scale=args.scale, yolo_repo_path=args.yolo_repo)
    backbone_random.eval()

    # Model C: YOLOv8s Supervised (COCO)
    yolov8s_coco = YOLO("yolov8s.pt")
    yolov8s_coco.eval()

    # 2. Preprocess input image
    img = Image.open(args.image).convert("RGB").resize((args.imgsz, args.imgsz))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()

    # 3. Extract feature maps
    with torch.no_grad():
        fmap_jepa = backbone_jepa.trunk(x)[0]
        fmap_random = backbone_random.trunk(x)[0]
        fmap_coco = get_yolov8s_features(yolov8s_coco, x)

    # 4. Analyze features
    print("Analyzing CNN-JEPA features...")
    d_jepa = analyze_features(fmap_jepa)
    print("Analyzing Random backbone features...")
    d_random = analyze_features(fmap_random)
    print("Analyzing Supervised COCO features...")
    d_coco = analyze_features(fmap_coco)

    # Print comparative table to console
    print("\n" + "="*80)
    print(f"QUANTITATIVE COMPARISON WITH BASELINES")
    print("="*80)
    print(f"{'Metric':<30} | {'CNN-JEPA (Ep 76)':<20} | {'YOLO26-SEA Random':<20} | {'YOLOv8s COCO':<20}")
    print("-"*80)
    print(f"{'Effective Rank (SVD)':<30} | {d_jepa['effective_rank']:<20.4f} | {d_random['effective_rank']:<20.4f} | {d_coco['effective_rank']:<20.4f}")
    print(f"{'Mean Spatial Cosine':<30} | {d_jepa['mean_cos']:<20.4f} | {d_random['mean_cos']:<20.4f} | {d_coco['mean_cos']:<20.4f}")
    print(f"{'Std Spatial Cosine':<30} | {d_jepa['std_cos']:<20.4f} | {d_random['std_cos']:<20.4f} | {d_coco['std_cos']:<20.4f}")
    print(f"{'Mean Abs Correlation':<30} | {d_jepa['mean_abs_corr']:<20.4f} | {d_random['mean_abs_corr']:<20.4f} | {d_coco['mean_abs_corr']:<20.4f}")
    print(f"{'Dead Channels (<1e-4 std)':<30} | {d_jepa['dead_channels']:<20} | {d_random['dead_channels']:<20} | {d_coco['dead_channels']:<20}")
    print("="*80)

    # 5. Visual comparison: PCA and L2
    # Compute representations for PCA and L2
    def get_up_maps(fmap):
        norm = fmap.norm(dim=0)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
        norm_up = F.interpolate(norm[None, None], size=(args.imgsz, args.imgsz),
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        rgb = pca_rgb(fmap)
        rgb_up = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (args.imgsz, args.imgsz), Image.BILINEAR)) / 255.0
        return rgb_up, norm_up

    pca_jepa, l2_jepa = get_up_maps(fmap_jepa)
    pca_random, l2_random = get_up_maps(fmap_random)
    pca_coco, l2_coco = get_up_maps(fmap_coco)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: PCA -> RGB
    axes[0, 0].imshow(pca_jepa); axes[0, 0].set_title("CNN-JEPA (Ep 76) PCA"); axes[0, 0].axis("off")
    axes[0, 1].imshow(pca_random); axes[0, 1].set_title("YOLO26-SEA Random PCA"); axes[0, 1].axis("off")
    axes[0, 2].imshow(pca_coco); axes[0, 2].set_title("YOLOv8s COCO PCA"); axes[0, 2].axis("off")

    # Row 2: L2-norm
    axes[1, 0].imshow(arr); axes[1, 0].imshow(l2_jepa, cmap="jet", alpha=0.5); axes[1, 0].set_title("CNN-JEPA L2 Heatmap"); axes[1, 0].axis("off")
    axes[1, 1].imshow(arr); axes[1, 1].imshow(l2_random, cmap="jet", alpha=0.5); axes[1, 1].set_title("Random L2 Heatmap"); axes[1, 1].axis("off")
    axes[1, 2].imshow(arr); axes[1, 2].imshow(l2_coco, cmap="jet", alpha=0.5); axes[1, 2].set_title("COCO Supervised L2 Heatmap"); axes[1, 2].axis("off")

    plt.suptitle(f"Visual representation comparison on {os.path.basename(args.image)}", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.out_viz, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visual comparison to {args.out_viz}")

    # 6. SVD Diagnostics plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Singular Values
    axes[0].plot(d_jepa["singular_values"] / d_jepa["singular_values"].max(), label=f"CNN-JEPA (Rank: {d_jepa['effective_rank']:.1f})", color="royalblue")
    axes[0].plot(d_random["singular_values"] / d_random["singular_values"].max(), label=f"Random (Rank: {d_random['effective_rank']:.1f})", color="gray", linestyle=":")
    axes[0].plot(d_coco["singular_values"] / d_coco["singular_values"].max(), label=f"Supervised COCO (Rank: {d_coco['effective_rank']:.1f})", color="forestgreen", linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_title("Singular Value Spectrum (SVD)")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Normalized Value (Log Scale)")
    axes[0].grid(True, which="both", linestyle=":")
    axes[0].legend()

    # Spatial Cosine Similarity
    axes[1].hist(d_jepa["cos_sims"], bins=50, alpha=0.4, label=f"CNN-JEPA (Mean: {d_jepa['mean_cos']:.3f})", color="royalblue", density=True)
    axes[1].hist(d_random["cos_sims"], bins=50, alpha=0.4, label=f"Random (Mean: {d_random['mean_cos']:.3f})", color="gray", density=True, histtype='step')
    axes[1].hist(d_coco["cos_sims"], bins=50, alpha=0.4, label=f"COCO Supervised (Mean: {d_coco['mean_cos']:.3f})", color="forestgreen", density=True, histtype='step', linewidth=1.5)
    axes[1].set_title("Pairwise Spatial Cosine Similarity Distribution")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    plt.suptitle("Quantitative representation metrics comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out_diag, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic comparison to {args.out_diag}")


if __name__ == "__main__":
    main()
