"""Evaluate CNN-JEPA, Random, and COCO backbones over the entire test set.

Loads all three backbones once, runs all test images through them, computes
representation metrics (Effective SVD Rank, Spatial Cosine Similarity, etc.),
and outputs the aggregated results (mean +/- std) in a Markdown table.
"""
import argparse
import glob
import os
import sys
from tqdm import tqdm

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


def get_yolov8s_features(model, x):
    h = x
    for i in range(9):  # layers 0 to 8 inclusive
        h = model.model.model[i](h)
    return h[0]


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
    ap.add_argument("--test-dir", required=True, help="Path to test images directory")
    ap.add_argument("--src-prefix", default="backbone_momentum")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-diag", default="artifacts/dataset_comparison_diag.png")
    ap.add_argument("--yolo-repo", default=None)
    args = ap.parse_args()

    _ensure_ultralytics(args.yolo_repo)
    from models.yolo_backbone import YOLO26SEABackbone
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the three models
    print("Loading models...")
    backbone_jepa = YOLO26SEABackbone(yaml_path=args.yaml, scale=args.scale, yolo_repo_path=args.yolo_repo)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    p = args.src_prefix + "."
    remapped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    backbone_jepa.load_state_dict(remapped, strict=False)
    backbone_jepa.to(device)
    backbone_jepa.eval()
    
    backbone_random = YOLO26SEABackbone(yaml_path=args.yaml, scale=args.scale, yolo_repo_path=args.yolo_repo)
    backbone_random.to(device)
    backbone_random.eval()

    yolov8s_coco = YOLO("yolov8s.pt")
    yolov8s_coco.to(device)
    yolov8s_coco.eval()

    # 2. Get list of test images
    img_exts = ["*.jpg", "*.jpeg", "*.png", "*.PNG", "*.JPG", "*.JPEG"]
    img_paths = []
    for ext in img_exts:
        img_paths.extend(glob.glob(os.path.join(args.test_dir, ext)))
    img_paths = sorted(list(set(img_paths)))
    
    if not img_paths:
        raise SystemExit(f"No images found in test directory: {args.test_dir}")
    print(f"Found {len(img_paths)} test images. Starting evaluation...")

    # Data structure to accumulate metrics
    metrics = {
        "jepa": {"rank": [], "mean_cos": [], "std_cos": [], "abs_corr": [], "dead": [], "s_vals": [], "cos_sims_all": []},
        "random": {"rank": [], "mean_cos": [], "std_cos": [], "abs_corr": [], "dead": [], "s_vals": [], "cos_sims_all": []},
        "coco": {"rank": [], "mean_cos": [], "std_cos": [], "abs_corr": [], "dead": [], "s_vals": [], "cos_sims_all": []}
    }

    mean_transform = np.array([0.485, 0.456, 0.406])
    std_transform = np.array([0.229, 0.224, 0.225])

    # 3. Loop over all test images
    for path in tqdm(img_paths, desc="Evaluating dataset"):
        try:
            # Preprocess image
            img = Image.open(path).convert("RGB").resize((args.imgsz, args.imgsz))
            arr = np.asarray(img).astype(np.float32) / 255.0
            x = torch.from_numpy(((arr - mean_transform) / std_transform).transpose(2, 0, 1)).unsqueeze(0).float().to(device)

            with torch.no_grad():
                fmap_jepa = backbone_jepa.trunk(x)[0]
                fmap_random = backbone_random.trunk(x)[0]
                fmap_coco = get_yolov8s_features(yolov8s_coco, x)

            # Analyze representation properties
            d_jepa = analyze_features(fmap_jepa)
            d_random = analyze_features(fmap_random)
            d_coco = analyze_features(fmap_coco)

            # Accumulate metrics
            for key, val in [("jepa", d_jepa), ("random", d_random), ("coco", d_coco)]:
                metrics[key]["rank"].append(val["effective_rank"])
                metrics[key]["mean_cos"].append(val["mean_cos"])
                metrics[key]["std_cos"].append(val["std_cos"])
                metrics[key]["abs_corr"].append(val["mean_abs_corr"])
                metrics[key]["dead"].append(val["dead_channels"])
                metrics[key]["s_vals"].append(val["singular_values"] / val["singular_values"].max())
                metrics[key]["cos_sims_all"].append(val["cos_sims"])

        except Exception as e:
            print(f"Error processing image {os.path.basename(path)}: {e}")

    # 4. Aggregate results
    results = {}
    for k in ["jepa", "random", "coco"]:
        results[k] = {
            "rank_mean": np.mean(metrics[k]["rank"]),
            "rank_std": np.std(metrics[k]["rank"]),
            "cos_mean": np.mean(metrics[k]["mean_cos"]),
            "cos_std": np.std(metrics[k]["mean_cos"]),
            "std_cos_mean": np.mean(metrics[k]["std_cos"]),
            "std_cos_std": np.std(metrics[k]["std_cos"]),
            "corr_mean": np.mean(metrics[k]["abs_corr"]),
            "corr_std": np.std(metrics[k]["abs_corr"]),
            "dead_mean": np.mean(metrics[k]["dead"]),
            "dead_std": np.std(metrics[k]["dead"]),
        }

    # Print comparative table to console
    print("\n" + "="*80)
    print(f"QUANTITATIVE COMPARISON OVER ENTIRE TEST SET ({len(img_paths)} images)")
    print("="*80)
    print(f"{'Metric':<30} | {'CNN-JEPA (Ep 42)':<22} | {'YOLO26-SEA Random':<22} | {'YOLOv8s COCO':<22}")
    print("-"*80)
    print(f"{'Effective Rank (SVD)':<30} | {results['jepa']['rank_mean']:.4f} ± {results['jepa']['rank_std']:.4f} | {results['random']['rank_mean']:.4f} ± {results['random']['rank_std']:.4f} | {results['coco']['rank_mean']:.4f} ± {results['coco']['rank_std']:.4f}")
    print(f"{'Mean Spatial Cosine':<30} | {results['jepa']['cos_mean']:.4f} ± {results['jepa']['cos_std']:.4f} | {results['random']['cos_mean']:.4f} ± {results['random']['cos_std']:.4f} | {results['coco']['cos_mean']:.4f} ± {results['coco']['cos_std']:.4f}")
    print(f"{'Std Spatial Cosine':<30} | {results['jepa']['std_cos_mean']:.4f} ± {results['jepa']['std_cos_std']:.4f} | {results['random']['std_cos_mean']:.4f} ± {results['random']['std_cos_std']:.4f} | {results['coco']['std_cos_mean']:.4f} ± {results['coco']['std_cos_std']:.4f}")
    print(f"{'Mean Abs Correlation':<30} | {results['jepa']['corr_mean']:.4f} ± {results['jepa']['corr_std']:.4f} | {results['random']['corr_mean']:.4f} ± {results['random']['corr_std']:.4f} | {results['coco']['corr_mean']:.4f} ± {results['coco']['corr_std']:.4f}")
    print(f"{'Dead Channels':<30} | {results['jepa']['dead_mean']:.1f} ± {results['jepa']['dead_std']:.1f} | {results['random']['dead_mean']:.1f} ± {results['random']['dead_std']:.1f} | {results['coco']['dead_mean']:.1f} ± {results['coco']['dead_std']:.1f}")
    print("="*80)

    # 5. Plot aggregated SVD & Cosine similarity diagnostics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Singular Values
    avg_s_jepa = np.mean(metrics["jepa"]["s_vals"], axis=0)
    avg_s_random = np.mean(metrics["random"]["s_vals"], axis=0)
    avg_s_coco = np.mean(metrics["coco"]["s_vals"], axis=0)
    
    axes[0].plot(avg_s_jepa, label=f"CNN-JEPA (Avg Rank: {results['jepa']['rank_mean']:.1f})", color="royalblue")
    axes[0].plot(avg_s_random, label=f"Random (Avg Rank: {results['random']['rank_mean']:.1f})", color="gray", linestyle=":")
    axes[0].plot(avg_s_coco, label=f"Supervised COCO (Avg Rank: {results['coco']['rank_mean']:.1f})", color="forestgreen", linestyle="--")
    axes[0].set_yscale("log")
    axes[0].set_title("Average Singular Value Spectrum (SVD)")
    axes[0].set_xlabel("Singular Value Index")
    axes[0].set_ylabel("Normalized Value (Log Scale)")
    axes[0].grid(True, which="both", linestyle=":")
    axes[0].legend()

    # Spatial Cosine Similarity (Flat histogram across all samples)
    flat_cos_jepa = np.concatenate(metrics["jepa"]["cos_sims_all"])
    flat_cos_random = np.concatenate(metrics["random"]["cos_sims_all"])
    flat_cos_coco = np.concatenate(metrics["coco"]["cos_sims_all"])

    # Subsample to avoid memory issues when plotting histograms
    subsample_idx = np.random.choice(len(flat_cos_jepa), min(len(flat_cos_jepa), 100000), replace=False)
    
    axes[1].hist(flat_cos_jepa[subsample_idx], bins=50, alpha=0.4, label=f"CNN-JEPA (Mean: {results['jepa']['cos_mean']:.3f})", color="royalblue", density=True)
    axes[1].hist(flat_cos_random[subsample_idx], bins=50, alpha=0.4, label=f"Random (Mean: {results['random']['cos_mean']:.3f})", color="gray", density=True, histtype='step')
    axes[1].hist(flat_cos_coco[subsample_idx], bins=50, alpha=0.4, label=f"COCO Supervised (Mean: {results['coco']['cos_mean']:.3f})", color="forestgreen", density=True, histtype='step', linewidth=1.5)
    axes[1].set_title("Spatial Cosine Similarity Distribution")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    os.makedirs(os.path.dirname(args.out_diag), exist_ok=True)
    plt.suptitle(f"Aggregated Representation Metrics over Test Set ({len(img_paths)} images)", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out_diag, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved aggregated diagnostic plot to {args.out_diag}")


if __name__ == "__main__":
    main()
