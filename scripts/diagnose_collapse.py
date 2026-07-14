"""Diagnose dimensional and representation collapse in CNN-JEPA backbones.

Loads two checkpoints, runs an image through them, and computes:
1. Singular Value Decomposition (SVD) spectrum of spatial feature vectors.
2. Distribution of pairwise spatial cosine similarities.
3. Distribution of channel standard deviations (variance).
4. Off-diagonal channel correlations.

Saves a combined diagnostic plot.
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


def load_backbone(ckpt_path, yaml_path, scale, yolo_repo, src_prefix):
    from models.yolo_backbone import YOLO26SEABackbone
    backbone = YOLO26SEABackbone(yaml_path=yaml_path, scale=scale, yolo_repo_path=yolo_repo)
    
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    p = src_prefix + "."
    remapped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    missing, unexpected = backbone.load_state_dict(remapped, strict=False)
    print(f"[{os.path.basename(ckpt_path)}] Loaded {len(remapped) - len(unexpected)}/{len(remapped)} backbone tensors")
    backbone.eval()
    return backbone


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
    return feats[layer_name][0]  # (C, H, W)


def analyze_features(fmap):
    # fmap: (C, H, W) tensor
    c, h, w = fmap.shape
    # Reshape to (HW, C) representing HW spatial positions each with a C-dim feature vector
    X = fmap.reshape(c, h * w).T.float().cpu()  # (HW, C)
    
    # 1. Singular Value Decomposition (SVD)
    # Center X
    X_centered = X - X.mean(dim=0, keepdim=True)
    # torch.svd returns U, S, V. S are the singular values.
    _, S, _ = torch.svd(X_centered)
    singular_values = S.numpy()
    
    # 2. Pairwise Spatial Cosine Similarities
    # Normalize features to unit length along C dimension
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim_matrix = torch.mm(X_norm, X_norm.t())  # (HW, HW)
    # Extract off-diagonal elements (exclude similarity of patch with itself)
    n = cos_sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off_diag_cos = cos_sim_matrix[mask].numpy()
    
    # 3. Channel Standard Deviations (Variances)
    channel_stds = X.std(dim=0).numpy()
    
    # 4. Off-diagonal Channel Correlations
    # Compute correlation matrix of shape (C, C)
    corr_matrix = torch.corrcoef(X.t())
    # Fill NaNs (if any channel is constant, std=0, correlation is NaN)
    corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
    c_mask = ~torch.eye(c, dtype=torch.bool)
    off_diag_corr = corr_matrix[c_mask].numpy()
    
    # Quantitative metrics
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
    ap.add_argument("--ckpt1", required=True, help="Path to first checkpoint")
    ap.add_argument("--ckpt2", required=True, help="Path to second checkpoint")
    ap.add_argument("--yaml", required=True, help="Model config YAML")
    ap.add_argument("--scale", default="s")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--src-prefix", default="backbone_momentum")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--layer", default="trunk8", help="layer to analyze")
    ap.add_argument("--out", default="collapse_diagnostics.png")
    ap.add_argument("--yolo-repo", default=None)
    args = ap.parse_args()

    _ensure_ultralytics(args.yolo_repo)
    
    # Load backbones
    print("Loading models...")
    backbone1 = load_backbone(args.ckpt1, args.yaml, args.scale, args.yolo_repo, args.src_prefix)
    backbone2 = load_backbone(args.ckpt2, args.yaml, args.scale, args.yolo_repo, args.src_prefix)

    # Preprocess image
    img = Image.open(args.image).convert("RGB").resize((args.imgsz, args.imgsz))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()

    # Extract feature maps
    fmap1 = get_features(backbone1, x, args.layer)
    fmap2 = get_features(backbone2, x, args.layer)

    # Run diagnostics
    print("Analyzing features for checkpoint 1...")
    d1 = analyze_features(fmap1)
    print("Analyzing features for checkpoint 2...")
    d2 = analyze_features(fmap2)

    # Print comparative table to console
    name1 = os.path.basename(args.ckpt1)
    name2 = os.path.basename(args.ckpt2)
    print("\n" + "="*60)
    print(f"COLLAPSE DIAGNOSTICS SUMMARY ({args.layer})")
    print("="*60)
    print(f"{'Metric':<30} | {name1:<20} | {name2:<20}")
    print("-"*60)
    print(f"{'Effective Rank (SVD)':<30} | {d1['effective_rank']:<20.4f} | {d2['effective_rank']:<20.4f}")
    print(f"{'Mean Spatial Cosine':<30} | {d1['mean_cos']:<20.4f} | {d2['mean_cos']:<20.4f}")
    print(f"{'Std Spatial Cosine':<30} | {d1['std_cos']:<20.4f} | {d2['std_cos']:<20.4f}")
    print(f"{'Mean Abs Channel Correlation':<30} | {d1['mean_abs_corr']:<20.4f} | {d2['mean_abs_corr']:<20.4f}")
    print(f"{'Dead Channels (<1e-4 std)':<30} | {d1['dead_channels']:<20} | {d2['dead_channels']:<20}")
    print("="*60)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Singular Value Spectrum (Log Scale)
    ax = axes[0, 0]
    ax.plot(d1["singular_values"] / d1["singular_values"].max(), label=f"{name1} (Rank: {d1['effective_rank']:.1f})", color="royalblue")
    ax.plot(d2["singular_values"] / d2["singular_values"].max(), label=f"{name2} (Rank: {d2['effective_rank']:.1f})", color="crimson", linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Normalized Singular Value Spectrum (SVD)")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Value (Log Scale)")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()

    # 2. Pairwise Cosine Similarity Histogram
    ax = axes[0, 1]
    ax.hist(d1["cos_sims"], bins=50, alpha=0.6, label=f"{name1} (Mean: {d1['mean_cos']:.3f})", color="royalblue", density=True)
    ax.hist(d2["cos_sims"], bins=50, alpha=0.6, label=f"{name2} (Mean: {d2['mean_cos']:.3f})", color="crimson", density=True, histtype='step', linewidth=2)
    ax.set_title("Pairwise Spatial Cosine Similarity Distribution")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle=":")
    ax.legend()

    # 3. Channel Standard Deviations Histogram
    ax = axes[1, 0]
    ax.hist(d1["channel_stds"], bins=50, alpha=0.6, label=name1, color="royalblue", density=True)
    ax.hist(d2["channel_stds"], bins=50, alpha=0.6, label=name2, color="crimson", density=True, histtype='step', linewidth=2)
    ax.set_title("Distribution of Channel Standard Deviations (Variance)")
    ax.set_xlabel("Channel Standard Deviation")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle=":")
    ax.legend()

    # 4. Off-diagonal Channel Correlation Histogram
    ax = axes[1, 1]
    ax.hist(d1["correlations"], bins=50, alpha=0.6, label=f"{name1} (Mean Abs: {d1['mean_abs_corr']:.3f})", color="royalblue", density=True)
    ax.hist(d2["correlations"], bins=50, alpha=0.6, label=f"{name2} (Mean Abs: {d2['mean_abs_corr']:.3f})", color="crimson", density=True, histtype='step', linewidth=2)
    ax.set_title("Off-diagonal Channel Correlation Distribution")
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle=":")
    ax.legend()

    plt.suptitle(f"Representational & Dimensional Collapse Diagnostics ({args.layer})", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved diagnostics plot to {args.out}")


if __name__ == "__main__":
    main()
