"""Generate clean 2x2 side-by-side visual comparisons between CNN-JEPA and Supervised COCO
for all trunk layers (trunk0 to trunk8).
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

def _ensure_ultralytics(repo):
    try:
        import ultralytics
        return
    except ImportError:
        pass
    cand = repo or os.environ.get("YOLO_SEA_REPO")
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)
    import ultralytics

def pca_rgb(fmap):
    c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).T.float().cpu().numpy()
    x = x - x.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:3].T
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    return proj.reshape(h, w, 3)

def get_yolov8s_features_up_to_layer(model, x, layer_idx):
    h = x
    with torch.no_grad():
        for i in range(layer_idx + 1):
            h = model.model.model[i](h)
    return h[0]

def get_jepa_features_up_to_layer(backbone, x, layer_idx):
    h = x
    with torch.no_grad():
        for i in range(layer_idx + 1):
            h = backbone.trunk[i](h)
    return h[0]

def get_up_maps(fmap, imgsz):
    norm = fmap.norm(dim=0)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    norm_up = F.interpolate(norm[None, None], size=(imgsz, imgsz),
                             mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    rgb = pca_rgb(fmap)
    rgb_up = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (imgsz, imgsz), Image.BILINEAR)) / 255.0
    return rgb_up, norm_up

def main():
    ckpt = "/home/dromsis/Documents/CNN-JEPA/epoch=76-step=45892.ckpt"
    yaml_path = "/home/dromsis/Documents/CNN-JEPA/ultralytics/cfg/models/26/yolo26-sea.yaml"
    image_path = "/home/dromsis/Documents/CNN-JEPA/images.jpeg"
    base_out_path = "/home/dromsis/Documents/CNN-JEPA/jepa_vs_coco_comparison"
    imgsz = 640

    _ensure_ultralytics(None)
    from models.yolo_backbone import YOLO26SEABackbone
    from ultralytics import YOLO

    # 1. Load models
    backbone_jepa = YOLO26SEABackbone(yaml_path=yaml_path, scale="s")
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    remapped = {k[len("backbone_momentum."):]: v for k, v in sd.items() if k.startswith("backbone_momentum.")}
    backbone_jepa.load_state_dict(remapped, strict=False)
    backbone_jepa.eval()

    yolov8s_coco = YOLO("yolov8s.pt")
    yolov8s_coco.eval()

    # 2. Preprocess image
    img = Image.open(image_path).convert("RGB").resize((imgsz, imgsz))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()

    for layer_idx in range(9):
        layer_name = f"trunk{layer_idx}"
        print(f"Generating comparison for layer: {layer_name}...")
        
        # 3. Extract and compute maps
        fmap_jepa = get_jepa_features_up_to_layer(backbone_jepa, x, layer_idx)
        fmap_coco = get_yolov8s_features_up_to_layer(yolov8s_coco, x, layer_idx)

        pca_jepa, l2_jepa = get_up_maps(fmap_jepa, imgsz)
        pca_coco, l2_coco = get_up_maps(fmap_coco, imgsz)

        # 4. Plot 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Row 0: PCA
        axes[0, 0].imshow(pca_jepa)
        axes[0, 0].set_title(f"CNN-JEPA (Epoch 76) PCA [{layer_name}]")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(pca_coco)
        axes[0, 1].set_title(f"YOLOv8s COCO (Pretrained) PCA [{layer_name}]")
        axes[0, 1].axis("off")

        # Row 1: L2 Heatmap
        axes[1, 0].imshow(arr)
        axes[1, 0].imshow(l2_jepa, cmap="jet", alpha=0.5)
        axes[1, 0].set_title(f"CNN-JEPA (Epoch 76) L2 Attention [{layer_name}]")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(arr)
        axes[1, 1].imshow(l2_coco, cmap="jet", alpha=0.5)
        axes[1, 1].set_title(f"YOLOv8s COCO (Pretrained) L2 Attention [{layer_name}]")
        axes[1, 1].axis("off")

        plt.suptitle(f"Comparaison Visuelle : CNN-JEPA vs YOLOv8s COCO ({layer_name})", fontsize=16, y=0.98)
        plt.tight_layout()
        
        out_path = f"{base_out_path}_{layer_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Comparison image successfully saved to {out_path}")

if __name__ == "__main__":
    main()
