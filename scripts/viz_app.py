"""Gradio App to visualize YOLO26-SEA CNN-JEPA backbone features.

Run it with:
    YOLO_SEA_REPO=/home/dromsis/Documents/yolo_sea_homemade PYTHONPATH=. \
      /home/dromsis/Documents/yolo_sea_homemade/.venv/bin/python scripts/viz_app.py
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr
import matplotlib

DEFAULT_CKPT = "/home/dromsis/Documents/jepa_yolo_20ep.ckpt"
DEFAULT_YAML = "/home/dromsis/Documents/yolo_sea_homemade/ultralytics/cfg/models/26/yolo26-sea.yaml"
DEFAULT_YOLO_REPO = "/home/dromsis/Documents/yolo_sea_homemade"

# Ensure Ultralytics can be imported
def _ensure_ultralytics(repo):
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    cand = repo or os.environ.get("YOLO_SEA_REPO") or DEFAULT_YOLO_REPO
    if cand and cand not in sys.path:
        sys.path.insert(0, cand)
    import ultralytics  # noqa: F401

# PCA to RGB conversion
def pca_rgb(fmap):  # fmap: (C, H, W) tensor
    c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).T.float().cpu().numpy()  # (HW, C)
    x = x - x.mean(0, keepdims=True)
    # top-3 principal components via SVD
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:3].T  # (HW, 3)
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    return proj.reshape(h, w, 3)

# Global variables for model state
_loaded_model = None
_loaded_ckpt_path = None
_loaded_yaml_path = None
_loaded_scale = None

def load_model(ckpt_path, yaml_path, scale, yolo_repo):
    global _loaded_model, _loaded_ckpt_path, _loaded_yaml_path, _loaded_scale
    
    # Check if already loaded
    if (_loaded_model is not None and 
        _loaded_ckpt_path == ckpt_path and 
        _loaded_yaml_path == yaml_path and 
        _loaded_scale == scale):
        return _loaded_model, "Model already loaded and cached."

    if not os.path.exists(ckpt_path):
        return None, f"Checkpoint not found at {ckpt_path}"
    if not os.path.exists(yaml_path):
        return None, f"YAML config not found at {yaml_path}"
        
    try:
        _ensure_ultralytics(yolo_repo)
        from models.yolo_backbone import YOLO26SEABackbone
        
        backbone = YOLO26SEABackbone(yaml_path=yaml_path, scale=scale, yolo_repo_path=yolo_repo)
        
        # Load weights
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        p = "backbone_momentum."
        remapped = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        
        missing, unexpected = backbone.load_state_dict(remapped, strict=False)
        backbone.eval()
        
        # Cache the model
        _loaded_model = backbone
        _loaded_ckpt_path = ckpt_path
        _loaded_yaml_path = yaml_path
        _loaded_scale = scale
        
        msg = (f"Successfully loaded model!\n"
               f"Tensors loaded: {len(remapped) - len(unexpected)}/{len(remapped)}\n"
               f"Missing: {len(missing)}\n"
               f"Unexpected: {len(unexpected)}")
        return backbone, msg
    except Exception as e:
        import traceback
        return None, f"Error loading model:\n{str(e)}\n{traceback.format_exc()}"

def run_visualization(img_pil, layer_name, imgsz, alpha, ckpt_path, yaml_path, scale, yolo_repo):
    if img_pil is None:
        return None, None, None, "Please upload or select an image first."
        
    backbone, load_msg = load_model(ckpt_path, yaml_path, scale, yolo_repo)
    if backbone is None:
        return None, None, None, load_msg
        
    try:
        # Preprocess
        img_resized = img_pil.convert("RGB").resize((imgsz, imgsz))
        arr = np.asarray(img_resized).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Run forward pass, saving activations
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
            return None, None, None, f"Layer {layer_name} not found. Available: {list(feats.keys())}"
            
        fmap = feats[layer_name][0]  # (C, H, W)
        
        # 1. Feature L2-norm per position (Activation Heatmap)
        norm = fmap.norm(dim=0)  # (H, W)
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
        norm_up = F.interpolate(norm[None, None], size=(imgsz, imgsz),
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                                 
        # Generate Overlay with "jet" colormap
        cmap = matplotlib.colormaps["jet"]
        heatmap_rgb = cmap(norm_up)[..., :3]
        overlay = arr * (1 - alpha) + heatmap_rgb * alpha
        overlay = np.clip(overlay, 0.0, 1.0)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        
        # 2. PCA -> RGB Map
        rgb = pca_rgb(fmap)  # (H, W, 3)
        rgb_up = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (imgsz, imgsz), Image.BILINEAR)) / 255.0
        pca_img = Image.fromarray((rgb_up * 255).astype(np.uint8))
        
        status_msg = (f"Visualized layer {layer_name}.\n"
                      f"Feature map shape: {tuple(fmap.shape)}\n"
                      f"Downsampling factor: {imgsz // fmap.shape[1]}x")
        
        return img_resized, pca_img, overlay_img, status_msg
    except Exception as e:
        import traceback
        return None, None, None, f"Error processing image:\n{str(e)}\n{traceback.format_exc()}"

def build_gui():
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="YOLO26-SEA JEPA Feature Visualizer") as demo:
        gr.Markdown(
            """
            # 🚢 YOLO26-SEA CNN-JEPA Feature Visualizer
            Visualize the self-supervised representations learned by training the **YOLO26-SEA backbone** with CNN-JEPA.
            
            This tool computes:
            * **PCA -> RGB**: Maps the first 3 principal components of feature vectors to RGB channels. Regions with similar colors share similar feature representations (semantic segment boundaries).
            * **Feature L2-Norm Heatmap**: Shows the intensity of model activations (which parts of the image the backbone pays most attention to).
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                ckpt_input = gr.Textbox(label="Checkpoint Path", value=DEFAULT_CKPT)
                yaml_input = gr.Textbox(label="Model YAML Path", value=DEFAULT_YAML)
                yolo_repo_input = gr.Textbox(label="YOLO Sea Repo Root", value=DEFAULT_YOLO_REPO)
                scale_input = gr.Dropdown(choices=["s", "n"], label="Backbone Scale", value="s")
                
                load_btn = gr.Button("Load/Reload Model", variant="secondary")
                load_status = gr.Textbox(label="Model Status", interactive=False, value="Model not loaded yet.")
                
                gr.Markdown("### 🛠️ Visualizer Controls")
                layer_choices = [f"trunk{i}" for i in range(9)] + [f"tail{j}_layer{9+j}" for j in range(3)]
                layer_input = gr.Dropdown(
                    choices=layer_choices, 
                    value="trunk8", 
                    label="Backbone Layer to Visualize",
                    info="Trunk layers keep spatial locality; trunk8 is the split point before tail."
                )
                imgsz_input = gr.Slider(minimum=256, maximum=1024, step=32, value=640, label="Input Image Size (px)")
                alpha_input = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Heatmap Overlay Opacity (Alpha)")
                
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Input Image")
                input_image = gr.Image(type="pil", label="Choose an Image")
                
                submit_btn = gr.Button("Run Representation Analysis", variant="primary")
                
                gr.Markdown("### 📊 Outputs")
                with gr.Row():
                    out_orig = gr.Image(label="Input Image (Resized)", interactive=False)
                    out_pca = gr.Image(label="Feature PCA -> RGB (Semantic Regions)", interactive=False)
                    out_heatmap = gr.Image(label="Feature L2-Norm (Attention Overlay)", interactive=False)
                
                out_status = gr.Textbox(label="Processing Output", interactive=False)
                
                # Predefined examples
                examples = [
                    ["/home/dromsis/Documents/drone_ws/drone_ws_fan/models/yolo_script/bus.jpg", "trunk8", 640, 0.5],
                    ["/home/dromsis/Documents/yolo_sea_homemade/.venv/lib/python3.12/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg", "trunk8", 640, 0.5],
                ]
                # Filter down to existing example images to avoid UI clutter of missing files
                valid_examples = [ex for ex in examples if os.path.exists(ex[0])]
                if valid_examples:
                    gr.Examples(
                        examples=valid_examples,
                        inputs=[input_image, layer_input, imgsz_input, alpha_input],
                        label="Example Images"
                    )
                    
        # Click handlers
        load_btn.click(
            fn=lambda ckpt, yml, sc, repo: load_model(ckpt, yml, sc, repo)[1],
            inputs=[ckpt_input, yaml_input, scale_input, yolo_repo_input],
            outputs=[load_status]
        )
        
        submit_btn.click(
            fn=run_visualization,
            inputs=[input_image, layer_input, imgsz_input, alpha_input, ckpt_input, yaml_input, scale_input, yolo_repo_input],
            outputs=[out_orig, out_pca, out_heatmap, out_status]
        )
        
    return demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    ap.add_argument("--share", action="store_true", help="Generate public sharing link")
    args = ap.parse_args()
    
    demo = build_gui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
