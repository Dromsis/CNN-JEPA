"""Gradio App to visualize YOLO26-SEA CNN-JEPA backbone features.

Runs PCA -> RGB projection of patch features to match the V-JEPA 2.1 paper style.
Includes support for permuting the PCA components across RGB channels.
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

DEFAULT_CKPT = "/home/dromsis/Documents/CNN-JEPA/artifacts/pretrain_lightly/ijepacnn_yolo_maritime/I-JEPA-YOLO_maritime_yolo26_sea_backbones_predL3K3_Maskmulti-block_lr0.001_wd0.01_bs32/version_1/epoch=48-step=39445.ckpt"
DEFAULT_YAML = "ultralytics/cfg/models/26/yolo26-sea.yaml"
DEFAULT_YOLO_REPO = "."

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

# PCA to RGB conversion with channel permutation support
def pca_rgb(fmap, perm=(0, 1, 2)):  # fmap: (C, H, W) tensor
    c, h, w = fmap.shape
    x = fmap.reshape(c, h * w).T.float().cpu().numpy()  # (HW, C)
    x = x - x.mean(0, keepdims=True)
    # top-3 principal components via SVD
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:3].T  # (HW, 3) -> PC0, PC1, PC2
    
    # Apply permutation to the PCA columns
    proj = proj[:, list(perm)]
    
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    return proj.reshape(h, w, 3)

# Global variables for model cache
_loaded_model = None
_loaded_ckpt_path = None
_loaded_yaml_path = None
_loaded_scale = None

def load_model(ckpt_path, yaml_path, scale, yolo_repo):
    global _loaded_model, _loaded_ckpt_path, _loaded_yaml_path, _loaded_scale
    
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
        
        # Cache
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

def run_visualization(img_pil, layer_name, imgsz, perm_str, ckpt_path, yaml_path, scale, yolo_repo):
    if img_pil is None:
        return None, None, "Please upload or select an image first."
        
    backbone, load_msg = load_model(ckpt_path, yaml_path, scale, yolo_repo)
    if backbone is None:
        return None, None, load_msg
        
    try:
        # Preprocess image
        img_resized = img_pil.convert("RGB").resize((imgsz, imgsz))
        arr = np.asarray(img_resized).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0).float()
        
        # Forward pass (Dense, no masking)
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
            return img_resized, None, f"Layer {layer_name} not found."
            
        fmap = feats[layer_name][0].detach()  # (C, H_feat, W_feat)
        
        # Parse permutation string
        perm_map = {
            "(0, 1, 2)": (0, 1, 2),
            "(0, 2, 1)": (0, 2, 1),
            "(1, 0, 2)": (1, 0, 2),
            "(1, 2, 0)": (1, 2, 0),
            "(2, 0, 1)": (2, 0, 1),
            "(2, 1, 0)": (2, 1, 0),
        }
        perm = perm_map.get(perm_str, (0, 1, 2))
        
        # PCA -> RGB with permutation
        rgb = pca_rgb(fmap, perm=perm)  # (H_feat, W_feat, 3)
        rgb_up = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
            (imgsz, imgsz), Image.BILINEAR))
        pca_img = Image.fromarray(rgb_up)
        
        status_msg = (f"Visualized layer {layer_name} features using PCA -> RGB.\n"
                      f"Permutation used: {perm_str}\n"
                      f"Feature map shape: {tuple(fmap.shape)}\n"
                      f"Resolution scale: {imgsz // fmap.shape[1]}x downsampling.")
        
        return img_resized, pca_img, status_msg
    except Exception as e:
        import traceback
        return None, None, f"Error processing image:\n{str(e)}\n{traceback.format_exc()}"

def build_gui():
    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="slate",
    )
    
    with gr.Blocks(theme=theme, title="V-JEPA 2.1 PCA Feature Visualizer") as demo:
        gr.Markdown(
            """
            # 🚢 V-JEPA 2.1 PCA Feature Visualizer
            This app visualizes dense features extracted from the YOLO26-SEA backbone using Principal Component Analysis (PCA) mapped to RGB channels, matching the visualization methodology in the V-JEPA 2.1 paper.
            """
        )
        
        with gr.Row():
            # Left panel for configuration
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                ckpt_input = gr.Textbox(label="Checkpoint Path", value=DEFAULT_CKPT)
                yaml_input = gr.Textbox(label="Model YAML Path", value=DEFAULT_YAML)
                yolo_repo_input = gr.Textbox(label="YOLO Sea Repo Root", value=DEFAULT_YOLO_REPO)
                scale_input = gr.Dropdown(choices=["s", "n"], label="Backbone Scale", value="s")
                
                layer_choices = [f"trunk{i}" for i in range(9)] + [f"tail{j}_layer{9+j}" for j in range(3)]
                layer_input = gr.Dropdown(
                    choices=layer_choices, 
                    value="tail2_layer11", 
                    label="Backbone Layer to Visualize",
                    info="Deep layers (like tail2_layer11) represent high-level semantics; trunk layers represent local geometry."
                )
                imgsz_input = gr.Slider(minimum=256, maximum=1024, step=32, value=640, label="Input Image Size (px)")
                
                # Permutation selection dropdown
                perm_input = gr.Dropdown(
                    choices=["(0, 1, 2)", "(0, 2, 1)", "(1, 0, 2)", "(1, 2, 0)", "(2, 0, 1)", "(2, 1, 0)"],
                    value="(0, 1, 2)",
                    label="RGB Component Permutation",
                    info="Choose the mapping of top-3 PCA components to R, G, B channels. Permute to find the most visually appealing coloring."
                )
                
                load_btn = gr.Button("Load/Reload Model", variant="secondary")
                load_status = gr.Textbox(label="Model Load Status", interactive=False, value="Model not loaded yet.")

            # Right panel for visual inputs and outputs
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Image Analysis")
                input_image = gr.Image(type="pil", label="Upload or Choose Input Image")
                submit_btn = gr.Button("Run PCA Analysis", variant="primary")
                
                gr.Markdown("### 📊 Visualization Outputs")
                with gr.Row():
                    out_orig = gr.Image(label="Input Image (Resized)", interactive=False)
                    out_pca = gr.Image(label="PCA -> RGB Feature Map", interactive=False)
                
                out_status = gr.Textbox(label="Processing Output", interactive=False)

        # Predefined examples
        examples = [
            ["/home/dromsis/Documents/drone_ws/drone_ws_fan/models/yolo_script/bus.jpg", "tail2_layer11", 640, "(0, 1, 2)"],
        ]
        valid_examples = [ex for ex in examples if os.path.exists(ex[0])]
        if valid_examples:
            gr.Examples(
                examples=valid_examples,
                inputs=[input_image, layer_input, imgsz_input, perm_input],
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
            inputs=[
                input_image, layer_input, imgsz_input, perm_input,
                ckpt_input, yaml_input, scale_input, yolo_repo_input
            ],
            outputs=[
                out_orig, out_pca, out_status
            ]
        )
        
    return demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    ap.add_argument("--share", action="store_true", help="Generate public sharing link")
    args = ap.parse_args()
    
    demo = build_gui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
