import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_synthetic_sea_image(img_size=640, patch_size=32):
    """Generates a synthetic maritime image: blue background (sea) 
    with a small gray rectangle (20x20 boat) at a random location."""
    # Sea background (blueish gray with slight noise)
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:, :, 0] = 50 + np.random.randint(0, 15, (img_size, img_size)) # R
    img[:, :, 1] = 80 + np.random.randint(0, 15, (img_size, img_size)) # G
    img[:, :, 2] = 120 + np.random.randint(0, 20, (img_size, img_size)) # B
    
    # Tiny boat: 20x20 pixels (high contrast gray)
    boat_size = 20
    # Place boat at a random patch, say patch (6, 12)
    by, bx = 6 * patch_size + 6, 12 * patch_size + 6
    img[by:by+boat_size, bx:bx+boat_size, :] = 180 # Light gray hull
    img[by+4:by+16, bx+8:bx+12, :] = 50 # Dark cabin
    
    # Let's add a second boat at patch (14, 5)
    by2, bx2 = 14 * patch_size + 6, 5 * patch_size + 6
    img[by2:by2+boat_size, bx2:bx2+boat_size, :] = 210
    img[by2+6:by2+14, bx2+6:bx2+14, :] = 30

    return img, (by, bx), (by2, bx2)

def compute_patch_variances(img, patch_size=32):
    """Divides the image into patches and computes the variance of pixels inside each patch."""
    h, w, c = img.shape
    num_h, num_w = h // patch_size, w // patch_size
    variances = np.zeros((num_h, num_w))
    
    # Convert image to grayscale for variance calculation
    gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
    
    for i in range(num_h):
        for j in range(num_w):
            patch = gray[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            # Local variance of the patch
            variances[i, j] = np.var(patch)
            
    return variances

def get_random_mask(num_patches, mask_ratio=0.6):
    """Uniform random masking: every patch has the same probability of being masked."""
    num_keep = int(num_patches * (1 - mask_ratio))
    idx = np.random.permutation(num_patches)
    keep_idx = idx[:num_keep]
    
    mask = np.zeros(num_patches, dtype=bool)
    mask[keep_idx] = True # True = keep (visible context), False = mask (hidden target)
    return mask

def get_variance_biased_mask(variances, mask_ratio=0.6, bias_temp=1.5):
    """Variance-biased masking: patches with high local variance 
    have a much higher probability of being kept in the visible context."""
    num_h, num_w = variances.shape
    num_patches = num_h * num_w
    num_keep = int(num_patches * (1 - mask_ratio))
    
    # Flatten variances
    flat_vars = variances.flatten()
    
    # Compute sampling probabilities
    # We add a small epsilon to avoid division by zero, and use a softmax-like temperature
    # to bias towards high variance
    probs = flat_vars ** bias_temp
    probs = probs + 1e-4 # Ensure background patches still have a tiny chance
    probs = probs / np.sum(probs)
    
    # Sample patches to keep (without replacement)
    keep_idx = np.random.choice(num_patches, size=num_keep, replace=False, p=probs)
    
    mask = np.zeros(num_patches, dtype=bool)
    mask[keep_idx] = True
    return mask

def apply_mask(img, mask_grid, patch_size=32):
    """Draws black squares on masked patches (where mask_grid is False)."""
    h, w, c = img.shape
    masked_img = img.copy()
    num_h, num_w = h // patch_size, w // patch_size
    
    for i in range(num_h):
        for j in range(num_w):
            if not mask_grid[i, j]: # If masked
                # Replace with black (or mask color)
                masked_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :] = 0
                
    return masked_img

def main():
    img_size = 640
    patch_size = 32
    mask_ratio = 0.80 # 80% masked, 20% kept (very sparse!)
    
    # Check if a custom image was passed via command line
    custom_img_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    is_synthetic = True
    if custom_img_path and os.path.exists(custom_img_path):
        print(f"Loading custom image from: {custom_img_path}")
        try:
            # Load and resize
            pil_img = Image.open(custom_img_path).convert("RGB").resize((img_size, img_size))
            img = np.array(pil_img)
            is_synthetic = False
        except Exception as e:
            print(f"Error loading {custom_img_path}: {e}. Falling back to synthetic image.")
            img, boat1_pos, boat2_pos = create_synthetic_sea_image(img_size, patch_size)
    else:
        if custom_img_path:
            print(f"Image {custom_img_path} not found. Falling back to synthetic image.")
        img, boat1_pos, boat2_pos = create_synthetic_sea_image(img_size, patch_size)
        
    num_h, num_w = img_size // patch_size, img_size // patch_size
    num_patches = num_h * num_w
    
    # Compute variances
    print("Computing patch variances...")
    variances = compute_patch_variances(img, patch_size)
    
    # 1. Standard Uniform Random Masking
    print("Generating standard random mask...")
    rand_mask_flat = get_random_mask(num_patches, mask_ratio)
    rand_mask_grid = rand_mask_flat.reshape(num_h, num_w)
    rand_masked_img = apply_mask(img, rand_mask_grid, patch_size)
    
    # 2. Variance-Biased Masking
    print("Generating variance-biased mask...")
    # Using bias_temp = 2.0 to give high weight to details (boats)
    bias_mask_flat = get_variance_biased_mask(variances, mask_ratio, bias_temp=2.0)
    bias_mask_grid = bias_mask_flat.reshape(num_h, num_w)
    bias_masked_img = apply_mask(img, bias_mask_grid, patch_size)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    if is_synthetic:
        # Draw red circles around the tiny boats
        circle1 = plt.Circle((boat1_pos[1]+10, boat1_pos[0]+10), 20, color='red', fill=False, linewidth=2)
        circle2 = plt.Circle((boat2_pos[1]+10, boat2_pos[0]+10), 20, color='red', fill=False, linewidth=2)
        axes[0].add_patch(circle1)
        axes[0].add_patch(circle2)
    axes[0].axis("off")
    
    # Uniform Random Masked Image
    axes[1].imshow(rand_masked_img)
    axes[1].set_title(f"Uniform Random Masking ({int(mask_ratio*100)}% Masked)")
    axes[1].axis("off")
    
    # Variance-Biased Masked Image
    axes[2].imshow(bias_masked_img)
    axes[2].set_title(f"Variance-Biased Masking ({int(mask_ratio*100)}% Masked)")
    axes[2].axis("off")
    
    plt.tight_layout()
    out_path = "variance_masking_demo.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to: {out_path}")

if __name__ == "__main__":
    main()
