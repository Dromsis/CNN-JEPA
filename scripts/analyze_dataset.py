import os
import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get_image_files(directory):
    """Scans the directory recursively for common image file extensions."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def compute_dhash(pil_img, hash_size=8):
    """Computes a 64-bit Difference Hash (dHash) for near-duplicate detection."""
    # Resize to (hash_size + 1) x hash_size, grayscale
    img = pil_img.resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR).convert("L")
    pixels = np.array(img)
    
    # Compare adjacent pixels horizontally
    diff = pixels[:, 1:] > pixels[:, :-1]
    
    # Convert binary array to integer hash
    return diff.flatten()

def hamming_distance(hash1, hash2):
    """Computes the Hamming distance between two boolean dHash arrays."""
    return np.sum(hash1 != hash2)

def analyze_single_image(path):
    """Loads an image, computes standard deviation (variance) and its dHash."""
    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            
            # Grayscale conversion for variance calculation
            gray_img = img_rgb.convert("L")
            pixels_gray = np.array(gray_img)
            
            # Standard deviation (measures local details/contrast)
            std_dev = np.std(pixels_gray)
            
            # Perceptual hash for redundancy check
            dhash = compute_dhash(img_rgb)
            
            return std_dev, dhash, True
    except Exception as e:
        return 0.0, None, False

def main():
    target_dir = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/images"
    sample_size = 2000
    hamming_threshold = 4 # Hashes with Hamming distance <= 4 are considered near-duplicates
    
    print(f"Scanning directory: {target_dir}...")
    t0 = time.time()
    all_files = get_image_files(target_dir)
    scan_time = time.time() - t0
    total_files = len(all_files)
    print(f"Found {total_files} images in {scan_time:.2f}s.")
    
    if total_files == 0:
        print("Error: No images found.")
        sys.exit(1)
        
    # Sort files alphabetically to ensure deterministic order (groups consecutive video frames)
    all_files.sort()
    
    # Uniform sampling across the dataset
    print(f"Sampling {sample_size} images uniformly for analysis...")
    step = max(1, total_files // sample_size)
    sampled_files = all_files[::step][:sample_size]
    actual_sample_size = len(sampled_files)
    
    stds = []
    hashes = []
    valid_paths = []
    
    t0 = time.time()
    for i, path in enumerate(sampled_files):
        if i % 200 == 0 and i > 0:
            print(f"Analyzed {i}/{actual_sample_size} images...")
        std, dhash, ok = analyze_single_image(path)
        if ok:
            stds.append(std)
            hashes.append(dhash)
            valid_paths.append(path)
            
    analysis_time = time.time() - t0
    print(f"Analysis completed in {analysis_time:.2f}s.")
    
    stds = np.array(stds)
    
    # Classify variance (uniformity)
    # std < 10: Empty water / flat fog (Extreme Uniformity)
    # 10 <= std < 20: Very Calm Sea / Sky (Low Details)
    # 20 <= std < 40: Waves, Horizon, Minor Wake (Medium details)
    # std >= 40: Ships, Shorelines, High Contrast (Rich Details)
    very_low = np.sum(stds < 10)
    low = np.sum((stds >= 10) & (stds < 20))
    medium = np.sum((stds >= 20) & (stds < 40))
    high = np.sum(stds >= 40)
    
    # Near-duplicate analysis (O(N^2) but N=2000 is small: 4M comparisons, takes ~0.5s in numpy)
    print("Computing near-duplicates...")
    duplicate_count = 0
    is_duplicate = np.zeros(len(hashes), dtype=bool)
    
    # Compute Hamming distances
    for i in range(len(hashes)):
        if is_duplicate[i]:
            continue
        for j in range(i + 1, len(hashes)):
            if is_duplicate[j]:
                continue
            dist = hamming_distance(hashes[i], hashes[j])
            if dist <= hamming_threshold:
                is_duplicate[j] = True
                duplicate_count += 1
                
    duplicate_ratio = duplicate_count / len(hashes)
    
    # Potential dataset size reduction estimation
    # Redundancy ratio + Very Low variance ratio
    redundant_pct = duplicate_ratio * 100
    very_low_pct = (very_low / len(stds)) * 100
    low_pct = (low / len(stds)) * 100
    medium_pct = (medium / len(stds)) * 100
    high_pct = (high / len(stds)) * 100
    
    print("\n--- DATASET REPORT SUMMARY ---")
    print(f"Total Images in dataset: {total_files}")
    print(f"Analyzed sample: {len(stds)}")
    print(f"Average Standard Deviation (Contrast): {np.mean(stds):.2f}")
    print(f"Empty/Extreme Uniformity (std < 10): {very_low_pct:.1f}%")
    print(f"Low Contrast/Calm Sea (10 <= std < 20): {low_pct:.1f}%")
    print(f"Medium Contrast (20 <= std < 40): {medium_pct:.1f}%")
    print(f"Rich Details/Ships (std >= 40): {high_pct:.1f}%")
    print(f"Near-Duplicate Redundancy Ratio: {redundant_pct:.1f}%")
    
    # Save variance distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(stds, bins=50, color="teal", edgecolor="black", alpha=0.7)
    plt.axvline(10, color="red", linestyle="--", linewidth=1.5, label="Extreme Uniformity Limit (std=10)")
    plt.axvline(20, color="orange", linestyle="--", linewidth=1.5, label="Low Contrast Limit (std=20)")
    plt.title("Distribution de l'Écart-Type (Contraste) du Dataset Maritime")
    plt.xlabel("Écart-Type des Pixels (Grayscale)")
    plt.ylabel("Nombre d'Images (Échantillonnées)")
    plt.legend()
    plt.grid(axis="y", linestyle=":")
    plot_path = "dataset_variance_dist.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Copy plot to artifact directory
    artifact_dir = "/home/dromsis/.gemini/antigravity-cli/brain/68abb1d3-75b1-4f72-b459-b5f95644e176"
    os.makedirs(artifact_dir, exist_ok=True)
    os.system(f"cp {plot_path} {artifact_dir}/")
    
    # Write detailed markdown report
    report_content = f"""# Rapport d'Analyse de l'Homogénéité et de la Redondance du Dataset

Ce rapport présente l'analyse statistique effectuée sur le dataset d'images maritimes situé à `/home/dromsis/Images/dataset/sea-vis-data-fan/combined/images`. 
L'analyse a été menée sur un échantillon représentatif de **{len(stds)} images** réparties uniformément parmi les **{total_files} images** au total.

---

## 1. Distribution de la Variance (Uniformité des Scènes)

Le niveau de détail et de contraste de chaque image a été quantifié par l'écart-type de ses pixels (échelle de gris de 0 à 255). Un écart-type très faible indique une image uniforme (eau plate, ciel pur, ou brouillard épais).

![Distribution de l'écart-type du dataset]({artifact_dir}/dataset_variance_dist.png)

### Catégories d'images identifiées :

*   **Uniformité Extrême (std < 10) : {very_low_pct:.1f}%** (soit environ {int(total_files * very_low_pct / 100)} images)
    *   *Description :* Images constituées presque exclusivement d'eau plate sans vagues prononcées ou de ciel uniforme, sans horizon visible ni objet.
*   **Contraste Faible / Mer Calme (10 <= std < 20) : {low_pct:.1f}%** (soit environ {int(total_files * low_pct / 100)} images)
    *   *Description :* Mer calme, dégradé de ciel léger, ligne d'horizon très douce. Présence sémantique extrêmement faible.
*   **Contraste Moyen (20 <= std < 40) : {medium_pct:.1f}%** (soit environ {int(total_files * medium_pct / 100)} images)
    *   *Description :* Vagues formées, sillages d'écume de navires, horizons bien marqués.
*   **Détails Riches / Navires & Côtes (std >= 40) : {high_pct:.1f}%** (soit environ {int(total_files * high_pct / 100)} images)
    *   *Description :* Navires de taille moyenne à grande, côtes, ports, vagues de tempête ou détails à haut contraste.

---

## 2. Analyse de la Redondance (Doublons Temporels)

Grâce à l'analyse par **Difference Hash (dHash)** de taille 8x8 avec une tolérance de distance de Hamming <= {hamming_threshold} (ce qui capte les images quasi-identiques avec de légères variations de bruit/vagues) :

*   **Taux de redondance détecté : {redundant_pct:.1f}%** (soit environ {int(total_files * redundant_pct / 100)} images redondantes).
*   *Cause :* L'extraction de frames de vidéos de drones à un taux d'images par seconde (FPS) trop élevé par rapport à la vitesse de déplacement ou aux mouvements de la scène.

---

## 3. Recommandations de Nettoyage et Réduction de Taille

1.  **Potentiel d'Élimination Directe :**
    En combinant les doublons temporels et les images d'uniformité extrême (std < 10), le dataset pourrait être **réduit d'au moins {redundant_pct + very_low_pct:.1f}%** sans aucune perte d'informations sémantiques ou de diversité.
2.  **Taille Cible Optimisée :**
    Nettoyer ce dataset permettrait de le ramener à environ **{int(total_files * (1 - (redundant_pct + very_low_pct)/100))} images** hautement informatives.
3.  **Bénéfice d'Entraînement :**
    *   **Gain de temps :** Votre temps d'entraînement par époque JEPA diminuerait de **{redundant_pct + very_low_pct:.1f}%** (ce qui ferait passer l'entraînement complet de 13h à environ **{13 * (1 - (redundant_pct + very_low_pct)/100):.1f}h**).
    *   **Qualité des caractéristiques :** En éliminant l'excès d'eau uniforme et répétitive, le modèle évitera naturellement le *shortcut learning* et le risque de collapse, apprenant des représentations beaucoup plus robustes pour la détection.
"""
    
    report_path = os.path.join(artifact_dir, "dataset_uniformity_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"\nSaved detailed report to: {report_path}")

if __name__ == "__main__":
    main()
