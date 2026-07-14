import os
import shutil
import numpy as np
from PIL import Image

def get_image_files(directory):
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def compute_dhash(path):
    try:
        with Image.open(path) as img:
            img_resized = img.resize((9, 8), Image.Resampling.BILINEAR).convert("L")
            pixels = np.array(img_resized)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return diff.flatten()
    except:
        return None

def main():
    target_dir = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/images"
    artifact_dir = "/home/dromsis/.gemini/antigravity-cli/brain/68abb1d3-75b1-4f72-b459-b5f95644e176"
    dup_out_dir = os.path.join(artifact_dir, "duplicates")
    os.makedirs(dup_out_dir, exist_ok=True)
    
    print("Scanning directory for files...")
    files = get_image_files(target_dir)
    files.sort()
    
    print("Searching for 5 pairs of consecutive duplicates...")
    found_pairs = []
    
    # We will step through the files and compute dHash for consecutive ones
    i = 0
    while i < len(files) - 1 and len(found_pairs) < 5:
        path1 = files[i]
        path2 = files[i+1]
        
        # Ensure they belong to the same video/sequence (filenames should be similar)
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        
        # Simple heuristic: if filenames are very different, they aren't consecutive video frames
        # For video frames, names are usually like frame_00001.jpg, frame_00002.jpg
        # If prefix differs, skip
        if len(name1) != len(name2) or name1[:5] != name2[:5]:
            i += 1
            continue
            
        hash1 = compute_dhash(path1)
        hash2 = compute_dhash(path2)
        
        if hash1 is not None and hash2 is not None:
            dist = np.sum(hash1 != hash2)
            if dist <= 1: # Very high similarity (consecutive near-duplicates)
                found_pairs.append((path1, path2, dist))
                print(f"Found Pair {len(found_pairs)}: {name1} <-> {name2} (Hamming Dist: {dist})")
                # Skip past this pair to find a different sequence
                i += 50 
                continue
        i += 1
        
    if not found_pairs:
        print("No duplicate pairs found with Hamming distance <= 1.")
        return
        
    # Copy files and generate markdown report
    markdown_content = """# Exemples de Doublons Visuels Détectés dans le Dataset

Ce rapport montre **5 paires d'images consécutives** qui ont été identifiées comme des doublons temporels quasi-identiques (distance de Hamming <= 1) dans votre dataset `/home/dromsis/Images/dataset/sea-vis-data-fan/combined/images`.

---

"""
    
    for idx, (p1, p2, dist) in enumerate(found_pairs):
        ext1 = os.path.splitext(p1)[1]
        ext2 = os.path.splitext(p2)[1]
        
        name1 = f"pair_{idx+1}_a{ext1}"
        name2 = f"pair_{idx+1}_b{ext2}"
        
        dest1 = os.path.join(dup_out_dir, name1)
        dest2 = os.path.join(dup_out_dir, name2)
        
        shutil.copy2(p1, dest1)
        shutil.copy2(p2, dest2)
        
        rel_path1 = f"duplicates/{name1}"
        rel_path2 = f"duplicates/{name2}"
        
        markdown_content += f"""### Paire n°{idx+1} : {os.path.basename(p1)} et {os.path.basename(p2)} (Distance Hamming : {dist})

| Image A (Frame N) | Image B (Frame N+1) |
| :---: | :---: |
| ![{name1}]({artifact_dir}/{rel_path1}) | ![{name2}]({artifact_dir}/{rel_path2}) |

---
"""

    report_path = os.path.join(artifact_dir, "duplicate_samples.md")
    with open(report_path, "w") as f:
        f.write(markdown_content)
        
    print(f"Copied duplicate samples and saved report to: {report_path}")

if __name__ == "__main__":
    main()
