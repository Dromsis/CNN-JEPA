import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

def analyze_single_image(path):
    try:
        with Image.open(path) as img:
            img.draft('L', (64, 64))
            gray_64 = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = np.array(gray_64)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return diff.flatten(), True
    except:
        return None, False

def main():
    images_dir = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/images"
    labels_dir = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/labels"
    max_workers = 12
    hamming_threshold = 4
    window_size = 10
    
    print("Scanning images directory...")
    all_files = get_image_files(images_dir)
    total_files = len(all_files)
    print(f"Found {total_files} images.")
    
    if total_files == 0:
        print("No images found to process.")
        return
        
    all_files.sort()
    
    print(f"Hashing all {total_files} images in parallel...")
    t0 = time.time()
    
    hashes = []
    valid_paths = []
    
    chunk_size = 10000
    num_chunks = (total_files + chunk_size - 1) // chunk_size
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for chunk_idx in range(num_chunks):
            chunk_files = all_files[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
            results = list(executor.map(analyze_single_image, chunk_files, chunksize=100))
            
            for path, (dhash, ok) in zip(chunk_files, results):
                if ok:
                    hashes.append(dhash)
                    valid_paths.append(path)
                    
    print(f"Hashing completed in {time.time() - t0:.1f}s.")
    num_processed = len(valid_paths)
    
    # Identify duplicates
    print("Identifying duplicates...")
    is_duplicate = np.zeros(num_processed, dtype=bool)
    
    for i in range(num_processed):
        if is_duplicate[i]:
            continue
        limit = min(num_processed, i + window_size + 1)
        h_i = hashes[i]
        for j in range(i + 1, limit):
            if is_duplicate[j]:
                continue
            dist = np.sum(h_i != hashes[j])
            if dist <= hamming_threshold:
                is_duplicate[j] = True
                
    duplicate_indices = np.where(is_duplicate)[0]
    num_duplicates = len(duplicate_indices)
    print(f"Found {num_duplicates} duplicate images to delete.")
    
    if num_duplicates == 0:
        print("Nothing to delete.")
        return
        
    # Delete duplicates
    print("Starting deletion of duplicate images and label files...")
    t0 = time.time()
    
    deleted_images = 0
    deleted_labels = 0
    
    for idx in duplicate_indices:
        img_path = valid_paths[idx]
        
        # Determine corresponding label path
        # Replace "/images/" with "/labels/" and replace extension with ".txt"
        lbl_path = img_path.replace("/images/", "/labels/")
        lbl_path = os.path.splitext(lbl_path)[0] + ".txt"
        
        # Delete image
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                deleted_images += 1
        except Exception as e:
            print(f"Error deleting image {img_path}: {e}")
            
        # Delete label
        try:
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
                deleted_labels += 1
        except Exception as e:
            print(f"Error deleting label {lbl_path}: {e}")
            
        if deleted_images % 5000 == 0 and deleted_images > 0:
            print(f"Deleted {deleted_images}/{num_duplicates} duplicate images...")
            
    deletion_time = time.time() - t0
    print(f"\n--- CLEANUP COMPLETE ---")
    print(f"Deleted {deleted_images} duplicate images.")
    print(f"Deleted {deleted_labels} corresponding label files.")
    print(f"Cleanup took {deletion_time:.1f}s.")
    print(f"Images remaining in dataset: {num_processed - deleted_images}")

if __name__ == "__main__":
    main()
