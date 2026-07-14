import os
import h5py
import random
import numpy as np
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def create_subset_h5(folder_path, output_h5_path, num_samples):
    print(f"Scanning files in {folder_path}...")
    image_paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                image_paths.append(os.path.join(dirpath, fn))
    
    total_found = len(image_paths)
    print(f"Found {total_found} images.")
    
    if total_found == 0:
        print(f"Warning: No images found in {folder_path}!")
        return

    # Select random subset
    num_to_take = min(num_samples, total_found)
    print(f"Selecting a random subset of {num_to_take} images...")
    random.seed(42)  # For reproducibility
    selected_paths = random.sample(image_paths, num_to_take)
    selected_paths.sort()  # Maintain sorted order for consistent indexing

    # Delete if exists
    if os.path.exists(output_h5_path):
        os.remove(output_h5_path)

    print(f"Creating HDF5 file: {output_h5_path}")
    with h5py.File(output_h5_path, 'w') as hf:
        images_group = hf.create_group('images')
        labels_group = hf.create_group('labels')

        for idx, img_path in enumerate(tqdm(selected_paths, desc="Converting")):
            with open(img_path, 'rb') as f:
                binary_data = f.read()
                binary_np = np.asarray(binary_data)
            
            # Store in H5 using string key
            images_group.create_dataset(str(idx), data=binary_np)
            # Store dummy label 0
            labels_group.create_dataset(str(idx), data=np.array(0, dtype=np.int32))

    print(f"Finished successfully: {output_h5_path}")

if __name__ == "__main__":
    dataset_root = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/images"
    
    # 1. 1000 random images for training
    train_folder = os.path.join(dataset_root, "train")
    train_output = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/maritime-train-1000.h5"
    create_subset_h5(train_folder, train_output, num_samples=1000)
    
    # 2. 200 random images for validation
    val_folder = os.path.join(dataset_root, "val")
    val_output = "/home/dromsis/Images/dataset/sea-vis-data-fan/combined/maritime-val-200.h5"
    create_subset_h5(val_folder, val_output, num_samples=200)
