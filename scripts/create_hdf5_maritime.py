import os
import h5py
import numpy as np
from tqdm import tqdm

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

def convert_folder_to_h5(folder_path, output_h5_path):
    print(f"Scanning files in {folder_path}...")
    image_paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                image_paths.append(os.path.join(dirpath, fn))
    
    image_paths.sort()
    num_images = len(image_paths)
    print(f"Found {num_images} images to convert.")
    
    if num_images == 0:
        print(f"Warning: No images found in {folder_path}!")
        return

    # Delete if exists
    if os.path.exists(output_h5_path):
        os.remove(output_h5_path)

    print(f"Creating HDF5 file: {output_h5_path}")
    with h5py.File(output_h5_path, 'w') as hf:
        images_group = hf.create_group('images')
        labels_group = hf.create_group('labels')

        for idx, img_path in enumerate(tqdm(image_paths, desc="Converting")):
            with open(img_path, 'rb') as f:
                binary_data = f.read()
                binary_np = np.asarray(binary_data)
            
            # Store in H5 using string key
            images_group.create_dataset(str(idx), data=binary_np)
            # Store dummy label 0
            labels_group.create_dataset(str(idx), data=np.array(0, dtype=np.int32))

    print(f"Finished successfully: {output_h5_path}")

if __name__ == "__main__":
    dataset_root = "/home/dromsis/Pictures/dataset/combined/images"
    
    # Convert Train split
    train_folder = os.path.join(dataset_root, "train")
    train_output = "/home/dromsis/Pictures/dataset/combined/maritime-train.h5"
    convert_folder_to_h5(train_folder, train_output)
    
    # Convert Val split
    val_folder = os.path.join(dataset_root, "val")
    val_output = "/home/dromsis/Pictures/dataset/combined/maritime-val.h5"
    convert_folder_to_h5(val_folder, val_output)

    # Convert Test split
    test_folder = os.path.join(dataset_root, "test")
    test_output = "/home/dromsis/Pictures/dataset/combined/maritime-test.h5"
    convert_folder_to_h5(test_folder, test_output)
