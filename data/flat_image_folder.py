"""Flat, label-free image dataset.

torchvision.datasets.ImageFolder requires one class sub-directory per label. The pretrain
data lives flat under a single directory (e.g. /data/combined/images/train/*.jpg) with no
class structure, so this just recursively collects every image file under `root` and returns
(PIL image, 0). The dummy 0 label is ignored by the self-supervised objective. Transforms are
applied by the lightly wrapper, so none are applied here.
"""
import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Workers do the decoding; keep OpenCV single-threaded so it doesn't oversubscribe the CPUs.
cv2.setNumThreads(0)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class FlatImageFolder(Dataset):
    # `transform` follows the torchvision contract: lightly's LightlyDataset.from_torch_dataset
    # sets `self.transform` and expects __getitem__ to apply it (it is None until then).
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(IMG_EXTS):
                    self.samples.append(os.path.join(dirpath, fn))
        self.samples.sort()
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        # cv2 (libjpeg-turbo) decodes JPEGs faster than stock PIL; fall back to PIL if cv2
        # can't read the file. Hand a PIL RGB image to the transform, as before.
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            img = Image.open(path).convert("RGB")
        else:
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
