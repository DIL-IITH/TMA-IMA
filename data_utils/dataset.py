import os

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension

from torchvision import transforms

class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # -------------------------
        # Discover classes (sorted)
        # -------------------------
        classes = [
            d.name for d in os.scandir(root)
            if d.is_dir()
        ]
        classes.sort()

        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        # -------------------------
        # Gather samples (sorted)
        # -------------------------
        samples = []
        for cls_name in classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_folder = os.path.join(root, cls_name)

            for root_dir, _, filenames in os.walk(cls_folder):
                filenames.sort()
                for fname in filenames:
                    if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                        path = os.path.join(root_dir, fname)
                        samples.append((path, cls_idx))

        if len(samples) == 0:
            raise RuntimeError(f"Found 0 images in {root}")

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label