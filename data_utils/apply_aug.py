import os

from PIL import Image

import torch
import torchvision.transforms as transforms

import numpy as np

# -------------------------- 
# IMPORTING AUGMENTATION OPS 
# --------------------------
from . import aug_ops as augmentations

# -------------------------- 
# FUNCTION: AugMix pre-augment 
# -------------------------- 
def get_preaugment(): 
    return transforms.Compose([ 
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
    ])

# -------------------------- 
# FUNCTION: AugMix core logic 
# -------------------------- 
def augmix(image, preprocess, aug_list, severity=1): 
    preaugment = get_preaugment() 
    x_orig = preaugment(image) 
    x_processed = preprocess(x_orig) 
    if len(aug_list) == 0: return x_processed 
    
    # mixture weights 
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0])) 
    m = np.float32(np.random.beta(1.0, 1.0)) 

    mix = torch.zeros_like(x_processed) 
    for i in range(3): 
        x_aug = x_orig.copy() 
        for _ in range(np.random.randint(1, 4)): 
            x_aug = np.random.choice(aug_list)(x_aug, severity) 
            mix += w[i] * preprocess(x_aug) 
    
    return m * x_processed + (1 - m) * mix

# -------------------------- 
# AUGMIX WRAPPER CLASS 
# --------------------------
class AugMixAugmenter(object): 
    def __init__(self, base_transform, preprocess, n_views=2, use_augmix=True, severity=1):
        self.base_transform = base_transform 
        self.preprocess = preprocess
        self.n_views = n_views 
        self.aug_list = augmentations.augmentations if use_augmix else [] 
        
        self.severity = severity 
        
    def __call__(self, x): 
        base_image = self.preprocess(self.base_transform(x))
        aug_views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)] 
        
        return [base_image] + aug_views



# if __name__ == "__main__":
#     input_path = "./data/imagenet/A/academic_gown/5.jpg"

#     output_dir = "augmix_outputs_temp"
#     os.makedirs(output_dir, exist_ok=True)

#     # Base transformation before AugMix
#     base_transform = transforms.Resize((224, 224))

#     # Standard ImageNet preprocessing
#     preprocess = transforms.ToTensor()

#     # Build the augmenter
#     augmenter = AugMixAugmenter(
#         base_transform=base_transform,
#         preprocess=preprocess,
#         n_views=63,          # number of AugMix views
#         use_augmix=True,
#         severity=1
#     )

#     # Load input image
#     image = Image.open(input_path).convert("RGB")

#     # Apply AugMix
#     images = augmenter(image)   # returns [base, aug1, aug2]

#     # Save outputs
#     for i, img_tensor in enumerate(images):
#         out_img = transforms.ToPILImage()(img_tensor)
#         out_img.save(os.path.join(output_dir, f"augmix_view_{i}.png"))

#     print("Saved:", os.listdir(output_dir))