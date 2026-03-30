import numpy as np
from PIL import Image, ImageOps, ImageEnhance

IMAGE_SIZE = 224

# -----------------------------
# Helper functions
# -----------------------------

def int_parameter(level, maxval):
    ''' Scale level ∈ [0,10] to [0,maxval] and return int '''
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    ''' Scale level ∈ [0,10] to [0,maxval] and return float '''
    return float(level) * maxval / 10


def sample_level(n):
    """Uniform random level in [0.1, n]"""
    return np.random.uniform(low=0.1, high=n)


# -----------------------------
# Augmentation operations
# -----------------------------
def autocontrast(pil_img, level):
    return ImageOps.autocontrast(pil_img)

def equalize(pil_img, level):
    return ImageOps.equalize(pil_img)

def posterize(pil_img, level):
    lvl = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - lvl)

def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)

def solarize(pil_img, level):
    thresh = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - thresh)


def shear_x(pil_img, level):
    lvl = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        lvl = -lvl

    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, lvl, 0, 0, 1, 0),
        resample=Image.BILINEAR
    )

def shear_y(pil_img, level):
    lvl = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        lvl = -lvl

    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, lvl, 1, 0),
        resample=Image.BILINEAR
    )


def translate_x(pil_img, level):
    shift = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        shift = -shift

    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, shift, 0, 1, 0),
        resample=Image.BILINEAR
    )

def translate_y(pil_img, level):
    shift = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        shift = -shift

    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, 0, 1, shift),
        resample=Image.BILINEAR
    )


def color(pil_img, level):
    lvl = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(lvl)

def contrast(pil_img, level):
    lvl = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(lvl)

def brightness(pil_img, level):
    lvl = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(lvl)

def sharpness(pil_img, level):
    lvl = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(lvl)


# -----------------------------
# Augmentation sets
# -----------------------------
augmentations = [
    autocontrast, equalize, posterize, rotate,
    solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = augmentations + [
    color, contrast, brightness, sharpness
]