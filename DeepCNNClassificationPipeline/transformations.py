"""
This file contains the used augmentation methods.
"""

from tkinter import Widget
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from utils import Pad, Wrapper, MinMaxScaler

MINCROPSIZE = 384
MAXCROPSIZE = 512
HEIGHT = 512
WIDTH = 512
TRAINHEIGHT = 512 
TRAINWIDTH = 512

# ****************************************
# Training transformations. 
train_transform = Wrapper(transformations=[
    # Composition 1. 
    A.Compose([
        # Composition 2.
        Pad(height=HEIGHT, width=WIDTH, depth=3,
            image_constant=0, mask_constant=0, 
            always_apply=True, 
            p=1.0
        ),
        A.Flip(
            p=0.5
        ),
        A.Rotate(limit=(-10, 10), interpolation=1, 
            border_mode=cv2.BORDER_CONSTANT, value=0,
            mask_value=None, 
            p=0.5
        ),
        A.ColorJitter (
            brightness=0.2,
            contrast=0.2,        
            saturation=0.2, 
            hue=0.2, 
            always_apply=False, 
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=20, alpha_affine=20,
                                interpolation=1, border_mode=0, value=0,
                                mask_value=None, approximate=False, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1,
                            border_mode=0, value=0, mask_value=None,
                            always_apply=False, p=0.5),
            A.Emboss(alpha=(0.2, 0.5),
                        strength=(0.2, 0.7),
                        always_apply=False,
                        p=1.0)
        ], p=0.2),
        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50,
                        p=1.0),
            A.ChannelDropout(channel_drop_range=(1, 2),
                            fill_value=0,
                            p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                val_shift_limit=20, p=1.0)
        ], p=0.3),
        A.OneOf([
            # A.Solarize(threshold=128, p=1.0),
            # A.InvertImg(p=1.0),
            A.ToGray(p=1.0),
            A.ToSepia(p=1.0),
            A.FancyPCA(alpha=0.2, p=1.0), 
            A.Posterize(num_bits=4, 
                        always_apply=True, p=1.0), 
            A.Sharpen(alpha=(0.2, 0.5), 
                    lightness=(0.5, 1.0), 
                    always_apply=False, p=1.0)
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.Blur(blur_limit=(3, 7), p=1.0),
            A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, mode='fast',
                        p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=(3, 7), always_apply=False, p=1.0)
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), mean=50, p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False,
                                elementwise=False, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.09), intensity=(0.1, 0.5),
                        p=1.0),
        ], p=0.5),
    ], p=1.0),
    # Composition 3. 
    A.OneOf([
        A.RandomCrop(384, 384, always_apply=True, p=1.0),
        A.RandomCrop(400, 400, always_apply=True, p=1.0),
        A.RandomCrop(400, 450, always_apply=True, p=1.0),
        A.RandomCrop(450, 400, always_apply=True, p=1.0),
        A.RandomCrop(400, 512, always_apply=True, p=1.0),
        A.RandomCrop(512, 400, always_apply=True, p=1.0),
        A.RandomCrop(384, 512, always_apply=True, p=1.0),
        A.RandomCrop(512, 384, always_apply=True, p=1.0),
        A.RandomCrop(512, 512, always_apply=True, p=1.0)
    ], p=0.8),
    A.Resize(height=TRAINHEIGHT, width=TRAINWIDTH, always_apply=True, p=1.0),
    # Composition 4.
    # MinMaxScaler(always_apply=True, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True, 
                p=1.0
    ),
    ToTensorV2(always_apply=True)
])

# ****************************************
# Validation transformations. 
valid_transform = Wrapper(transformations=[
    # Composition 2.
    Pad(height=HEIGHT, width=WIDTH, depth=3,
        image_constant=0, mask_constant=0, 
        always_apply=True, p=1.0),
    # Composition 3. 
    A.Resize(height=TRAINHEIGHT, width=TRAINWIDTH, always_apply=True, p=1.0),
    # Composition 4.
    # MinMaxScaler(always_apply=True, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True, 
                p=1.0
    ),
    ToTensorV2(always_apply=True)
])

# ****************************************
# Testing transformations. 
test_transform = Wrapper(transformations=[
    # Composition 2.
    Pad(height=HEIGHT, width=WIDTH, depth=3,
        image_constant=0, mask_constant=0, 
        always_apply=True, p=1.0),
    # Composition 3. 
    A.Resize(height=TRAINHEIGHT, width=TRAINWIDTH, always_apply=True, p=1.0),
    # Composition 4.
    # MinMaxScaler(always_apply=True, p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True, 
                p=1.0
    ),
    ToTensorV2(always_apply=True)
])

# ****************************************
# Prediction transformations. 
pred_transform = Wrapper(transformations=[
    # Composition 2.
    Pad(height=HEIGHT, width=WIDTH, depth=3,
        image_constant=0, mask_constant=0, 
        always_apply=True, p=1.0),
    # Composition 3. 
    A.Resize(height=TRAINHEIGHT, width=TRAINWIDTH, always_apply=True),
    # A.RandomSizedCrop(min_max_height=(MINCROPSIZE, MAXCROPSIZE), 
    #                     height=HEIGHT, width=WIDTH, w2h_ratio=1.0, 
    #                     interpolation=1, always_apply=True),
    # Composition 4.
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True, 
                p=1.0
    ),
    ToTensorV2(always_apply=True)
])

transforms = {
    'train': train_transform,
    'valid': valid_transform,
    'test': test_transform,
    'predict': pred_transform
}
