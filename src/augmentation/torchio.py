### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Callable
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import torchio as tio

### Internal Imports ###
from augmentation import aug
from preprocessing import preprocessing_volumetric as pre_vol
from helpers import utils as u

########################

### SEGA ###


def sega_final_transforms_8a():
    random_flip = tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=5, translation=15, p=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.4, 0.4), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.65, 1.25), degrees=10, translation=25, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.5), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.03), p=0.5)
    random_blur = tio.RandomBlur(std=(0, 0.8), p=0.5)
    
    transform_dict = {
        random_flip : 1,
        random_motion : 1,
        random_gamma : 1,
        random_affine : 1,
        random_anisotropy : 1,
        random_noise : 1,
        random_blur : 1,
    }
    transform_1 = tio.OneOf(transform_dict)
    transform_2 = tio.OneOf(transform_dict)
    transform_3 = tio.OneOf(transform_dict)
    transform_4 = tio.OneOf(transform_dict)
    transform_5 = tio.OneOf(transform_dict)
    transform_6 = tio.OneOf(transform_dict)
    transform_7 = tio.OneOf(transform_dict)
    transforms = tio.Compose([transform_1, transform_2, transform_3, transform_4, transform_5, transform_6, transform_7])
    return transforms

def sega_final_transforms_8b():
    random_flip = tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
    random_motion = tio.RandomMotion(degrees=5, translation=15, p=0.5)
    random_gamma = tio.RandomGamma(log_gamma=(-0.4, 0.4), p=0.5)
    random_affine = tio.RandomAffine(scales=(0.65, 1.25), degrees=10, translation=20, p=0.5)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.1, 2.5), p=0.5)
    random_noise = tio.RandomNoise(std=(0, 0.03), p=0.5)
    random_blur = tio.RandomBlur(std=(0, 0.8), p=0.5)
    transforms = tio.Compose([random_flip, random_motion, random_gamma, random_affine, random_anisotropy, random_noise, random_blur])
    return transforms


def sega_final_transforms_8():
    transform_1 = sega_final_transforms_8a()
    transform_2 = sega_final_transforms_8b()
    transform_dict = {
        transform_1 : 0.5,
        transform_2 : 0.5,
    }
    transforms = tio.OneOf(transform_dict)
    return transforms


def sega_intensity_geometric_transforms():
    random_motion = tio.RandomMotion(degrees=5, translation=5, p=0.2)
    random_ghosting = tio.RandomGhosting(p=0.2)
    random_spike = tio.RandomSpike(p=0.2)
    random_bias_field = tio.RandomBiasField(p=0.2)
    random_blur = tio.RandomBlur(std=(0, 0.5), p=0.2)
    random_noise = tio.RandomNoise(std=(0, 0.001), p=0.2)
    random_swap = tio.RandomSwap(patch_size=15, num_iterations=20, p=0.2)
    random_gamma = tio.RandomGamma(p=0.2)
    random_flip = tio.RandomFlip(axes=(0, 1), flip_probability=0.3)
    random_affine = tio.RandomAffine(scales=(0.85, 1.15), degrees=5, translation=5, p=0.3)
    random_anisotropy = tio.RandomAnisotropy(downsampling=(1.2, 2.0), p=0.3)
    transforms = tio.Compose([random_motion, random_ghosting, random_spike, random_bias_field, random_blur, random_noise, random_swap, random_gamma, random_flip, random_affine, random_anisotropy])
    return transforms






















