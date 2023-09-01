### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib
from typing import Union, Callable
import time
import random

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import torchio as tio

### Internal Imports ###

from augmentation import aug
from input_output import volumetric as v
from helpers import utils as u

########################

class SEGADataset(tc.utils.data.Dataset):
    """
    TODO
    """
    def __init__(
        self,
        data_path : Union[str, pathlib.Path],
        csv_path : Union[str, pathlib.Path],
        iteration_size : int = -1,
        volumetric_transforms : Callable[[tc.Tensor, dict], tuple[tc.Tensor, dict]] = None,
        volumetric_transforms_params : dict = {},
        loading_params : dict = {},
        return_load_time : bool=False,
        return_paths : bool=False,
        torchio_transforms = None,
        initial_thresholds=None):
        """
        TODO
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(self.csv_path)
        self.iteration_size = iteration_size
        self.volumetric_transforms = volumetric_transforms # Must follow to structure from the "augmentation" module
        self.volumetric_transforms_params = volumetric_transforms_params
        self.torchio_transforms = torchio_transforms
        self.loading_params = loading_params
        self.return_load_time = return_load_time
        self.return_paths = return_paths
        self.initial_thresholds=initial_thresholds
        if self.iteration_size > len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=self.iteration_size, replace=True).reset_index(drop=True)

    def __len__(self):
        if self.iteration_size < 0:
            return len(self.dataframe)
        else:
            return self.iteration_size
        
    def shuffle(self):
        if self.iteration_size > 0:
            self.dataframe = self.dataframe.sample(n=len(self.dataframe), replace=False).reset_index(drop=True)

    def __getitem__(self, idx):
        current_case = self.dataframe.loc[idx]
        input_path = self.data_path / current_case['Input Path']
        ground_truth_path = self.data_path / current_case['Ground-Truth Path']
        b_t = time.time()
        input_loader = v.VolumetricLoader(**self.loading_params).load(input_path)
        input, spacing, input_metadata = input_loader.volume, input_loader.spacing, input_loader.metadata
        if self.initial_thresholds is not None:
            input[input < self.initial_thresholds[0]] = self.initial_thresholds[0]
            input[input > self.initial_thresholds[1]] = self.initial_thresholds[1]
            input = u.normalize_to_window(input, self.initial_thresholds[0], self.initial_thresholds[1])
        input = (input - tc.min(input)) / (tc.max(input) - tc.min(input))
        gt_loader = v.VolumetricLoader(**self.loading_params).load(ground_truth_path)
        ground_truth = gt_loader.volume
        output = (input, ground_truth)
        e_t = time.time()
        loading_time = e_t - b_t

        b_t = time.time()
        if self.volumetric_transforms is not None:
            output, metadata = aug.apply_transform(*output, transform=self.volumetric_transforms, **{**self.volumetric_transforms_params, **{"old_spacing" : spacing}})
            try:
                spacing = tc.Tensor(metadata['spacing'])
            except:
                pass
            
        if self.torchio_transforms is not None:
            subject = tio.Subject(
            input = tio.ScalarImage(tensor=output[0]),
            label = tio.LabelMap(tensor=output[1]))
            result = self.torchio_transforms(subject)
            transformed_input = result['input'].data
            transformed_input[0] = u.normalize(transformed_input[0])
            transformed_gt = result['label'].data
            output = (transformed_input, transformed_gt)    
            
        e_t = time.time()
        augmentation_time = e_t - b_t
        total_time = (loading_time, augmentation_time)
        
        if self.return_load_time:
            return *output, spacing, total_time
        if self.return_paths:
            return *output, spacing, dict(**current_case, **input_metadata)
        else:
            return *output, spacing
