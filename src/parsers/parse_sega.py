### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pathlib
import shutil

### External Imports ###
import numpy as np
import torch as tc
import pandas as pd
import SimpleITK as sitk

### Internal Imports ###
from paths import paths as p

from input_output import volumetric as io
from visualization import volumetric as vis
from augmentation import volumetric as aug_vol
from input_output import utils_io as uio
from helpers import utils as u
from preprocessing import preprocessing_volumetric as pre_vol



def parse_sega():
    input_path = p.raw_sega_path
    output_path = p.parsed_sega_path
    output_original_path = output_path / "Original"
    output_shape_path = output_path / "Shape_400_400_400"
    output_csv_path = output_path / "dataset.csv"
    if not os.path.exists(output_original_path):
        os.makedirs(output_original_path)
    if not os.path.exists(output_shape_path):
        os.makedirs(output_shape_path)

    ### Parsing Params ###
    output_size = (400, 400, 400)
    dataset_names = ["Dongyang", "KiTS", 'Rider']
    device = "cuda:0"

    ### Parsing ###
    dataframe = []
    for dataset_name in dataset_names:
        cases = os.listdir(os.path.join(input_path, dataset_name))
        cases = [item for item in cases if os.path.isdir(os.path.join(input_path, dataset_name, item))]
        for idx, case in enumerate(cases):
            case_path = os.path.join(input_path, dataset_name, case)
            print()
            print(f"Current case: {case_path}")
            volume_path = os.path.join(case_path, f"{case.split(' ')[0]}.nrrd")
            segmentation_path = os.path.join(case_path, f"{case.split(' ')[0]}.seg.nrrd")
            volume, segmentation, volume_to_shape, segmentation_to_shape, spacing = parse_case2(volume_path, segmentation_path, output_size, device)
            shape = volume.shape

            out_volume_path = f"{dataset_name}_{idx}.nrrd"
            out_segmentation_path = f"{dataset_name}_{idx}.seg.nrrd"
            dataframe.append((out_volume_path, out_segmentation_path))

            volume_to_shape_path = output_shape_path / out_volume_path
            segmentation_to_shape_path = output_shape_path / out_segmentation_path

            new_spacing = tuple(np.array(spacing) * np.array(shape) / np.array(output_size))
            print(f"Spacing: {spacing}")
            print(f"New Spacing: {new_spacing}")

            to_save = sitk.GetImageFromArray(volume_to_shape.swapaxes(2, 1).swapaxes(1, 0))
            to_save.SetSpacing(new_spacing)
            sitk.WriteImage(to_save, str(volume_to_shape_path))

            to_save = sitk.GetImageFromArray(segmentation_to_shape.swapaxes(2, 1).swapaxes(1, 0))
            to_save.SetSpacing(new_spacing)
            sitk.WriteImage(to_save, str(segmentation_to_shape_path), useCompression=True)


    dataframe = pd.DataFrame(dataframe, columns=['Input Path', 'Ground-Truth Path'])
    dataframe.to_csv(output_csv_path, index=False)


def parse_case(volume_path, segmentation_path, output_size, device):
    volume = sitk.ReadImage(volume_path)
    segmentation = sitk.ReadImage(segmentation_path)
    spacing = volume.GetSpacing()
    volume = sitk.GetArrayFromImage(volume).swapaxes(0, 1).swapaxes(1, 2)
    segmentation = sitk.GetArrayFromImage(segmentation).swapaxes(0, 1).swapaxes(1, 2)
    print(f"Volume shape: {volume.shape}")
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Spacing: {spacing}")

    volume_tc = tc.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    segmentation_tc = tc.from_numpy(segmentation.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    print(f"Volume TC shape: {volume_tc.shape}")
    print(f"Segmentation TC shape: {segmentation_tc.shape}")

    resampled_volume_tc = pre_vol.resample_tensor(volume_tc, (1, 1, *output_size), mode='bilinear')
    resampled_segmentation_tc = pre_vol.resample_tensor(segmentation_tc, (1, 1, *output_size), mode='nearest')

    print(f"Resampled Volume TC shape: {resampled_volume_tc.shape}")
    print(f"Resampled Segmentation TC shape: {resampled_segmentation_tc.shape}")

    volume_tc = volume_tc[0, 0, :, :, :].detach().cpu().numpy()
    resampled_volume_tc = resampled_volume_tc[0, 0, :, :, :].detach().cpu().numpy()

    segmentation_tc = segmentation_tc[0, 0, :, :, :].detach().cpu().numpy().astype(np.uint8)
    resampled_segmentation_tc = resampled_segmentation_tc[0, 0, :, :, :].detach().cpu().numpy().astype(np.uint8)

    return volume_tc, segmentation_tc, resampled_volume_tc, resampled_segmentation_tc, spacing


def split_dataframe(input_csv_path, training_csv_path, validation_csv_path, split_ratio = 0.9, seed=1234):
    dataframe = pd.read_csv(input_csv_path)
    dataframe = dataframe.sample(frac=1, random_state=seed)
    training_dataframe = dataframe[:int(split_ratio*len(dataframe))]
    validation_dataframe = dataframe[int(split_ratio*len(dataframe)):]
    print(f"Dataset size: {len(dataframe)}")
    print(f"Training dataset size: {len(training_dataframe)}")
    print(f"Validation dataset size: {len(validation_dataframe)}")
    if not os.path.isdir(os.path.dirname(training_csv_path)):
        os.makedirs(os.path.dirname(training_csv_path))
    if not os.path.isdir(os.path.dirname(validation_csv_path)):
        os.makedirs(os.path.dirname(validation_csv_path))
    training_dataframe.to_csv(training_csv_path)
    validation_dataframe.to_csv(validation_csv_path)


def save_image(volume, spacing, save_path, origin=None, direction=None):
    image  = sitk.GetImageFromArray(volume.swapaxes(2, 1).swapaxes(1, 0))
    image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    sitk.WriteImage(image, str(save_path))

def copy_file(input_path, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    shutil.copy(input_path, output_path)

def create_folder(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))    


def run():
    parse_sega()
    split_dataframe(p.parsed_sega_path / "dataset.csv", p.parsed_sega_path / "training_dataset.csv", p.parsed_sega_path / "validation_dataset.csv", split_ratio = 0.8, seed=1234)
    
if __name__ == "__main__":
    run()