import SimpleITK as sitk
import os
import numpy as np
from tqdm import tqdm

def calculate_staple_consensus(mask_paths):
    """
    Calculate the STAPLE consensus for binary masks.

    Parameters:
    - mask_paths: List of paths to the binary mask images.

    Returns:
    - consensus_mask: Binary image representing the STAPLE consensus.
    """

    # Read the masks and convert them to binary (0 and 1)
    masks = [sitk.ReadImage(mask_path, sitk.sitkUInt8) for mask_path in mask_paths]
    
    # Normalize masks from [0, 255] to [0, 1] (STAPLE requires binary input)
    masks = [sitk.Cast(m > 127, sitk.sitkUInt8) for m in masks]  # Any value >127 becomes 1

    # Calculate the STAPLE consensus
    staple_filter = sitk.STAPLEImageFilter()
    consensus_mask = staple_filter.Execute(masks)

    return consensus_mask

# Define the output directory
out_dir = 'save_masks/combined_mask'
os.makedirs(out_dir, exist_ok=True)

# Define the folders containing masks
mask_folders = [
    'save_masks/masks_b3',
    'save_masks/masks_b4',
    'save_masks/masks_b5'
]

# Choose a reference folder (for matching filenames)
reference_folder = mask_folders[0]

# Process each image in the reference folder
for image_filename in tqdm(os.listdir(reference_folder), desc='Processing STAPLE consensus'):
    mask_paths = [os.path.join(folder, image_filename) for folder in mask_folders]

    # Ensure all mask files exist
    if not all(os.path.exists(mask_path) for mask_path in mask_paths):
        print(f"Skipping {image_filename} as not all masks are present.")
        continue

    # Compute STAPLE consensus
    consensus = calculate_staple_consensus(mask_paths)

    # Convert STAPLE output to numpy array and check the value range
    consensus_array = sitk.GetArrayFromImage(consensus)
    print(f"Consensus range for {image_filename}: min={consensus_array.min()}, max={consensus_array.max()}")

    # Apply threshold to create a binary mask
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(0.3)  
    threshold_filter.SetUpperThreshold(1.0)
    threshold_filter.SetInsideValue(1)
    threshold_filter.SetOutsideValue(0)
    binary_consensus = threshold_filter.Execute(consensus)

    binary_consensus = sitk.Cast(binary_consensus * 255, sitk.sitkUInt8)

 
    save_path = os.path.join(out_dir, image_filename)
    sitk.WriteImage(binary_consensus, save_path)

print("✅ STAPLE consensus masks saved successfully!")
