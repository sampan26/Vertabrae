import os
import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops
from scipy import ndimage

def load_masks(mask_dir):
    masks = {}
    vertebrae_name = mask_dir.split(".")[0].split("/")[-1]
    mask = nib.load(os.path.join(mask_dir)).get_fdata()
    masks[vertebrae_name] = mask
    return masks


def remove_small_components(mask, threshold):
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    for region in regions:
        if region.area < threshold:
            labeled_mask[labeled_mask == region.label] = 0
    return labeled_mask > 0

def fill_holes(mask):
    return ndimage.binary_fill_holes(mask)

def enforce_anatomical_constraints(mask):
    # 1. Enforce cylindrical shape:
    #    - Calculate distance transform to identify the "core" of the vertebra.
    #    - Erode and dilate the core to create a cylindrical mask.
    distance_map = ndimage.distance_transform_edt(mask)
    core = distance_map > distance_map.max() * 0.5  # Adjust threshold as needed
    cylindrical_mask = ndimage.binary_dilation(ndimage.binary_erosion(core, iterations=3), iterations=3)

    # 2. Enforce stacking:
    #    - Check if the mask overlaps with the masks of neighboring vertebrae.
    #    - If not, consider removing or adjusting the mask.
    #    (This step requires loading neighboring vertebrae masks and implementing overlap checks)

    # 3. Enforce size and aspect ratio:
    #    - Calculate the bounding box of the mask.
    #    - Check if the size and aspect ratio are within expected ranges.
    #    - If not, consider removing or adjusting the mask.
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    for region in regions:
        bbox = region.bbox
        size = (bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2])
        aspect_ratio = (bbox[3] - bbox[0]) / (bbox[4] - bbox[1])  # Adjust as needed
        if size < 1000 or aspect_ratio < 0.5 or aspect_ratio > 2:  # Adjust thresholds
            mask[labeled_mask == region.label] = 0

    return mask & cylindrical_mask  

def postprocess_vertebrae_masks(mask_data, output_folder):
    # Load masks

    # Apply post-processing steps
    for vertebrae_name, mask in mask_data.items():
        mask = remove_small_components(mask, threshold=100)
        mask = fill_holes(mask)
        mask = enforce_anatomical_constraints(mask)  # Apply anatomical constraints

        # Refine based on combined labels (optional)

        # Save processed mask
        output_filename = os.path.join(output_folder, f"{vertebrae_name}_processed.nii.gz")
        nib.save(nib.Nifti1Image(mask, np.eye(4)), output_filename)


def main():
    """Processes all vertebrae masks in a directory."""
    mask_dir = "/content/segmentations"  # Replace with actual directory

    for filename in os.listdir(mask_dir):
        if filename.endswith(".nii.gz"):
            mask_path = os.path.join(mask_dir, filename)
            mask_data = load_masks(mask_path)
            processed_mask_data = postprocess_vertebrae_masks(mask_data)
            save_mask(processed_mask_data, mask_path)

if __name__ == "__main__":
    main()