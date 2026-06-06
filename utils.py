import os
import cv2
import numpy as np


def yolo_segmentation_to_unet_mask(
    img_path,
    txt_file,
    output_mask_path=None,
):
    """
    Convert YOLO segmentation annotations to U-Net style mask.

    Parameters
    ----------
    img_path : str
        Path to image.

    txt_file : str
        Path to YOLO segmentation label file.

    output_mask_path : str or None
        If provided, saves the generated mask.

    class_values : dict or None
        Example:
            {
                0: 1,
                1: 2,
                2: 3
            }

        Maps YOLO class_id -> mask pixel value.

        If None:
            all objects use fill_value.

    fill_value : int
        Default pixel value for objects if class_values is None.

    Returns
    -------
    mask : np.ndarray
        Generated mask image.
    """

    # Read image
    image = cv2.imread(img_path)

    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")

    height, width = image.shape[:2]

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Check label file
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Label file not found: {txt_file}")

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # Need at least:
        # class_id x1 y1 x2 y2 x3 y3 ...
        if len(parts) < 7:
            continue

        class_id = int(parts[0])

        coords = list(map(float, parts[1:]))

        # Convert normalized coords to pixel coords
        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i + 1] * height)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        color = class_id + 1

        # class_values = None
        # if class_number:
        #     class_values = { i:i+1 for i in range(class_number)}


        # Determine mask value
        # if class_values is not None:
        #     color = class_values.get(class_id, 0)
        # else:
        #     color = fill_value

        # Fill polygon
        cv2.fillPoly(mask, [points], color)

    # Save if requested
    if output_mask_path is not None:
        cv2.imwrite(output_mask_path, mask)

    return mask



