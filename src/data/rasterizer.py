import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple


def coordinates_to_polygon(
    x1_coords: List[float],
    y1_coords: List[float],
    x2_coords: List[float],
    y2_coords: List[float]
) -> List[Tuple[float, float]]:
    """
    Assemble the paired Simpson's disk coordinates into a continuous closed polygon boundary.
    Coordinates trace:
    Base to Apex along side 1 (X1, Y1), then Apex to Base along side 2 (X2, Y2).
    """
    side1 = list(zip(x1_coords, y1_coords))
    side2 = list(zip(x2_coords, y2_coords))
    
    # Connect side1 forward and side2 reversed to close the loop
    polygon = side1 + side2[::-1]
    return polygon


def rasterize_lv_mask(
    x1_coords: List[float],
    y1_coords: List[float],
    x2_coords: List[float],
    y2_coords: List[float],
    orig_shape: Tuple[int, int] = (112, 112),
    target_shape: Tuple[int, int] = (112, 112)
) -> np.ndarray:
    """
    Rasterize Simpson's disk coordinate tracings into a filled binary mask.
    
    Args:
        x1_coords, y1_coords: Left-side boundary coordinates.
        x2_coords, y2_coords: Right-side boundary coordinates.
        orig_shape: (orig_height, orig_width) of coordinates in CSV.
        target_shape: (target_height, target_width) of output mask (e.g., 112x112).
        
    Returns:
        np.ndarray of shape target_shape, boolean/uint8 binary mask (1 inside LV, 0 outside).
    """
    if not x1_coords or len(x1_coords) < 3:
        return np.zeros(target_shape, dtype=np.uint8)

    orig_h, orig_w = orig_shape
    target_h, target_w = target_shape

    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)

    # Scale coordinates to target resolution
    scaled_x1 = [x * scale_x for x in x1_coords]
    scaled_y1 = [y * scale_y for y in y1_coords]
    scaled_x2 = [x * scale_x for x in x2_coords]
    scaled_y2 = [y * scale_y for y in y2_coords]

    poly = coordinates_to_polygon(scaled_x1, scaled_y1, scaled_x2, scaled_y2)

    # Render polygon onto a PIL grayscale mask
    mask_img = Image.new("L", (target_w, target_h), 0)
    ImageDraw.Draw(mask_img).polygon(poly, outline=1, fill=1)

    return np.array(mask_img, dtype=np.uint8)
