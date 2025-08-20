from typing import Tuple, List, Union

def normalize_bbox_coordinates(x1, y1, x2, y2, 
                              image_width: int, 
                              image_height: int) -> Tuple[float, float, float, float]:

    bbox = (x1, y1, x2, y2)
    
    if not (0 <= x1 <= image_width and 0 <= x2 <= image_width and
            0 <= y1 <= image_height and 0 <= y2 <= image_height):
        raise ValueError(f"Coordinates {bbox} outside image bounds {image_width}x{image_height}")
    
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid bbox coordinates: {bbox}. x1 < x2 and y1 < y2 required")
    
    x1_norm = (x1 / image_width) * 998
    y1_norm = (y1 / image_height) * 998
    x2_norm = (x2 / image_width) * 998
    y2_norm = (y2 / image_height) * 998
    
    return round(x1_norm), round(y1_norm), round(x2_norm), round(y2_norm)

def denormalize_bbox_coordinates(bbox_norm: Union[Tuple[float, float, float, float], List[float]],
                                image_width: int,
                                image_height: int) -> Tuple[int, int, int, int]:
    
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
    
    x1 = int((x1_norm / 998) * image_width)
    y1 = int((y1_norm / 998) * image_height)
    x2 = int((x2_norm / 998) * image_width)
    y2 = int((y2_norm / 998) * image_height)
    
    return x1, y1, x2, y2
