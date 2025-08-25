from typing import Tuple, List, Union
import PIL.Image
import PIL.ExifTags
from PIL import Image, ImageOps
import io
from PyQt6.QtGui import QImage

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

def fix_image_orientation(image: Image.Image) -> Image.Image:
    try:
        exif = image._getexif()
        if exif is None:
            return image.convert('RGB')
        
        exif_dict = {
            PIL.ExifTags.TAGS[k]: v 
            for k, v in exif.items() 
            if k in PIL.ExifTags.TAGS
        }
        
        orientation = exif_dict.get('Orientation')
        if orientation is None:
            return image.convert('RGB')
        
        if orientation == 2:
            image = ImageOps.mirror(image)
        elif orientation == 3:
            image = image.rotate(180, expand=True)
        elif orientation == 4:
            image = ImageOps.flip(image)
        elif orientation == 5:
            image = ImageOps.mirror(image.rotate(-90, expand=True))
        elif orientation == 6:
            image = image.rotate(-90, expand=True)
        elif orientation == 7:
            image = ImageOps.mirror(image.rotate(90, expand=True))
        elif orientation == 8:
            image = image.rotate(90, expand=True)
            
    except (AttributeError, KeyError, IndexError, Exception):
        pass
    
    return image.convert('RGB')

def fix_image_colorspace(image: Image.Image) -> Image.Image:

    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background

    return image

def ensure_proper_image(image: Image.Image) -> Image.Image:

    image = fix_image_orientation(image)
    
    image = fix_image_colorspace(image)
    
    return image

def image_to_qimage(pil_image: Image.Image) -> QImage:
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    
    qimage = QImage()
    qimage.loadFromData(buffer.getvalue())
    
    return qimage