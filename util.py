import cv2
import pillow_heif
import numpy as np
from pathlib import Path
from PIL import Image


FACE_BORDER_COLOR = (0, 255, 255)  # (Blue, Green, Red)
FACE_BORDER_THICKNESS = 2
NAME_SIZE = 1.0
NAME_COLOR = (0, 255, 0)
NAME_THICKNESS = 2


def create_temp_names(num, prefix="face_"):
    names = []
    for i in range(1, num+1):
        temp_name = prefix + str(i)
        names.append(temp_name)
    return names


def draw_faces(img, faces, names=None):
    height, width = img.shape[:2]

    if names:
        assert len(faces) == len(names), "Number of faces and names must match"

    if names is None:
        names = [None] * len(faces)

    for face, name in zip(faces, names):
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      FACE_BORDER_COLOR, FACE_BORDER_THICKNESS)

        if name is None:
            continue

        text_x = max(x1 - 1, 0)
        text_y = max(y1 - 6, 0)
        cv2.putText(img, name, (text_x, text_y),
                    cv2.FONT_HERSHEY_COMPLEX,
                    NAME_SIZE, NAME_COLOR, NAME_THICKNESS)
    return img


def read_image(image_path: str | Path) -> np.ndarray:
    if Path(image_path).suffix in (".HEIC", ".heic"):
        pillow_heif.register_heif_opener()
        image = Image.open(image_path)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(image_path))
    return img
