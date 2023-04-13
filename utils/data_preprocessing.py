import cv2
import numpy as np


def normalize_image(image):
    return image / 255.0

def random_horizontal_flip(image, boxes, probability=0.5):
    if np.random.uniform() < probability:
        image = cv2.flip(image, 1)
        width = image.shape[1]
        flipped_boxes = np.zeros_like(boxes)
        flipped_boxes[:, 0] = width - boxes[:, 2]
        flipped_boxes[:, 2] = width - boxes[:, 0]
        flipped_boxes[:, 1:] = boxes[:, 1:]
        return image, flipped_boxes
    else:
        return image, boxes

def random_crop_image(image, boxes, min_scale=0.5, max_scale=1.0):
    height, width, _ = image.shape
    min_dim = min(height, width)
    scale = np.random.uniform(min_scale, max_scale)
    new_dim = int(min_dim * scale)

    x = np.random.randint(0, width - new_dim)
    y = np.random.randint(0, height - new_dim)

    cropped_image = image[y:y+new_dim, x:x+new_dim, :]
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] -= x
    cropped_boxes[:, [1, 3]] -= y

    return cropped_image, cropped_boxes
