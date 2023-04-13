import cv2
import numpy as np

def draw_bounding_box(image, box, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box on the given image.
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def draw_track(image, track, color=(0, 255, 0), thickness=2):
    """
    Draw a track on the given image.
    """
    for i in range(len(track) - 1):
        pt1 = tuple(map(int, track[i]))
        pt2 = tuple(map(int, track[i + 1]))
        cv2.line(image, pt1, pt2, color, thickness)

def read_image(path):
    """
    Read an image from the given path.
    """
    return cv2.imread(path)

def resize_image(image, size):
    """
    Resize the given image to the specified size.
    """
    return cv2.resize(image, size)

def crop_image(image, box):
    """
    Crop the given image using the given bounding box.
    """
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]

def show_image(image, title='Image'):
    """
    Show the given image in a window with the given title.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
