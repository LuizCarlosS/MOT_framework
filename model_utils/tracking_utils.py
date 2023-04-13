import cv2
import numpy as np


def create_mask(bounding_boxes, image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8) # initialize mask with zeros
    for bbox in bounding_boxes:
        # extract the coordinates of the bounding box
        xmin, ymin, xmax, ymax = bbox
        
        # create a binary mask for the bounding box
        bbox_mask = np.zeros_like(mask)
        bbox_mask[ymin:ymax, xmin:xmax] = 255
        
        # combine the bounding box mask with the overall mask
        mask = cv2.bitwise_or(mask, bbox_mask)
    
    return mask