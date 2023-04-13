import cv2


def draw_bounding_box(frame, bbox, color=(0, 255, 0)):
    """
    Draw a bounding box on the given frame.
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def draw_id(frame, bbox, id_, color=(0, 255, 0)):
    """
    Draw an ID on the given frame.
    """
    x, y, w, h = bbox
    cv2.putText(frame, str(id_), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_track(frame, bbox_list, id_, color=(0, 255, 0)):
    """
    Draw a track on the given frame.
    """
    for bbox in bbox_list:
        draw_bounding_box(frame, bbox, color=color)
    draw_id(frame, bbox_list[0], id_, color=color)
