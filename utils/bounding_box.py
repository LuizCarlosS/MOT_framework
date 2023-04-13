import cv2
import numpy as np


def draw_bounding_boxes(image, boxes, class_names, colors):
    for box in boxes:
        x, y, w, h = box[1:]
        left = int(x - w/2)
        top = int(y - h/2)
        right = int(x + w/2)
        bottom = int(y + h/2)
        class_id = int(box[0])
        class_name = class_names[class_id]
        color = colors[class_id]
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness=2)
        cv2.putText(image, class_name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color, thickness=1)

def draw_bounding_boxes_on_video(video_path, output_path, boxes, class_names, colors):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        draw_bounding_boxes(frame, boxes, class_names, colors)
        out.write(frame)
    cap.release()
    out.release()
