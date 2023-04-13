import os

import cv2
import numpy as np


def convert_image_data_to_yolo_format(data_dir, class_names_file):
    class_names = []
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.txt'):
            image_path = os.path.join(data_dir, file_name)
            txt_path = os.path.join(data_dir, os.path.splitext(file_name)[0] + '.txt')
            if not os.path.exists(txt_path):
                with open(txt_path, 'w') as f:
                    f.write('')

            image = cv2.imread(image_path)
            height, width, _ = image.shape

            with open(txt_path, 'r') as f:
                boxes = [line.strip().split() for line in f.readlines()]

            with open(txt_path, 'w') as f:
                for box in boxes:
                    class_name, x_min, y_min, x_max, y_max = box
                    x_center = (float(x_min) + float(x_max)) / 2 / width
                    y_center = (float(y_min) + float(y_max)) / 2 / height
                    w = (float(x_max) - float(x_min)) / width
                    h = (float(y_max) - float(y_min)) / height
                    class_id = class_names.index(class_name)
                    f.write(f'{class_id} {x_center} {y_center} {w} {h}\n')

def convert_sequence_data_to_yolo_format(data_dir, class_names_file):
    class_names = []
    with open(class_names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    for sequence_name in os.listdir(data_dir):
        sequence_dir = os.path.join(data_dir, sequence_name)
        for image_name in os.listdir(sequence_dir):
            if not image_name.endswith('.txt'):
                image_path = os.path.join(sequence_dir, image_name)
                txt_path = os.path.join(sequence_dir, os.path.splitext(image_name)[0] + '.txt')
                if not os.path.exists(txt_path):
                    with open(txt_path, 'w') as f:
                        f.write('')

                image = cv2.imread(image_path)
                height, width, _ = image.shape

                with open(txt_path, 'r') as f:
                    boxes = [line.strip().split() for line in f.readlines()]

                with open(txt_path, 'w') as f:
                    for box in boxes:
                        class_name, x_min, y_min, x_max, y_max, detection_id = box
                        x_center = (float(x_min) + float(x_max)) / 2 / width
                        y_center = (float(y_min) + float(y_max)) / 2 / height
                        w = (float(x_max) - float(x_min)) / width
                        h = (float(y_max) - float(y_min)) / height
                        class_id = class_names.index(class_name)
                        f.write(f'{class_id} {x_center} {y_center} {w} {h} {detection_id}\n')
