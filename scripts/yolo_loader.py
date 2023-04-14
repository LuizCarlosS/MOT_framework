import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn

from ..tracker.tracking_history import HistoryKeeper


class MOTDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, n_frames, transform=None):
        self.data_path = data_path
        self.n_frames = n_frames
        self.transform = transform

        self.file_list = []
        for seq_folder in os.listdir(data_path):
            seq_path = os.path.join(data_path, seq_folder)
            gt_path = os.path.join(seq_path, 'gt', 'gt.txt')
            gt_df = pd.read_csv(gt_path, names=['frame', 'id', 'xmin', 'ymin', 'xmax', 'ymax', 'drop1', 'drop2', 'drop3'])
            gt_df['xmax'] += gt_df['xmin']
            gt_df['ymax'] += gt_df['ymin']

            if not os.path.isdir(seq_path):
                continue
            seq_files = sorted(os.listdir(os.path.join(seq_path, 'img1')))
            for i in range(n_frames, len(seq_files)):
                data = {'seq_path': seq_path, 'frame_idx': i, 'detections': []}
                for j in range(i-n_frames, i):
                    frame_path = os.path.join(seq_path, 'img1' seq_files[j])
                    data['detections'].append(load_detections(gt_df, j))
                data['detections'].append(load_detections(gt_df, i))
                self.file_list.append(data)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.file_list[idx]
        history = []
        for i in range(self.n_frames):
            detections = data['detections'][i]
            image = load_image(os.path.join(data['seq_path'], f'{i+1:06d}.jpg'))
            if self.transform:
                image = self.transform(image)
            history.append({'image': image, 'detections': detections})
        
        current_detections = data['detections'][-1]
        current_image = load_image(os.path.join(data['seq_path'], f'{data["frame_idx"]+1:06d}.jpg'))
        if self.transform:
            current_image = self.transform(current_image)

        tracker = HistoryKeeper()
        for detection in current_detections:
            tracker.update()

        return {'history': history, 'current_image': current_image, 'current_detections': current_detections, 'tracker': tracker}

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_detections(gt_df, frame):
    # load detections from file
    sel_df = gt_df[gt_df['frame'] == frame]
    detections = []
    for idx, row in sel_df.iterrows():
        det = {}
        det['xmin'] = row['xmin']
        det['ymin'] = row['ymin']
        det['xmax'] = row['xmax']
        det['ymax'] = row['ymax']
        det['ID'] = row['id']
        detections.append(det)
    # return as a list of dictionaries containing xmin, ymin, xmax, ymax, and ID
    
    
