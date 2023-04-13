import argparse
import os

import numpy as np

from utils.data_conversion import yolo_to_coco
from utils.data_preprocessing import preprocess_data, preprocess_image
from utils.evaluation_utils import evaluate_model
from utils.visualization_utils import draw_track

# Define command line arguments
parser = argparse.ArgumentParser(description='Train a multiple object tracking model')
parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
parser.add_argument('--model-dir', type=str, required=True, help='Path to directory to save trained model')
parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--matching-algo', type=str, default='greedy', help='Choose the algorithm that will match the detections to previous tracks.')

# Parse command line arguments
args = parser.parse_args()

