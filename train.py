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

# Parse command line arguments
args = parser.parse_args()


# Define optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=args.lr)
loss_fn = keras.losses.BinaryCrossentropy()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_fn)

# Load dataset
train_data = load_dataset(args.data_dir)

# Preprocess the data
train_data = preprocess_data(train_data)

# Train the model
for epoch in range(args.num_epochs):
    print(f'Epoch {epoch + 1}/{args.num_epochs}')
    for batch_idx, (images, labels) in enumerate(train_data.batch(args.batch_size)):
        # Preprocess images
        images = np.stack([preprocess_image(image) for image in images])

        # Convert YOLO format labels to COCO format
        labels = np.array([yolo_to_coco(label) for label in labels])

        # Train on batch
        loss = model.train_on_batch(images, labels)

        # Print loss every 10 batches
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss:.4f}')

# Evaluate the model
evaluate_model(model, train_data)

# Save the model
os.makedirs(args.model_dir, exist_ok=True)
model.save(os.path.join(args.model_dir, 'model.h5'))
