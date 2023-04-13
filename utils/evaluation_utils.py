import numpy as np


def compute_overlap(box1, box2):
    """
    Compute the overlap between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    overlap = w * h
    return overlap

def compute_multi_overlap(tracks, detections):
    """
    Compute the overlap between multiple tracks and multiple detections.
    """
    num_tracks = len(tracks)
    num_detections = len(detections)
    overlaps = np.zeros((num_tracks, num_detections))
    for i, track in enumerate(tracks):
        for j, detection in enumerate(detections):
            overlaps[i, j] = compute_overlap(track, detection)
    return overlaps

def compute_assignment(overlaps, threshold=0.5):
    """
    Compute the optimal assignment between tracks and detections based on the overlap.
    """
    num_tracks, num_detections = overlaps.shape
    assignments = np.zeros(num_tracks, dtype=np.int32)
    used_detections = np.zeros(num_detections, dtype=np.bool)
    for i in range(num_tracks):
        max_overlap = -1
        max_index = -1
        for j in range(num_detections):
            if overlaps[i, j] > max_overlap and not used_detections[j]:
                max_overlap = overlaps[i, j]
                max_index = j
        if max_overlap > threshold:
            assignments[i] = max_index
            used_detections[max_index] = True
    return assignments

def compute_metrics(assignments, num_tracks, num_false_positives, num_misses):
    """
    Compute the precision, recall and F1 score based on the assignment and number of false positives and misses.
    """
    true_positives = len(assignments)
    false_positives = num_false_positives
    false_negatives = num_tracks - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score
