import numpy as np

def evaluate_predictions(predictions, labels, num_classes):
    """
    Evaluate predictions against labels for multi-class classification.

    Args:
        predictions (np.ndarray): Predicted labels.
        labels (np.ndarray): True labels.
        num_classes (int): Number of classes.

    Returns:
        tuple: True positives, false positives, true negatives, false negatives, total samples.
    """
    valid_indices = np.where((labels >= 0) & (labels < num_classes))
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]

    true_positives = np.zeros((num_classes, 1))
    false_positives = np.zeros((num_classes, 1))
    true_negatives = np.zeros((num_classes, 1))
    false_negatives = np.zeros((num_classes, 1))

    for class_idx in range(num_classes):
        true_positives[class_idx] = np.sum(labels[np.where(predictions == class_idx)] == class_idx)
        false_positives[class_idx] = np.sum(labels[np.where(predictions == class_idx)] != class_idx)
        true_negatives[class_idx] = np.sum(labels[np.where(predictions != class_idx)] != class_idx)
        false_negatives[class_idx] = np.sum(labels[np.where(predictions != class_idx)] == class_idx)

    return true_positives, false_positives, true_negatives, false_negatives, len(labels)
