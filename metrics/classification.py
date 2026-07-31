"""
Classification metrics for LiDAR point cloud segmentation.

This module provides functions to compute standard classification metrics
such as confusion matrices, IoU, F1-score, precision and recall for evaluating
semantic segmentation models on point clouds.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .distance import get_nearest_distances

def get_confusion_matrix(
    pred_classes: np.ndarray, 
    gt_classes: np.ndarray, 
    classification_dict: Dict[int, str]
) -> np.ndarray:
    """
    Compute the confusion matrix between predicted and ground-truth class arrays,
    considering only the classes defined in the classification dictionary (useful classes).
    
    Args:
        pred_classes: Array with predicted classes.
        gt_classes: Array with ground-truth classes.
        classification_dict: Classification dictionary {class_id: class_name}.
        
    Returns:
        Confusion matrix for the useful classes as a NumPy integer array.
        
    Raises:
        ValueError: If pred_classes and gt_classes have different lengths.
    """
    # Check that prediction and ground truth have the same number of points
    if len(pred_classes) != len(gt_classes):
        raise ValueError("Las predicciones y el ground truth deben tener el mismo número de puntos")
    
    # Get sorted useful classes for searchsorted from the classification dictionary
    useful_classes = np.array(sorted(classification_dict.keys()))
    n_classes = len(useful_classes)
    
    # Create mask to consider only points whose ground truth belongs to useful classes
    valid_mask = np.isin(gt_classes, useful_classes) & np.isin(pred_classes, useful_classes)
    gt_classes_valid = gt_classes[valid_mask]
    pred_classes_valid = pred_classes[valid_mask]

    # Map ground-truth and predicted classes to indices [0, n_classes)
    # Assumes useful_classes is sorted (required for searchsorted)
    gt_indices = np.searchsorted(useful_classes, gt_classes_valid)
    pred_indices = np.searchsorted(useful_classes, pred_classes_valid)

    # 2D histogram to build the confusion matrix
    confusion_matrix, _, _ = np.histogram2d(
        gt_indices, pred_indices,
        bins=(n_classes, n_classes),
        range=[[0, n_classes], [0, n_classes]]
    )

    return confusion_matrix.astype(int)

def get_classification_metrics(
    confusion_matrix: np.ndarray, 
    classification_dict: Dict[int, str]
) -> Dict:
    """
    Compute classification metrics from a confusion matrix,
    considering the classes defined in the classification dictionary (useful classes).
    
    Args:
        confusion_matrix: Confusion matrix.
        classification_dict: Classification dictionary {class_id: class_name}.
        
    Returns:
        dict with the following metrics:
        - 'miou': Mean IoU over all classes
        - 'mean_f1': Mean F1-score over all classes
        - 'mean_recall': Mean recall over all classes
        - 'mean_precision': Mean precision over all classes
        - 'overall_accuracy': Overall accuracy
        - 'iou_per_class': Dictionary with IoU per class
        - 'f1_per_class': Dictionary with F1-score per class
        - 'recall_per_class': Dictionary with recall per class
        - 'precision_per_class': Dictionary with precision per class
    """
    useful_classes = sorted(classification_dict)
    
    # Calculate metrics per class
    iou_per_class = {}
    f1_per_class = {}
    recall_per_class = {}
    precision_per_class = {}

    for i, class_id in enumerate(useful_classes):
        tp = confusion_matrix[i, i] # True positives: diagonal of the confusion matrix
        fp = np.sum(confusion_matrix[:, i]) - tp # False positives: sum of column i (excluding tp)
        fn = np.sum(confusion_matrix[i, :]) - tp # False negatives: sum of row i (excluding tp)
        
        iou_per_class[class_id] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0 # IoU = TP / (TP + FP + FN)
        f1_per_class[class_id] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0 # F1-score = 2 * TP / (2 * TP + FP + FN)
        recall_per_class[class_id] = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall = TP / (TP + FN)
        precision_per_class[class_id] = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision = TP / (TP + FP)
    
    # Global metrics
    miou = np.mean(list(iou_per_class.values()))
    mean_f1 = np.mean(list(f1_per_class.values()))
    mean_recall = np.mean(list(recall_per_class.values()))
    mean_precision = np.mean(list(precision_per_class.values()))
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix) if np.sum(confusion_matrix) > 0 else 0.0
    
    return {
        'miou': miou,
        'mean_f1': mean_f1,
        'mean_recall': mean_recall,
        'mean_precision': mean_precision,
        'overall_accuracy': overall_accuracy,
        'iou_per_class': iou_per_class,
        'f1_per_class': f1_per_class,
        'recall_per_class': recall_per_class,
        'precision_per_class': precision_per_class,
    }

def interpolate_point_cloud(
    pred_point_cloud: Dict, 
    ground_truth_point_cloud: Dict, 
    precision_decimals: int = 2,
    force_interpolation: bool = False
) -> Dict:
    """
    Interpolate predicted classes onto the ground-truth point cloud
    when both clouds have different point sets (e.g., due to subsampling).
    Each point in the ground truth is assigned the class of its nearest neighbor
    in the predicted cloud.

    Args:
        pred_point_cloud: Predicted point cloud (must include 'coords' and 'classification').
        ground_truth_point_cloud: Ground-truth point cloud (must include 'coords' and 'classification').
        precision_decimals: Number of decimals used to compare coordinate equality (default: 2).
        force_interpolation: If True, always performs interpolation. Useful to enforce the same point order.

    Returns:
        A point cloud with the same points as the ground truth, but with interpolated predicted classes.
    """
    # Skip interpolation only when point order already matches. Equal unordered
    # point sets still require reindexing predictions onto the GT order.
    if not force_interpolation and pred_point_cloud['coords'].shape[0] == ground_truth_point_cloud['coords'].shape[0]:
        if np.allclose(
            pred_point_cloud['coords'],
            ground_truth_point_cloud['coords'],
            atol=10 ** (-precision_decimals),
            rtol=0,
        ):
            return pred_point_cloud
        
    # Find nearest neighbors
    _, neighbors_idx = get_nearest_distances(pred_point_cloud['coords'], ground_truth_point_cloud['coords'])
       
    # Assign predictions according to nearest neighbors and copy the remaining ground-truth attributes
    return {
        'classification': pred_point_cloud['classification'][neighbors_idx],
        **{k: v for k, v in ground_truth_point_cloud.items() if k != 'classification'}
    }
