"""
Metric functions for evaluating LiDAR point cloud segmentation.

This package provides a complete set of utilities to evaluate semantic segmentation models,
including classification metrics, spatial distance metrics, and visualization/reporting helpers.
"""

# Main classification imports
from .classification import (
    get_confusion_matrix,
    get_classification_metrics,
    interpolate_point_cloud
)

# Distance metric imports
from .distance import (
    get_nearest_distances,
    get_distance_metrics
)

# Main evaluation imports
from .evaluation import (
    find_matching_files,
    evaluate_single_point_cloud,
    aggregate_metrics,
    evaluate_point_clouds,
    evaluate_union_errors
)

# Visualization/reporting imports
from .display import (
    create_iou_table,
    create_f1_table,
    create_precision_table,
    create_recall_table,
    create_distance_table,
    create_error_contribution_table,
    plot_confusion_matrix,
    plot_distance_histogram,
    display_evaluation_results,
    save_confusion_matrix_csv
)

# Public API
__all__ = [
    # Classification
    'get_confusion_matrix',
    'get_classification_metrics',
    'interpolate_point_cloud',

    # Distance
    'get_nearest_distances',
    'get_distance_metrics',

    # Evaluation
    'find_matching_files',
    'evaluate_single_point_cloud',
    'aggregate_metrics',
    'evaluate_point_clouds',
    'evaluate_union_errors',

    # Visualization
    'create_iou_table',
    'create_f1_table',
    'create_precision_table',
    'create_recall_table',
    'create_distance_table',
    'create_error_contribution_table',
    'plot_confusion_matrix',
    'plot_distance_histogram',
    'display_evaluation_results',
    'save_confusion_matrix_csv'
]
