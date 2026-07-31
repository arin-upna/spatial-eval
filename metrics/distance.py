"""Spatial distance metrics for point-cloud semantic segmentation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


def get_nearest_distances(
    reference: np.ndarray,
    query: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the distance and index of the nearest reference point per query."""
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)

    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference must have shape (N, 3)")
    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError("query must have shape (M, 3)")
    if len(reference) == 0:
        raise ValueError("reference must contain at least one point")

    distances, neighbor_indices = cKDTree(reference).query(
        query,
        k=1,
        workers=-1,
    )
    return distances.ravel(), neighbor_indices.ravel()


def _evaluation_mask(
    size: int,
    valid_mask: np.ndarray,
    indices: Optional[np.ndarray],
) -> np.ndarray:
    """Combine the valid-class mask with an optional evaluation subset."""
    if indices is None:
        return valid_mask

    indices = np.asarray(indices)
    if indices.dtype == bool:
        if indices.shape != (size,):
            raise ValueError("A boolean error_idx must have shape (N,)")
        subset_mask = indices
    else:
        indices = indices.astype(np.int64, copy=False).ravel()
        if np.any(indices < 0) or np.any(indices >= size):
            raise IndexError("error_idx contains an out-of-range point index")
        subset_mask = np.zeros(size, dtype=bool)
        subset_mask[indices] = True

    return valid_mask & subset_mask


def get_distance_metrics(
    gt_classes: np.ndarray,
    pred_classes: np.ndarray,
    coords: np.ndarray,
    classification_dict: Dict[int, str],
    class_distance_limits: Dict[int, float],
    ignore_classes: Optional[List[int]] = None,
    tp_distance: bool = True,
    error_idx: Optional[np.ndarray] = None,
    include_fn: bool = False,
) -> Dict:
    """Compute the distance metrics introduced in the accompanying paper.

    For every evaluated point predicted as class ``c``, the raw distance is the
    Euclidean distance to the nearest ground-truth point of class ``c``. MDE is
    the mean of these distances after clipping them at the class threshold
    ``tau_c``. Correct predictions contribute a zero when ``tp_distance=True``.

    Distant errors are false positives whose raw distance is strictly greater
    than ``tau_c``. The mean non-distant error is computed only from
    misclassified points whose distance is at most ``tau_c``.

    Args:
        gt_classes: Ground-truth class IDs with shape ``(N,)``.
        pred_classes: Predicted class IDs with shape ``(N,)``.
        coords: XYZ coordinates in metric units with shape ``(N, 3)``.
        classification_dict: Mapping from evaluated class ID to class name.
        class_distance_limits: Per-class clipping thresholds in meters.
        ignore_classes: Classes omitted from macro/global distance summaries.
        tp_distance: Include correct predictions (zero distance) in MDE. This
            must be ``True`` to reproduce the paper definition.
        error_idx: Optional indices (or boolean mask) defining the evaluated
            subset. Reference neighbors are still searched in the full cloud.
        include_fn: Reserved for backward compatibility. The proposed metrics
            are defined by predicted class (false-positive perspective).

    Returns:
        A dictionary containing per-class MDE, mMDE, distant-error counts,
        non-distant error means, raw distances, and normalization counts.
    """
    if include_fn:
        raise NotImplementedError(
            "False-negative distance summaries are not part of the published "
            "metric. Use the predicted-class (false-positive) formulation."
        )

    gt_classes = np.asarray(gt_classes).ravel()
    pred_classes = np.asarray(pred_classes).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    if len(gt_classes) != len(pred_classes) or len(gt_classes) != len(coords):
        raise ValueError("gt_classes, pred_classes, and coords must have equal length")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if not classification_dict:
        raise ValueError("classification_dict cannot be empty")

    useful_classes = sorted(classification_dict)
    missing_thresholds = [
        class_id for class_id in useful_classes
        if class_id not in class_distance_limits
    ]
    if missing_thresholds:
        raise ValueError(
            f"Missing distance thresholds for class IDs: {missing_thresholds}"
        )
    invalid_thresholds = [
        class_id for class_id in useful_classes
        if class_distance_limits[class_id] <= 0
    ]
    if invalid_thresholds:
        raise ValueError(
            f"Distance thresholds must be positive for class IDs: {invalid_thresholds}"
        )

    valid_mask = np.isin(gt_classes, useful_classes)
    eval_mask = _evaluation_mask(len(gt_classes), valid_mask, error_idx)

    unexpected_predictions = np.unique(
        pred_classes[eval_mask & ~np.isin(pred_classes, useful_classes)]
    )
    if len(unexpected_predictions):
        raise ValueError(
            "Predictions contain class IDs absent from classification_dict: "
            f"{unexpected_predictions.tolist()}"
        )

    reference_gt = gt_classes[valid_mask]
    reference_coords = coords[valid_mask]
    eval_gt = gt_classes[eval_mask]
    eval_pred = pred_classes[eval_mask]
    eval_coords = coords[eval_mask]

    ignore_set = set(ignore_classes or [])
    mean_distances = {class_id: 0.0 for class_id in useful_classes}
    mean_non_distant = {class_id: 0.0 for class_id in useful_classes}
    distant_counts = {class_id: 0 for class_id in useful_classes}
    all_distances = {class_id: [] for class_id in useful_classes}
    error_distances = {class_id: [] for class_id in useful_classes}
    true_classes = {class_id: [] for class_id in useful_classes}
    gt_counts = {
        class_id: int(np.count_nonzero(eval_gt == class_id))
        for class_id in useful_classes
    }
    predicted_counts = {
        class_id: int(np.count_nonzero(eval_pred == class_id))
        for class_id in useful_classes
    }

    for class_id in useful_classes:
        threshold = float(class_distance_limits[class_id])
        reference_points = reference_coords[reference_gt == class_id]
        predicted_mask = eval_pred == class_id
        error_mask = predicted_mask & (eval_gt != class_id)
        true_classes[class_id] = eval_gt[error_mask].tolist()

        query_mask = predicted_mask if tp_distance else error_mask
        query_points = eval_coords[query_mask]
        wrong_points = eval_coords[error_mask]

        if len(query_points) == 0:
            continue

        if len(reference_points) == 0:
            # There is no spatial reference for this class in the current
            # scene. Treat every prediction as maximally clipped and every
            # false positive as distant.
            query_distances = np.full(len(query_points), threshold, dtype=float)
            raw_error_distances = np.full(len(wrong_points), threshold, dtype=float)
            class_distant_count = len(wrong_points)
            non_distant_errors = np.empty(0, dtype=float)
        else:
            query_distances, _ = get_nearest_distances(
                reference_points,
                query_points,
            )
            raw_error_distances, _ = get_nearest_distances(
                reference_points,
                wrong_points,
            ) if len(wrong_points) else (np.empty(0), np.empty(0, dtype=int))
            distant_mask = raw_error_distances > threshold
            class_distant_count = int(np.count_nonzero(distant_mask))
            non_distant_errors = raw_error_distances[~distant_mask]

        all_distances[class_id] = query_distances.tolist()
        error_distances[class_id] = raw_error_distances.tolist()
        distant_counts[class_id] = class_distant_count
        mean_distances[class_id] = float(
            np.mean(np.minimum(query_distances, threshold))
        )
        if len(non_distant_errors):
            mean_non_distant[class_id] = float(np.mean(non_distant_errors))

    included_classes = [
        class_id for class_id in useful_classes if class_id not in ignore_set
    ]
    mmde = (
        float(np.mean([mean_distances[class_id] for class_id in included_classes]))
        if included_classes
        else 0.0
    )
    total_distant = int(
        sum(distant_counts[class_id] for class_id in included_classes)
    )
    flattened_raw_distances = [
        distance
        for class_id in included_classes
        for distance in all_distances[class_id]
    ]
    micro_average = (
        float(np.mean(flattened_raw_distances))
        if flattened_raw_distances
        else 0.0
    )

    return {
        "dist_errors": all_distances,
        "error_distances": error_distances,
        "true_classes": true_classes,
        "gt_count_per_class": gt_counts,
        "predicted_count_per_class": predicted_counts,
        "mean_dist_errors": mean_distances,
        "mean_non_distant_errors": mean_non_distant,
        "mean_non_critic_errors": mean_non_distant,
        "distant_errors_per_class": distant_counts,
        "critic_points_per_class": distant_counts,
        "total_distant_errors": total_distant,
        "total_critic_points": total_distant,
        "mmde": mmde,
        "avg_error_distance": mmde,
        "micro_avg_distance": micro_average,
        "subset_point_count": int(np.count_nonzero(eval_mask)),
        "tp_distance": bool(tp_distance),
    }
