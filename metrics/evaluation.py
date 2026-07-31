"""
Evaluation and metric aggregation functions for LiDAR point cloud segmentation.

This module contains the main functions to evaluate segmentation models,
including per-file evaluation, global metric aggregation, and comparison
between multiple models.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

# Relative imports from the same package
from .classification import get_confusion_matrix, get_classification_metrics, interpolate_point_cloud
from .distance import get_distance_metrics
from .io import load_point_cloud


def apply_class_grouping(
    classes: np.ndarray, 
    class_grouping: Dict[int, List[int]]
) -> np.ndarray:
    """
    Apply class grouping according to the provided configuration.
    
    Args:
        classes: Array with original classes.
        class_grouping: Dict {target_class: [list_of_source_classes]}.
        
    Returns:
        Array with grouped classes applied.
        
    Example:
        class_grouping = {3: [3, 4, 5]}  # All vegetation classes -> Vegetation (3)
        classes = [2, 3, 4, 5, 6]  -> [2, 3, 3, 3, 6]
    """
    if not class_grouping:
        return classes.copy()
    
    grouped_classes = classes.copy()
    
    for target_class, source_classes in class_grouping.items():
        for source_class in source_classes:
            grouped_classes[classes == source_class] = target_class
            
    return grouped_classes


def find_matching_files(
    pred_dir: str, 
    gt_dir: str, 
    file_extension: str = ".laz", 
    suffix: Optional[str] = None, 
    prefix: Optional[str] = None,
    match_by_name_only: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Match prediction files with their corresponding ground-truth files
    based on filename matches (with optional suffix/prefix removal).
    
    Args:
        pred_dir: Directory containing prediction files.
        gt_dir: Directory containing ground-truth files.
        file_extension: Extension of files to consider (default: ".laz").
        suffix: Suffix to remove from prediction filenames (e.g., "_pred").
        prefix: Prefix to remove from prediction filenames (e.g., "pred_").
        match_by_name_only: If True, match only by base filename without considering
                            the extension. Useful when files have different extensions
                            (e.g., .las vs .laz). Default: False.
    
    Returns:
        Tuple containing:
        - matched_gt_files: Paths to ground-truth files.
        - matched_pred_files: Paths to prediction files.
        
    Raises:
        Exception: If directories do not exist or no files are found.
    """
    # Verify that directories exist
    if not os.path.exists(pred_dir):
        raise Exception(f"El directorio de predicciones no existe: {pred_dir}")
    if not os.path.exists(gt_dir):
        raise Exception(f"El directorio de ground truth no existe: {gt_dir}")
    
    # Find prediction files
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith(file_extension)]
    
    if len(pred_files) == 0:
        raise Exception(f"No se encontraron archivos con extensión {file_extension} en {pred_dir}")
       
    # Create lists to keep matched paths
    matched_gt_files = []
    matched_pred_files = []
    
    # If match_by_name_only is enabled, build a lookup for GT files
    if match_by_name_only:
        gt_files_dict = {}
        for f in os.listdir(gt_dir):
            if os.path.isfile(os.path.join(gt_dir, f)):
                # Create dict with base name (without extension) as key
                base_name_no_ext = os.path.splitext(f)[0]
                if base_name_no_ext not in gt_files_dict:
                    gt_files_dict[base_name_no_ext] = []
                gt_files_dict[base_name_no_ext].append(f)
    
    # For each prediction file, find its corresponding ground truth
    for pred_file in pred_files:
        # Get base name of prediction file, removing suffix/prefix if necessary
        base_name = pred_file
        
        if suffix and suffix in base_name:
            base_name = base_name.split(suffix)[0] + file_extension
        
        if prefix and base_name.startswith(prefix):
            base_name = base_name[len(prefix):]
        
        # If match_by_name_only is enabled, match by base name without extension
        if match_by_name_only:
            # Get base name without extension
            base_name_no_ext = os.path.splitext(base_name)[0]
            
            # Look for GT file with the same base name (any extension)
            if base_name_no_ext in gt_files_dict:
                # Take the first file found with that base name
                gt_file_name = gt_files_dict[base_name_no_ext][0]
                gt_file_path = os.path.join(gt_dir, gt_file_name)
                matched_gt_files.append(gt_file_path)
                matched_pred_files.append(os.path.join(pred_dir, pred_file))
        else:
            # Original behavior: look for file with same name and extension
            gt_file_path = os.path.join(gt_dir, base_name)
            
            if os.path.exists(gt_file_path):
                matched_gt_files.append(gt_file_path)
                matched_pred_files.append(os.path.join(pred_dir, pred_file))

    return matched_gt_files, matched_pred_files


def evaluate_single_point_cloud(
    gt_dir: str,
    pred_dir: Optional[str],
    classification_dict: Dict[int, str],
    class_distance_limits: Dict[int, float],
    ignore_classes: Optional[List[int]] = None,
    class_grouping: Optional[Dict[int, List[int]]] = None,
    tp_distance: bool = True,
    error_idx: Optional[np.ndarray] = None,
    include_fn: bool = False,
    pred_field: str = "Prediction",
) -> Dict:
    """
    Compute classification and distance metrics for a single point cloud.

    Args:
        gt_dir: Path to the ground-truth file, or single file with both GT and predictions.
        pred_dir: Path to the prediction file. If None, gt_dir contains both
                  'classification' (GT) and pred_field (predictions) in the same file.
        classification_dict: Classification dictionary {class_id: class_name}.
        class_distance_limits: Per-class critical distance thresholds {class_id: distance}.
        ignore_classes: Class IDs to ignore in aggregated metrics.
        class_grouping: Class grouping {target_class: [list_of_source_classes]}.
        tp_distance: Whether to include distances for true positives.
        error_idx: Specific indices to evaluate (None evaluates all).
        include_fn: Whether to include false negative metrics.
        pred_field: Name of the prediction field when using single-file mode (default: "Prediction").

    Returns:
        Dict with classification and distance metrics.
    """
    if pred_dir is None:
        # Single-file mode: GT and predictions in the same file
        pc = load_point_cloud(gt_dir, features=["coords", "classification", pred_field])
        gt_point_cloud = {"coords": pc["coords"], "classification": pc["classification"]}
        pred_point_cloud = {"coords": pc["coords"], "classification": pc[pred_field]}
    else:
        gt_point_cloud = load_point_cloud(gt_dir, features=["coords", "classification"])
        pred_point_cloud = load_point_cloud(pred_dir, features=["coords", "classification"])

    # Align predictions to the ground-truth point order when needed.
    pred_point_cloud = interpolate_point_cloud(
        pred_point_cloud,
        gt_point_cloud,
        force_interpolation=False,
    )
    
    # Apply class grouping if configured
    if class_grouping:
        gt_point_cloud['classification'] = apply_class_grouping(
            gt_point_cloud['classification'], 
            class_grouping
        )
        pred_point_cloud['classification'] = apply_class_grouping(
            pred_point_cloud['classification'], 
            class_grouping
        )
    
    # Compute confusion matrix
    file_cm = get_confusion_matrix(
        pred_point_cloud['classification'], 
        gt_point_cloud['classification'], 
        classification_dict
    )
    
    # Compute classification metrics
    file_metrics = get_classification_metrics(file_cm, classification_dict)
    
    # Compute distance metrics
    file_dist_metrics = get_distance_metrics(
        gt_point_cloud['classification'],
        pred_point_cloud['classification'],
        gt_point_cloud['coords'],
        classification_dict,
        class_distance_limits,
        ignore_classes,
        tp_distance,
        error_idx,
        include_fn,
    )
    
    # Combine all metrics
    file_metrics.update(file_dist_metrics)

    # Store confusion matrix
    file_metrics['confusion_matrix'] = file_cm

    # Store class names
    file_metrics['class_names'] = {class_id: name for class_id, name in classification_dict.items()}
    
    return file_metrics


def aggregate_metrics(
    individual_metrics: Dict, 
    classification_dict: Dict[int, str],
    class_distance_limits: Dict[int, float],
    ignore_classes: Optional[List[int]] = None,
    class_grouping: Optional[Dict[int, List[int]]] = None,
    include_fn: bool = False,
) -> Dict:
    """Aggregate per-file results without averaging already-averaged values."""
    if include_fn:
        raise NotImplementedError(
            "False-negative distance summaries are not part of the published metric."
        )

    useful_classes = sorted(classification_dict)
    ignore_set = set(ignore_classes or [])
    global_cm = np.zeros((len(useful_classes), len(useful_classes)), dtype=int)
    global_distances = {class_id: [] for class_id in useful_classes}
    global_error_distances = {class_id: [] for class_id in useful_classes}
    global_true_classes = {class_id: [] for class_id in useful_classes}
    global_distant_counts = {class_id: 0 for class_id in useful_classes}
    global_gt_counts = {class_id: 0 for class_id in useful_classes}
    global_predicted_counts = {class_id: 0 for class_id in useful_classes}
    subset_point_count = 0
    tp_distance_values = set()

    for filename, file_metrics in individual_metrics.items():
        if "confusion_matrix" not in file_metrics:
            raise KeyError(f"Missing confusion_matrix for '{filename}'")
        global_cm += file_metrics["confusion_matrix"]
        subset_point_count += int(
            file_metrics.get(
                "subset_point_count",
                np.sum(file_metrics["confusion_matrix"]),
            )
        )
        tp_distance_values.add(bool(file_metrics.get("tp_distance", False)))

        for class_id in useful_classes:
            global_distances[class_id].extend(
                file_metrics.get("dist_errors", {}).get(class_id, [])
            )
            global_error_distances[class_id].extend(
                file_metrics.get("error_distances", {}).get(class_id, [])
            )
            global_true_classes[class_id].extend(
                file_metrics.get("true_classes", {}).get(class_id, [])
            )
            global_distant_counts[class_id] += int(
                file_metrics.get(
                    "distant_errors_per_class",
                    file_metrics.get("critic_points_per_class", {}),
                ).get(class_id, 0)
            )
            global_gt_counts[class_id] += int(
                file_metrics.get("gt_count_per_class", {}).get(class_id, 0)
            )
            global_predicted_counts[class_id] += int(
                file_metrics.get("predicted_count_per_class", {}).get(class_id, 0)
            )

    if len(tp_distance_values) > 1:
        raise ValueError("Cannot aggregate files computed with different tp_distance values")
    tp_distance = tp_distance_values.pop() if tp_distance_values else False

    mean_distances = {}
    mean_non_distant = {}
    for class_id in useful_classes:
        threshold = float(class_distance_limits[class_id])
        distances = np.asarray(global_distances[class_id], dtype=float)
        errors = np.asarray(global_error_distances[class_id], dtype=float)
        mean_distances[class_id] = (
            float(np.mean(np.minimum(distances, threshold)))
            if len(distances)
            else 0.0
        )
        non_distant_errors = errors[errors <= threshold]
        mean_non_distant[class_id] = (
            float(np.mean(non_distant_errors))
            if len(non_distant_errors)
            else 0.0
        )

    included_classes = [
        class_id for class_id in useful_classes if class_id not in ignore_set
    ]
    mmde = (
        float(np.mean([mean_distances[class_id] for class_id in included_classes]))
        if included_classes
        else 0.0
    )
    total_distant = int(
        sum(global_distant_counts[class_id] for class_id in included_classes)
    )
    raw_distances = [
        distance
        for class_id in included_classes
        for distance in global_distances[class_id]
    ]
    micro_average = float(np.mean(raw_distances)) if raw_distances else 0.0

    global_metrics = get_classification_metrics(global_cm, classification_dict)
    global_metrics.update({
        "confusion_matrix": global_cm,
        "class_names": {
            class_id: classification_dict[class_id]
            for class_id in useful_classes
        },
        "dist_errors": global_distances,
        "error_distances": global_error_distances,
        "true_classes": global_true_classes,
        "gt_count_per_class": global_gt_counts,
        "predicted_count_per_class": global_predicted_counts,
        "mean_dist_errors": mean_distances,
        "mean_non_distant_errors": mean_non_distant,
        "mean_non_critic_errors": mean_non_distant,
        "distant_errors_per_class": global_distant_counts,
        "critic_points_per_class": global_distant_counts,
        "total_distant_errors": total_distant,
        "total_critic_points": total_distant,
        "mmde": mmde,
        "avg_error_distance": mmde,
        "micro_avg_distance": micro_average,
        "subset_point_count": subset_point_count,
        "tp_distance": tp_distance,
    })
    return global_metrics


def evaluate_point_clouds(
    gt_dir: str,
    pred_dir: Optional[str] = None,
    classification_dict: Dict[int, str] = None,
    file_extension: str = ".laz",
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    class_distance_limits: Dict[int, float] = None,
    ignore_classes: Optional[List[int]] = None,
    class_grouping: Optional[Dict[int, List[int]]] = None,
    tp_distance: bool = False,
    error_idx: Optional[np.ndarray] = None,
    include_fn: bool = False,
    match_by_name_only: bool = True,
    pred_field: str = "Prediction",
    max_workers: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """
    Evaluate predicted point clouds against ground truth and compute metrics.

    Supports two modes:
    - Two-directory mode (pred_dir given): matches files between gt_dir and pred_dir,
      each file has 'classification' as its respective label.
    - Single-directory mode (pred_dir=None): each file in gt_dir contains both
      'classification' (GT) and pred_field (predictions).

    Args:
        gt_dir: Directory containing ground-truth files (or files with both GT and predictions).
        pred_dir: Directory containing prediction files. If None, single-directory mode.
        classification_dict: Classification dictionary {class_id: class_name}.
        file_extension: Extension of files to process (default: ".laz").
        suffix: Suffix to remove from prediction filenames (e.g., "_pred").
        prefix: Prefix to remove from prediction filenames (e.g., "pred_").
        class_distance_limits: Dict {class_id: distance} with per-class critical thresholds.
        ignore_classes: List of class IDs to ignore in distance aggregates.
        class_grouping: Class grouping {target_class: [list_of_source_classes]}.
        tp_distance: Whether to include distances for true positives.
        error_idx: Specific indices to evaluate (None evaluates all).
        include_fn: Whether to include false negative metrics (default: False).
        match_by_name_only: If True, match only by base filename (two-dir mode only).
        pred_field: Name of the prediction field in single-directory mode (default: "Prediction").
        max_workers: Maximum number of parallel file workers. If None, uses
            min(32, os.cpu_count()) to avoid excessive memory pressure.

    Returns:
        Tuple containing:
        - global_metrics: Global metrics.
        - individual_metrics: Per-file metrics.
    """
    if pred_dir is None:
        # Single-directory mode: each file has both GT and predictions
        files = sorted([f for f in os.listdir(gt_dir) if f.endswith(file_extension)])
        if not files:
            print(f"No files with extension {file_extension} found in {gt_dir}")
            return None, None
        file_paths = [os.path.join(gt_dir, f) for f in files]
    else:
        # Two-directory mode: match files between gt_dir and pred_dir
        matched_gt_files, matched_pred_files = find_matching_files(
            pred_dir=pred_dir,
            gt_dir=gt_dir,
            file_extension=file_extension,
            suffix=suffix,
            prefix=prefix,
            match_by_name_only=match_by_name_only,
        )
        if len(matched_gt_files) == 0:
            print("No matching files found. Please check paths and filenames.")
            return None, None

    # Dictionary to store metrics for each file
    individual_metrics = {}

    # Parallel processing. For very large point clouds, using all CPUs can create
    # excessive memory pressure because every worker loads full LAZ arrays.
    default_workers = min(32, os.cpu_count() or 1)
    worker_count = max(1, int(max_workers or default_workers))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        if pred_dir is None:
            futures = {
                executor.submit(
                    evaluate_single_point_cloud,
                    fp, None,
                    classification_dict, class_distance_limits,
                    ignore_classes, class_grouping,
                    tp_distance, error_idx, include_fn, pred_field,
                ): os.path.basename(fp)
                for fp in file_paths
            }
        else:
            futures = {
                executor.submit(
                    evaluate_single_point_cloud,
                    gt_file, pred_file,
                    classification_dict, class_distance_limits,
                    ignore_classes, class_grouping,
                    tp_distance, error_idx, include_fn,
                ): os.path.basename(gt_file)
                for gt_file, pred_file in zip(matched_gt_files, matched_pred_files)
            }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating predictions", unit="sample"):
            filename = futures[future]
            try:
                metrics = future.result()
                individual_metrics[filename] = metrics
            except Exception as e:
                print(f"Error while processing {filename}: {e}") 

    # Aggregate per-file metrics into global metrics
    global_metrics = aggregate_metrics(
        individual_metrics, 
        classification_dict,
        class_distance_limits,
        ignore_classes,
        class_grouping,
        include_fn,
    )
    
    return global_metrics, individual_metrics

def evaluate_union_errors(
    gt_dir: str,
    pred_dirs: List[str],
    classification_dict: Dict[int, str], 
    file_extension: str = ".laz",
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    class_distance_limits: Dict[int, float] = None,
    ignore_classes: Optional[List[int]] = None,
    class_grouping: Optional[Dict[int, List[int]]] = None,
    include_fn: bool = False,
    match_by_name_only: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate errors of multiple models using the union of errors as the evaluation set.
    
    Args:
        gt_dir: Directory with ground-truth files.
        pred_dirs: List of directories with prediction files.
        classification_dict: Classification dictionary.
        file_extension: Extension of files to process (default: ".laz").
        suffix: Suffix to remove from prediction filenames.
        prefix: Prefix to remove from prediction filenames.
        class_distance_limits: Per-class critical distance thresholds.
        ignore_classes: List of class IDs to ignore in distance aggregates.
        class_grouping: Class grouping {target_class: [source classes]}.
        include_fn: Whether to include false negative metrics.
        match_by_name_only: If True, match only by base filename without considering
                            the extension (useful when files have different extensions,
                            e.g., .las vs .laz). Default: False.
        
    Returns:
        union_global_metrics: Global metrics per model on the union evaluation set.
        individual_metrics: Per-file and per-model metrics.
        union_summary: Comparative statistics of unique errors.
    """
    
    # 1. Group paths by ground-truth file
    gt_to_preds = defaultdict(dict)
    model_names = []
    for pred_dir in pred_dirs:
        model_name = os.path.basename(pred_dir)
        model_names.append(model_name)
        gt_files, pred_files = find_matching_files(pred_dir, gt_dir, file_extension, suffix, prefix, match_by_name_only)
        for gt_file, pred_file in zip(gt_files, pred_files):
            gt_to_preds[gt_file][model_name] = pred_file

    # 2. Evaluate each point cloud on its union error set
    individual_metrics = {model: {} for model in model_names}
    error_contribution_stats = {model: {'total_errors': 0, 'unique_errors': 0, 'shared_errors': 0} for model in model_names}
    total_union_errors = 0
    
    for gt_path, preds_map in tqdm(gt_to_preds.items(), desc="Evaluando predicciones", unit="muestra"):
        # 2.1. Load ground-truth point cloud
        gt_pc = load_point_cloud(gt_path, features=["coords","classification"])
        coords_gt = gt_pc['coords']
        class_gt = apply_class_grouping(
            gt_pc['classification'],
            class_grouping,
        )
        gt_pc['classification'] = class_gt

        # 2.2. Interpolate and collect error indices (UNION)
        useful_classes = list(classification_dict.keys())
        valid_mask = np.isin(class_gt, useful_classes)
        
        union_error_idx = set()
        interpolated_pcs = {}
        model_error_idx = {}
        
        for model in model_names:
            pred_pc = load_point_cloud(preds_map[model], features=["coords","classification"])
            interpolated_pc = interpolate_point_cloud(pred_pc, gt_pc, force_interpolation=True)
            interpolated_pc['classification'] = apply_class_grouping(
                interpolated_pc['classification'],
                class_grouping,
            )
            interpolated_pcs[model] = interpolated_pc
            
            # Consider only errors on points with useful classes
            errs = np.where((interpolated_pc['classification'] != class_gt) & valid_mask)[0]
            model_error_idx[model] = set(errs)
            union_error_idx |= set(errs)
        
        # Update error statistics
        total_union_errors += len(union_error_idx)
        
        if union_error_idx:
            for model in model_names:
                model_errors = model_error_idx[model]
                error_contribution_stats[model]['total_errors'] += len(model_errors)
                
                # Compute unique errors
                other_models_errors = set()
                for other_model in model_names:
                    if other_model != model:
                        other_models_errors |= model_error_idx[other_model]
                
                unique_errors = model_errors - other_models_errors
                shared_errors = model_errors & other_models_errors
                
                error_contribution_stats[model]['unique_errors'] += len(unique_errors)
                error_contribution_stats[model]['shared_errors'] += len(shared_errors)

        # Convert union to index array
        union_idx = np.array(sorted(union_error_idx), dtype=int) if union_error_idx else np.array([], dtype=int)
        
        # 2.3. Métricas individuales por modelo usando la UNIÓN como error_idx
        # 2.3. Per-model metrics using the UNION as error_idx
        for model in model_names:
            pred_classification = interpolated_pcs[model]['classification']
            
            # Compute distance metrics using:
            # - tp_distance=True: includes both TP and FP in the evaluation
            # - error_idx=union_idx: evaluates only on the union error set
            distance_metrics = get_distance_metrics(
                gt_classes=class_gt,                    # GT of the full cloud
                pred_classes=pred_classification,       # Prediction of the full cloud
                coords=coords_gt,                       # Coordinates of the full cloud
                classification_dict=classification_dict,
                class_distance_limits=class_distance_limits,
                ignore_classes=ignore_classes,
                tp_distance=True,                       # Include TP (distance 0) and FP
                error_idx=union_idx,                    # Evaluate only on union of errors
                include_fn=include_fn
            )

            confusion_matrix = get_confusion_matrix(pred_classification[union_idx], class_gt[union_idx], classification_dict)
            classification_metrics = get_classification_metrics(confusion_matrix, classification_dict)

            individual_metrics[model][os.path.basename(gt_path)] = {
                **classification_metrics,
                **distance_metrics,
                'confusion_matrix': confusion_matrix,
                'class_names': classification_dict,
                'subset_point_count': len(union_idx)
            }

    # 3. Aggregate global metrics (sum over all samples)
    union_global_metrics = {}
    for model in model_names:
        global_metrics = aggregate_metrics(
            individual_metrics=individual_metrics[model],
            classification_dict=classification_dict,
            class_distance_limits=class_distance_limits,
            ignore_classes=ignore_classes,
            class_grouping=class_grouping,
            include_fn=include_fn,
        )
        
        # Add error contribution statistics
        global_metrics['error_contribution'] = error_contribution_stats[model].copy()
        
        # Compute percentages
        total_errors = error_contribution_stats[model]['total_errors']
        if total_errors > 0:
            global_metrics['error_contribution']['unique_error_percentage'] = (error_contribution_stats[model]['unique_errors'] / total_errors * 100)
            global_metrics['error_contribution']['shared_error_percentage'] = (error_contribution_stats[model]['shared_errors'] / total_errors * 100)
        else:
            global_metrics['error_contribution']['unique_error_percentage'] = 0.0
            global_metrics['error_contribution']['shared_error_percentage'] = 0.0
        
        union_global_metrics[model] = global_metrics

    # 4. Comparative summary
    union_summary = {
        'total_union_errors': total_union_errors,
        'model_error_breakdown': error_contribution_stats,
        'model_comparison': {}
    }
    
    for model in model_names:
        total_errors = error_contribution_stats[model]['total_errors']
        unique_errors = error_contribution_stats[model]['unique_errors']
        total_all_errors = sum(stats['total_errors'] for stats in error_contribution_stats.values())
        
        union_summary['model_comparison'][model] = {
            'total_errors': total_errors,
            'unique_errors': unique_errors,
            'unique_error_rate': unique_errors / total_errors * 100 if total_errors > 0 else 0.0,
            'error_share_in_union': total_errors / total_all_errors * 100 if total_all_errors > 0 else 0.0
        }

    return union_global_metrics, individual_metrics, union_summary    
