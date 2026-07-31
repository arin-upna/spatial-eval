"""
CLI script to compute classification and distance metrics for point cloud segmentation.
"""
import os
import sys
import argparse
from pathlib import Path
import yaml

if __package__:
    from .evaluation import evaluate_point_clouds, evaluate_union_errors
    from .display import display_evaluation_results
else:
    # Allow direct execution: python metrics/metrics.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from metrics.evaluation import evaluate_point_clouds, evaluate_union_errors
    from metrics.display import display_evaluation_results

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    """CLI entry point: evaluate and save/display segmentation metrics."""
    parser = argparse.ArgumentParser(
        description="Classification and distance metrics for point cloud segmentation"
    )
    
    # Main arguments
    parser.add_argument(
        "--pred_dir", "-p", action="append",
        help="Directory with prediction files (can be used multiple times for union errors)",
    )
    parser.add_argument(
        "--pred_dirs", nargs="+",
        help="List of directories with prediction files (alternative to multiple -p)",
    )
    parser.add_argument(
        "--gt_dir", "-g", default=None,
        help="Directory with ground-truth files (optional: if omitted, each file in pred_dir must contain both 'classification' and 'Prediction' fields)",
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Dataset YAML configuration file",
    )
    parser.add_argument(
        "--output_dir", "-o", default=None,
        help="Base directory where results will be saved (optional)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--file_extension", "-e", default=None,
        help="File extension (default: from config)",
    )
    parser.add_argument(
        "--suffix", "-s", default=None,
        help="Suffix to remove from prediction filenames",
    )
    parser.add_argument(
        "--prefix", "-x", default=None,
        help="Prefix to remove from prediction filenames",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Resolution for saved figures (default: 300)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=None,
        help="Parallel file workers for standard evaluation (default: min(32, CPUs))",
    )
    parser.add_argument(
        "--union_errors", action="store_true",
        help="Use evaluate_union_errors when there are multiple prediction directories",
    )
    # parser.add_argument("--include_fn", action="store_true", help="Include false negative metrics")

    
    args = parser.parse_args()
    
    # Process prediction directories
    pred_dirs = []
    if args.pred_dirs:
        pred_dirs = args.pred_dirs
    elif args.pred_dir:
        pred_dirs = args.pred_dir
    else:
        print("Error: You must provide at least one prediction directory with -p or --pred_dirs")
        return 1
    
    # Decide whether to use union_errors
    use_union = args.union_errors or (len(pred_dirs) > 1)
    
    # Load dataset configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Extract configuration parameters
    classification_dict = config['classes']['names']
    class_distance_limits = config['classes']['distance_limits']
    ignore_classes = config['classes'].get('ignore_in_metrics', [])
    class_grouping = config['classes'].get('grouping', {})
    # Use source_format as file_extension
    file_extension = args.file_extension or config['dataset'].get('source_format', '.laz')
    
    print(f"Evaluating with configuration: {config['dataset']['name']}")
    print(f"Classes: {list(classification_dict.keys())}")
    print(f"Extension: {file_extension}")
    single_dir_mode = args.gt_dir is None
    if use_union:
        if single_dir_mode:
            print("Error: --gt_dir is required when using union errors mode")
            return 1
        print(f"Mode: Union Errors ({len(pred_dirs)} models)")
    elif single_dir_mode:
        print("Mode: Single-directory (GT in 'classification', predictions in 'Prediction')")
    else:
        print("Mode: Standard evaluation")
    worker_count = args.workers or min(32, os.cpu_count() or 1)
    print(f"Workers: {worker_count}")
    
    # Run evaluation
    if use_union:
        # Evaluation with union errors
        union_global_metrics, individual_metrics_dict, union_summary = evaluate_union_errors(
            gt_dir=args.gt_dir,
            pred_dirs=pred_dirs,
            classification_dict=classification_dict,
            file_extension=file_extension,
            suffix=args.suffix,
            prefix=args.prefix,
            class_distance_limits=class_distance_limits,
            ignore_classes=ignore_classes,
            class_grouping=class_grouping,
            include_fn=False,
            match_by_name_only=True,
        )
        
        if union_global_metrics is None or len(union_global_metrics) == 0:
            print("Error: Could not compute union errors metrics")
            return 1
        
        # Create mapping from model_name to its full directory
        model_to_dir = {}
        for pred_dir in pred_dirs:
            model_name = os.path.basename(pred_dir)
            model_to_dir[model_name] = pred_dir
        
        # Save results for each model in its own directory
        print("Saving union errors results...")
        saved_dirs = []
        for model_name, global_metrics in union_global_metrics.items():
            print(f"\nProcesando modelo: {model_name}")
            
            # Determine output directory for this specific model
            if args.output_dir:
                model_out_dir = os.path.join(args.output_dir, model_name)
            else:
                # Save inside the corresponding model directory
                model_dir = model_to_dir.get(model_name, pred_dirs[0])
                model_out_dir = os.path.join(model_dir, "hard_metrics")
            
            os.makedirs(model_out_dir, exist_ok=True)
            saved_dirs.append(model_out_dir)
            
            # Get individual metrics for this model
            individual_metrics = individual_metrics_dict.get(model_name, {})
            
            display_evaluation_results(
                global_metrics=global_metrics,
                individual_metrics=individual_metrics,
                save_path=model_out_dir,
                markdown=True,  # CLI always outputs markdown
                class_distance_limits=class_distance_limits,
                class_names=classification_dict,
                dpi=args.dpi,
            )
        
        print("\nHard-points evaluation completed.")
        print("Results saved in:")
        for saved_dir in saved_dirs:
            print(f"  - {saved_dir}")
        
    else:
        # Determine output directory for standard evaluation
        if args.output_dir:
            out_base = args.output_dir
        else:
            out_base = os.path.join(pred_dirs[0], "metrics")
        os.makedirs(out_base, exist_ok=True)
        
        # Standard evaluation
        if single_dir_mode:
            # Single-directory mode: each file has both classification (GT) and Prediction
            global_metrics, individual_metrics = evaluate_point_clouds(
                gt_dir=pred_dirs[0],
                pred_dir=None,
                classification_dict=classification_dict,
                file_extension=file_extension,
                class_distance_limits=class_distance_limits,
                ignore_classes=ignore_classes,
                class_grouping=class_grouping,
                tp_distance=False,
                error_idx=None,
                include_fn=False,
                pred_field="Prediction",
                max_workers=worker_count,
            )
        else:
            global_metrics, individual_metrics = evaluate_point_clouds(
                gt_dir=args.gt_dir,
                pred_dir=pred_dirs[0],
                classification_dict=classification_dict,
                file_extension=file_extension,
                suffix=args.suffix,
                prefix=args.prefix,
                class_distance_limits=class_distance_limits,
                ignore_classes=ignore_classes,
                class_grouping=class_grouping,
                tp_distance=False,
                error_idx=None,
                include_fn=False,
                max_workers=worker_count,
            )
        
        if global_metrics is None:
            print("Error: Could not compute metrics")
            return 1
        
        # Display and save results
        print("Saving results...")
        display_evaluation_results(
            global_metrics=global_metrics,
            individual_metrics=individual_metrics,
            save_path=out_base,
            markdown=True,  # CLI always outputs markdown
            class_distance_limits=class_distance_limits,
            class_names=classification_dict,
            dpi=args.dpi,
        )
        
        print(f"\nEvaluation completed. Results in: {out_base}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
