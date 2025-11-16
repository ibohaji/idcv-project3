"""Main script to generate consolidated, multi-dimensional plots for scientific reports."""
import argparse
from pathlib import Path
from typing import List, Dict, Any

from analysis.load_data import load_all_experiments, extract_experiment_data
from analysis.plotting_consolidated import (
    plot_all_metrics_by_model,
    plot_metrics_by_optimizer_and_loss,
    plot_best_config_per_metric,
    plot_training_curves_comparison,
    plot_ablation_summary_consolidated,
    plot_key_metrics_over_epochs,
    plot_metric_tradeoffs,
    plot_comprehensive_metrics_comparison,
    plot_convergence_comparison
)


def generate_all_plots(output_dir: Path, save_dir: Path, show: bool = False):
    """
    Generate consolidated, multi-dimensional plots for scientific reports.
    Only generates strategic plots that show multiple dimensions together.
    
    Args:
        output_dir: Directory containing experiment JSON files
        save_dir: Directory to save generated plots
        show: Whether to display plots (set to False for batch processing)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load only specific experiment files
    print("Loading experiment data...")
    specific_files = ['segmentation_ph2_2.json', 'segmentation_DRIVE.json']
    all_experiments = []
    for filename in specific_files:
        json_path = output_dir / filename
        if json_path.exists():
            from analysis.load_data import load_experiment_json
            data = load_experiment_json(json_path)
            all_experiments.append(data)
            print(f"  Loaded: {filename}")
        else:
            print(f"  Warning: {filename} not found")
    
    if len(all_experiments) == 0:
        print(f"No experiment files found in {output_dir}")
        return
    
    print(f"Loaded {len(all_experiments)} experiment file(s)")
    
    # Process each dataset - generate only strategic consolidated plots
    for exp_dict in all_experiments:
        dataset_name = exp_dict['dataset']
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_save_dir = save_dir / dataset_name
        dataset_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data to DataFrame
        df = extract_experiment_data(exp_dict)
        print(f"  Extracted {len(df)} experiment configurations")
        
        print("  Generating consolidated plots...")
        
        # 1. All metrics by model (single plot showing dice, iou, sensitivity, specificity)
        plot_all_metrics_by_model(
            df,
            save_path=dataset_save_dir / f'{dataset_name}_all_metrics_by_model.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 2. Metrics by optimizer and loss (shows interaction)
        plot_metrics_by_optimizer_and_loss(
            df,
            save_path=dataset_save_dir / f'{dataset_name}_metrics_by_optimizer_loss.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 3. Best configuration per metric (which config wins for each metric)
        plot_best_config_per_metric(
            df,
            save_path=dataset_save_dir / f'{dataset_name}_best_config_per_metric.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 4. Training curves for best configuration (multiple metrics together)
        plot_training_curves_comparison(
            exp_dict,
            metrics=['dice', 'iou', 'loss'],
            save_path=dataset_save_dir / f'{dataset_name}_best_training_curves.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 5. Comprehensive ablation summary (all comparisons in one figure)
        plot_ablation_summary_consolidated(
            df,
            save_path=dataset_save_dir / f'{dataset_name}_ablation_summary.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 6. Key metrics over epochs (dice, sensitivity, specificity) - SCIENTIFIC FOCUS
        plot_key_metrics_over_epochs(
            exp_dict,
            top_n=5,
            save_path=dataset_save_dir / f'{dataset_name}_key_metrics_over_epochs.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 7. Metric trade-offs (Dice vs Sensitivity, Dice vs Specificity)
        plot_metric_tradeoffs(
            exp_dict,
            save_path=dataset_save_dir / f'{dataset_name}_metric_tradeoffs.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 8. Comprehensive metrics comparison (all configs, easy to read)
        plot_comprehensive_metrics_comparison(
            df,
            save_path=dataset_save_dir / f'{dataset_name}_comprehensive_metrics.png',
            dataset_name=dataset_name,
            show=show
        )
        
        # 9. Convergence comparison for dice, sensitivity, specificity
        for metric in ['dice', 'sensitivity', 'specificity']:
            plot_convergence_comparison(
                exp_dict,
                metric=metric,
                top_n=6,
                save_path=dataset_save_dir / f'{dataset_name}_convergence_{metric}.png',
                dataset_name=dataset_name,
                show=show
            )
        
        print(f"  Generated 9 scientific plots saved to: {dataset_save_dir}")
    
    print(f"\n{'='*60}")
    print("Plot generation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate consolidated analysis plots")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing experiment JSON files"
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("analysis/plots"),
        help="Directory to save generated plots"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots (default: False, saves only)"
    )
    
    args = parser.parse_args()
    generate_all_plots(args.output_dir, args.save_dir, args.show)
