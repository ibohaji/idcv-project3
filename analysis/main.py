"""Main script to generate all analysis plots from experiment JSON files."""
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from analysis.load_data import load_all_experiments, extract_experiment_data
from analysis.plotting import (
    plot_training_curves,
    plot_metric_comparison,
    plot_heatmap_comparison,
    plot_all_metrics_comparison,
    plot_learning_curves_all_experiments,
    plot_ablation_study_summary,
    plot_dataset_comparison
)


def generate_all_plots(output_dir: Path, save_dir: Path, show: bool = False):
    """
    Generate all analysis plots from experiment JSON files.
    
    Args:
        output_dir: Directory containing experiment JSON files
        save_dir: Directory to save generated plots
        show: Whether to display plots (set to False for batch processing)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiments
    print("Loading experiment data...")
    all_experiments = load_all_experiments(output_dir)
    
    if len(all_experiments) == 0:
        print(f"No JSON files found in {output_dir}")
        return
    
    print(f"Loaded {len(all_experiments)} experiment file(s)")
    
    # Process each dataset
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
        
        # 1. Training curves for best configuration
        best_config_idx = df['best_val_dice'].idxmax()
        best_config = df.loc[best_config_idx]
        best_exp = None
        for exp in exp_dict['experiments']:
            if (exp['model'] == best_config['model'] and 
                exp['loss'] == best_config['loss'] and 
                exp['optimizer'] == best_config['optimizer']):
                best_exp = exp
                break
        
        if best_exp:
            results = best_exp['results']
            train_history = results.get('train_history', [])
            val_history = results.get('val_history', [])
            epochs = list(range(1, len(train_history) + 1))
            
            plot_training_curves(
                train_history, val_history, epochs,
                metrics=['loss', 'dice', 'iou', 'accuracy'],
                save_path=dataset_save_dir / f'{dataset_name}_best_config_training_curves.png',
                title=f'{dataset_name} - Best Configuration Training Curves\n'
                      f"{best_config['model']} + {best_config['loss']} + {best_config['optimizer']}",
                show=show
            )
        
        # 2. Metric comparisons by different factors
        for group_by in ['model', 'loss', 'optimizer']:
            plot_metric_comparison(
                df, 'best_val_dice', group_by=group_by,
                save_path=dataset_save_dir / f'{dataset_name}_dice_by_{group_by}.png',
                show=show
            )
        
        # 3. Heatmaps
        plot_heatmap_comparison(
            df, 'best_val_dice', row_var='model', col_var='loss',
            save_path=dataset_save_dir / f'{dataset_name}_dice_heatmap_model_loss.png',
            show=show
        )
        plot_heatmap_comparison(
            df, 'best_val_dice', row_var='model', col_var='optimizer',
            save_path=dataset_save_dir / f'{dataset_name}_dice_heatmap_model_optimizer.png',
            show=show
        )
        plot_heatmap_comparison(
            df, 'best_val_dice', row_var='loss', col_var='optimizer',
            save_path=dataset_save_dir / f'{dataset_name}_dice_heatmap_loss_optimizer.png',
            show=show
        )
        
        # 4. All metrics comparison
        plot_all_metrics_comparison(
            df,
            metrics=['best_val_dice', 'best_val_iou', 'best_val_accuracy', 
                    'best_val_sensitivity', 'best_val_specificity'],
            group_by='config',
            save_path=dataset_save_dir / f'{dataset_name}_all_metrics_comparison.png',
            title=f'{dataset_name} - All Metrics Comparison',
            show=show
        )
        
        # 5. Learning curves for all experiments
        for metric in ['dice', 'iou', 'loss']:
            plot_learning_curves_all_experiments(
                exp_dict, metric=metric,
                save_path=dataset_save_dir / f'{dataset_name}_learning_curves_{metric}.png',
                show=show
            )
        
        # 6. Ablation study summary
        plot_ablation_study_summary(
            df,
            save_path=dataset_save_dir / f'{dataset_name}_ablation_summary.png',
            show=show
        )
        
        print(f"  Generated plots saved to: {dataset_save_dir}")
    
    # 7. Cross-dataset comparison
    if len(all_experiments) > 1:
        print(f"\n{'='*60}")
        print("Generating cross-dataset comparisons...")
        print(f"{'='*60}")
        
        plot_dataset_comparison(
            all_experiments, metric='best_val_dice',
            save_path=save_dir / 'cross_dataset_comparison_dice.png',
            show=show
        )
        plot_dataset_comparison(
            all_experiments, metric='best_val_iou',
            save_path=save_dir / 'cross_dataset_comparison_iou.png',
            show=show
        )
    
    # 8. Combined summary table
    print(f"\n{'='*60}")
    print("Generating summary tables...")
    print(f"{'='*60}")
    
    all_dfs = []
    for exp_dict in all_experiments:
        df = extract_experiment_data(exp_dict)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save summary CSV
    summary_csv = save_dir / 'experiment_summary.csv'
    summary_cols = ['dataset', 'model', 'loss', 'optimizer', 
                   'best_val_dice', 'best_val_iou', 'best_val_accuracy',
                   'best_val_sensitivity', 'best_val_specificity', 'best_epoch',
                   'total_time_seconds']
    combined_df[summary_cols].to_csv(summary_csv, index=False)
    print(f"  Saved summary table to: {summary_csv}")
    
    # Print top configurations
    print("\n" + "="*60)
    print("TOP CONFIGURATIONS BY DATASET")
    print("="*60)
    for dataset in combined_df['dataset'].unique():
        dataset_df = combined_df[combined_df['dataset'] == dataset]
        top = dataset_df.nlargest(3, 'best_val_dice')
        print(f"\n{dataset}:")
        for idx, row in top.iterrows():
            print(f"  {row['model']} + {row['loss']} + {row['optimizer']}: "
                  f"Dice={row['best_val_dice']:.4f}, IoU={row['best_val_iou']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {save_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate analysis plots from experiment results')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory containing experiment JSON files (default: ./outputs)')
    parser.add_argument('--save-dir', type=str, default='./analysis/plots',
                       help='Directory to save generated plots (default: ./analysis/plots)')
    parser.add_argument('--show', action='store_true',
                       help='Display plots (default: False, saves only)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        return
    
    generate_all_plots(output_dir, save_dir, show=args.show)


if __name__ == '__main__':
    main()

