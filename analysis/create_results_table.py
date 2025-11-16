"""Create LaTeX tables from experiment results for the report."""
import json
import pandas as pd
from pathlib import Path
from analysis.load_data import load_experiment_json, extract_experiment_data


def create_results_table(json_path: Path, dataset_name: str, save_path: Path = None):
    """
    Create a LaTeX table with all experiment results.
    
    Args:
        json_path: Path to experiment JSON file
        dataset_name: Name of dataset (PH2 or DRIVE)
        save_path: Optional path to save LaTeX table
    """
    experiment_dict = load_experiment_json(json_path)
    df = extract_experiment_data(experiment_dict)
    
    # Sort by Dice score (descending)
    df_sorted = df.sort_values('best_val_dice', ascending=False)
    
    # Create LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Segmentation performance on " + dataset_name + " dataset.}")
    latex_lines.append("\\label{tab:" + dataset_name.lower() + "_results}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Configuration & Dice & Sensitivity & Specificity \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in df_sorted.iterrows():
        config = f"{row['model']} + {row['loss']} + {row['optimizer']}"
        dice = f"{row['best_val_dice']:.4f}"
        sens = f"{row['best_val_sensitivity']:.4f}"
        spec = f"{row['best_val_specificity']:.4f}"
        
        # Escape special LaTeX characters
        config = config.replace('_', '\\_')
        
        latex_lines.append(f"{config} & {dice} & {sens} & {spec} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved LaTeX table to {save_path}")
    
    return latex_table


def create_summary_table(json_paths: dict, save_path: Path = None):
    """
    Create a summary table comparing best configurations across datasets.
    
    Args:
        json_paths: Dict mapping dataset names to JSON file paths
        save_path: Optional path to save LaTeX table
    """
    best_configs = {}
    
    for dataset_name, json_path in json_paths.items():
        experiment_dict = load_experiment_json(json_path)
        df = extract_experiment_data(experiment_dict)
        
        best_idx = df['best_val_dice'].idxmax()
        best_row = df.loc[best_idx]
        
        best_configs[dataset_name] = {
            'config': f"{best_row['model']} + {best_row['loss']} + {best_row['optimizer']}",
            'dice': best_row['best_val_dice'],
            'iou': best_row['best_val_iou'],
            'sensitivity': best_row['best_val_sensitivity'],
            'specificity': best_row['best_val_specificity'],
            'accuracy': best_row['best_val_accuracy']
        }
    
    # Create LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Best performing configurations on PH2 and DRIVE datasets.}")
    latex_lines.append("\\label{tab:best_configs_summary}")
    latex_lines.append("\\begin{tabular}{lccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Configuration & Dice & IoU & Sensitivity & Specificity \\\\")
    latex_lines.append("\\midrule")
    
    for dataset_name in ['PH2', 'DRIVE']:
        if dataset_name in best_configs:
            config = best_configs[dataset_name]
            config_str = config['config'].replace('_', '\\_')
            latex_lines.append(
                f"{dataset_name} & {config_str} & "
                f"{config['dice']:.4f} & {config['iou']:.4f} & "
                f"{config['sensitivity']:.4f} & {config['specificity']:.4f} \\\\"
            )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved summary LaTeX table to {save_path}")
    
    return latex_table


if __name__ == '__main__':
    output_dir = Path('./outputs')
    
    # Create tables for each dataset
    ph2_path = output_dir / 'segmentation_ph2_2.json'
    drive_path = output_dir / 'segmentation_DRIVE.json'
    
    if ph2_path.exists():
        print("\n=== PH2 Results Table ===")
        table = create_results_table(ph2_path, 'PH2', Path('./tables/ph2_results.tex'))
        print(table)
    
    if drive_path.exists():
        print("\n=== DRIVE Results Table ===")
        table = create_results_table(drive_path, 'DRIVE', Path('./tables/drive_results.tex'))
        print(table)
    
    # Create summary table
    if ph2_path.exists() and drive_path.exists():
        print("\n=== Summary Table ===")
        summary = create_summary_table(
            {'PH2': ph2_path, 'DRIVE': drive_path},
            Path('./tables/summary_table.tex')
        )
        print(summary)

