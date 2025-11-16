"""Create essential LaTeX tables for the report (2-3 tables only)."""
import json
from pathlib import Path
from analysis.load_data import load_experiment_json, extract_experiment_data


def extract_gamma_from_loss(loss_name: str):
    """Extract gamma value from loss name."""
    loss_name = loss_name.lower()
    if loss_name.startswith('focal'):
        digits = ''.join(ch for ch in loss_name if ch.isdigit())
        if digits:
            return int(digits)
    return None


def create_summary_table(ph2_path: Path, drive_path: Path, save_path: Path = None):
    """
    Table 1: Summary comparison of best configurations on both datasets.
    This is the most important table showing the contrast.
    """
    ph2_dict = load_experiment_json(ph2_path)
    drive_dict = load_experiment_json(drive_path)
    
    ph2_df = extract_experiment_data(ph2_dict)
    drive_df = extract_experiment_data(drive_dict)
    
    ph2_best = ph2_df.loc[ph2_df['best_val_dice'].idxmax()]
    drive_best = drive_df.loc[drive_df['best_val_dice'].idxmax()]
    
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Best performing configurations on PH2 and DRIVE datasets.}")
    latex_lines.append("\\label{tab:best_configs_summary}")
    latex_lines.append("\\begin{tabular}{lccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Dataset & Configuration & Dice & IoU & Sensitivity & Specificity \\\\")
    latex_lines.append("\\midrule")
    
    # PH2
    ph2_config = f"{ph2_best['model']} + {ph2_best['loss']} + {ph2_best['optimizer']}".replace('_', '\\_')
    latex_lines.append(
        f"PH2 & {ph2_config} & "
        f"{ph2_best['best_val_dice']:.4f} & {ph2_best['best_val_iou']:.4f} & "
        f"{ph2_best['best_val_sensitivity']:.4f} & {ph2_best['best_val_specificity']:.4f} \\\\"
    )
    
    # DRIVE
    drive_config = f"{drive_best['model']} + {drive_best['loss']} + {drive_best['optimizer']}".replace('_', '\\_')
    latex_lines.append(
        f"DRIVE & {drive_config} & "
        f"{drive_best['best_val_dice']:.4f} & {drive_best['best_val_iou']:.4f} & "
        f"{drive_best['best_val_sensitivity']:.4f} & {drive_best['best_val_specificity']:.4f} \\\\"
    )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"[OK] Saved summary table to {save_path}")
    
    return latex_table


def create_drive_results_table(drive_path: Path, save_path: Path = None):
    """
    Table 2: DRIVE results showing BCE failures and Focal Loss success.
    This highlights the class imbalance problem.
    """
    drive_dict = load_experiment_json(drive_path)
    df = extract_experiment_data(drive_dict)
    
    # Sort by Dice (descending) to show best first
    df_sorted = df.sort_values('best_val_dice', ascending=False)
    
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Segmentation performance on DRIVE dataset. BCE and Weighted BCE failed completely (Dice 0.00), while Focal Loss successfully addressed the class imbalance.}")
    latex_lines.append("\\label{tab:drive_results}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Configuration & Dice & IoU & Sensitivity & Specificity \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in df_sorted.iterrows():
        config = f"{row['model']} + {row['loss']} + {row['optimizer']}".replace('_', '\\_')
        dice = f"{row['best_val_dice']:.4f}"
        iou = f"{row['best_val_iou']:.4f}"
        sens = f"{row['best_val_sensitivity']:.4f}"
        spec = f"{row['best_val_specificity']:.4f}"
        
        latex_lines.append(f"{config} & {dice} & {iou} & {sens} & {spec} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"[OK] Saved DRIVE results table to {save_path}")
    
    return latex_table


def create_gamma_ablation_table(drive_gamma_path: Path, save_path: Path = None):
    """
    Table 3 (optional): DRIVE gamma ablation study.
    Only include if you want to show the detailed gamma analysis.
    """
    drive_dict = load_experiment_json(drive_gamma_path)
    df = extract_experiment_data(drive_dict)
    
    # Extract gamma values
    df['gamma'] = df['loss'].apply(extract_gamma_from_loss)
    focal_df = df[df['gamma'].notna()].copy()
    
    if len(focal_df) == 0:
        return None
    
    focal_df = focal_df.sort_values('gamma')
    
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Focal Loss $\\gamma$ ablation study on DRIVE dataset.}")
    latex_lines.append("\\label{tab:drive_gamma_ablation}")
    latex_lines.append("\\begin{tabular}{lcccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("$\\gamma$ & Dice & IoU & Sensitivity & Specificity \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in focal_df.iterrows():
        gamma = int(row['gamma'])
        dice = f"{row['best_val_dice']:.4f}"
        iou = f"{row['best_val_iou']:.4f}"
        sens = f"{row['best_val_sensitivity']:.4f}"
        spec = f"{row['best_val_specificity']:.4f}"
        
        latex_lines.append(f"{gamma} & {dice} & {iou} & {sens} & {spec} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"[OK] Saved gamma ablation table to {save_path}")
    
    return latex_table


if __name__ == '__main__':
    output_dir = Path('./outputs')
    tables_dir = Path('./tables')
    tables_dir.mkdir(exist_ok=True)
    
    ph2_path = output_dir / 'segmentation_ph2_2.json'
    drive_path = output_dir / 'segmentation_DRIVE.json'
    
    print("=" * 60)
    print("CREATING ESSENTIAL TABLES FOR REPORT")
    print("=" * 60)
    
    if ph2_path.exists() and drive_path.exists():
        # Table 1: Summary (MOST IMPORTANT)
        print("\n[Table 1] Summary Comparison")
        table1 = create_summary_table(
            ph2_path, drive_path,
            tables_dir / 'summary_table.tex'
        )
        print(table1)
        
        # Table 2: DRIVE Results (shows the problem)
        print("\n[Table 2] DRIVE Results")
        table2 = create_drive_results_table(
            drive_path,
            tables_dir / 'drive_results.tex'
        )
        print(table2)
        
        # Table 3: Gamma Ablation (optional - only if you want detailed gamma analysis)
        print("\n[Table 3] DRIVE Gamma Ablation (OPTIONAL)")
        gamma_files = ['DRIVE_GAMMA_STUDY2.json', 'DRIVE_GAMMA_STUDY.json']
        for gamma_file in gamma_files:
            gamma_path = output_dir / gamma_file
            if gamma_path.exists():
                table3 = create_gamma_ablation_table(
                    gamma_path,
                    tables_dir / 'drive_gamma_ablation.tex'
                )
                if table3:
                    print(table3)
                break
        
        print("\n" + "=" * 60)
        print("RECOMMENDATION: Use Table 1 (Summary) + Table 2 (DRIVE Results)")
        print("Table 3 (Gamma Ablation) is optional - only if you want detailed gamma analysis")
        print("=" * 60)
    else:
        print(f"Error: Missing experiment files")
        print(f"  PH2: {ph2_path.exists()}")
        print(f"  DRIVE: {drive_path.exists()}")

