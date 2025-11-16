"""Create LaTeX table for DRIVE gamma ablation study."""
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


def create_gamma_ablation_table(json_path: Path, save_path: Path = None):
    """Create table showing gamma ablation results for DRIVE."""
    experiment_dict = load_experiment_json(json_path)
    df = extract_experiment_data(experiment_dict)
    
    # Filter to Focal Loss experiments and extract gamma
    df['gamma'] = df['loss'].apply(extract_gamma_from_loss)
    focal_df = df[df['gamma'].notna()].copy()
    
    if len(focal_df) == 0:
        print("No Focal Loss experiments found with gamma values")
        return None
    
    # Sort by gamma
    focal_df = focal_df.sort_values('gamma')
    
    # Create LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Focal Loss $\gamma$ ablation study on DRIVE dataset.}")
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
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_table)
        print(f"Saved gamma ablation table to {save_path}")
    
    return latex_table


if __name__ == '__main__':
    # Try to find DRIVE gamma study file
    output_dir = Path('./outputs')
    
    # Try different possible filenames
    possible_files = [
        'DRIVE_GAMMA_STUDY2.json',
        'segmentation_DRIVE.json',
        'DRIVE_GAMMA_STUDY.json'
    ]
    
    for filename in possible_files:
        filepath = output_dir / filename
        if filepath.exists():
            print(f"\n=== DRIVE Gamma Ablation Table (from {filename}) ===")
            table = create_gamma_ablation_table(
                filepath,
                Path('./tables/drive_gamma_ablation.tex')
            )
            if table:
                print(table)
            break
    else:
        print("No DRIVE experiment file found with gamma ablation data")

