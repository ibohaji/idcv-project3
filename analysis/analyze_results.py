"""Comprehensive analysis script to extract insights from experiment JSON files."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from analysis.load_data import load_experiment_json, extract_experiment_data


def extract_gamma_from_loss(loss_name: str) -> Optional[int]:
    """Extract gamma value from loss name (e.g., 'Focal1' -> 1, 'Focal2' -> 2)."""
    loss_name = loss_name.lower()
    if loss_name.startswith('focal'):
        digits = ''.join(ch for ch in loss_name if ch.isdigit())
        if digits:
            return int(digits)
    return None


def analyze_dataset(experiment_dict: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Comprehensive analysis of a single dataset's experiments.
    
    Returns a dictionary with:
    - summary: Overall statistics
    - by_gamma: Analysis grouped by Focal Loss gamma values
    - by_optimizer: Analysis grouped by optimizer
    - best_configs: Top performing configurations
    - training_dynamics: Convergence analysis
    """
    dataset_name = experiment_dict['dataset']
    experiments = experiment_dict['experiments']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {dataset_name}")
        print(f"{'='*80}")
        print(f"Total experiments: {len(experiments)}")
    
    # Convert to DataFrame
    df = extract_experiment_data(experiment_dict)
    
    # Extract gamma values
    df['gamma'] = df['loss'].apply(extract_gamma_from_loss)
    
    analysis = {
        'dataset': dataset_name,
        'total_experiments': len(experiments),
        'summary': {},
        'by_gamma': {},
        'by_optimizer': {},
        'by_loss': {},
        'best_configs': {},
        'training_dynamics': {}
    }
    
    # Overall summary
    metrics = ['best_val_dice', 'best_val_iou', 'best_val_sensitivity', 'best_val_specificity', 'best_val_accuracy']
    analysis['summary'] = {
        'mean': df[metrics].mean().to_dict(),
        'std': df[metrics].std().to_dict(),
        'min': df[metrics].min().to_dict(),
        'max': df[metrics].max().to_dict(),
        'best_dice': df['best_val_dice'].max(),
        'best_dice_config': df.loc[df['best_val_dice'].idxmax(), ['model', 'loss', 'optimizer', 'best_val_dice', 'best_val_sensitivity', 'best_val_specificity']].to_dict()
    }
    
    if verbose:
        print(f"\n[OVERALL SUMMARY]")
        print(f"  Best Dice: {analysis['summary']['best_dice']:.4f}")
        print(f"  Best Config: {analysis['summary']['best_dice_config']['model']} + "
              f"{analysis['summary']['best_dice_config']['loss']} + "
              f"{analysis['summary']['best_dice_config']['optimizer']}")
        print(f"  Mean Dice: {analysis['summary']['mean']['best_val_dice']:.4f} +/- {analysis['summary']['std']['best_val_dice']:.4f}")
        print(f"  Mean Sensitivity: {analysis['summary']['mean']['best_val_sensitivity']:.4f} +/- {analysis['summary']['std']['best_val_sensitivity']:.4f}")
        print(f"  Mean Specificity: {analysis['summary']['mean']['best_val_specificity']:.4f} +/- {analysis['summary']['std']['best_val_specificity']:.4f}")
    
    # Analysis by gamma value
    focal_experiments = df[df['gamma'].notna()]
    if len(focal_experiments) > 0:
        gamma_groups = focal_experiments.groupby('gamma')
        for gamma, group in gamma_groups:
            analysis['by_gamma'][int(gamma)] = {
                'count': len(group),
                'mean_dice': group['best_val_dice'].mean(),
                'std_dice': group['best_val_dice'].std(),
                'mean_sensitivity': group['best_val_sensitivity'].mean(),
                'mean_specificity': group['best_val_specificity'].mean(),
                'best_dice': group['best_val_dice'].max(),
                'best_config': group.loc[group['best_val_dice'].idxmax(), ['model', 'loss', 'optimizer', 'best_val_dice']].to_dict()
            }
        
        if verbose:
            print(f"\n[ANALYSIS BY GAMMA VALUE]")
            for gamma in sorted(analysis['by_gamma'].keys()):
                g = analysis['by_gamma'][gamma]
                print(f"  gamma={gamma}:")
                print(f"    Mean Dice: {g['mean_dice']:.4f} +/- {g['std_dice']:.4f}")
                print(f"    Best Dice: {g['best_dice']:.4f}")
                print(f"    Best Config: {g['best_config']['model']} + {g['best_config']['optimizer']}")
                print(f"    Sensitivity: {g['mean_sensitivity']:.4f}, Specificity: {g['mean_specificity']:.4f}")
    
    # Analysis by optimizer
    optimizer_groups = df.groupby('optimizer')
    for opt, group in optimizer_groups:
        analysis['by_optimizer'][opt] = {
            'count': len(group),
            'mean_dice': group['best_val_dice'].mean(),
            'std_dice': group['best_val_dice'].std(),
            'mean_sensitivity': group['best_val_sensitivity'].mean(),
            'mean_specificity': group['best_val_specificity'].mean(),
            'best_dice': group['best_val_dice'].max(),
            'best_config': group.loc[group['best_val_dice'].idxmax(), ['model', 'loss', 'optimizer', 'best_val_dice', 'best_val_sensitivity', 'best_val_specificity']].to_dict()
        }
    
    if verbose:
        print(f"\n[ANALYSIS BY OPTIMIZER]")
        for opt in sorted(analysis['by_optimizer'].keys()):
            o = analysis['by_optimizer'][opt]
            print(f"  {opt.upper()}:")
            print(f"    Mean Dice: {o['mean_dice']:.4f} +/- {o['std_dice']:.4f}")
            print(f"    Best Dice: {o['best_dice']:.4f}")
            print(f"    Best Config: {o['best_config']['model']} + {o['best_config']['loss']}")
            print(f"    Sensitivity: {o['mean_sensitivity']:.4f}, Specificity: {o['mean_specificity']:.4f}")
    
    # Analysis by loss function
    loss_groups = df.groupby('loss')
    for loss, group in loss_groups:
        analysis['by_loss'][loss] = {
            'count': len(group),
            'mean_dice': group['best_val_dice'].mean(),
            'std_dice': group['best_val_dice'].std(),
            'mean_sensitivity': group['best_val_sensitivity'].mean(),
            'mean_specificity': group['best_val_specificity'].mean(),
            'best_dice': group['best_val_dice'].max(),
            'best_config': group.loc[group['best_val_dice'].idxmax(), ['model', 'loss', 'optimizer', 'best_val_dice']].to_dict()
        }
    
    # Top configurations
    top_n = 5
    top_configs = df.nlargest(top_n, 'best_val_dice')
    analysis['best_configs'] = []
    for idx, row in top_configs.iterrows():
        config = {
            'rank': len(analysis['best_configs']) + 1,
            'model': row['model'],
            'loss': row['loss'],
            'optimizer': row['optimizer'],
            'dice': row['best_val_dice'],
            'iou': row['best_val_iou'],
            'sensitivity': row['best_val_sensitivity'],
            'specificity': row['best_val_specificity'],
            'accuracy': row['best_val_accuracy'],
            'best_epoch': row['best_epoch']
        }
        analysis['best_configs'].append(config)
    
    if verbose:
        print(f"\n[TOP {top_n} CONFIGURATIONS]")
        for config in analysis['best_configs']:
            print(f"  #{config['rank']}: {config['model']} + {config['loss']} + {config['optimizer']}")
            print(f"     Dice: {config['dice']:.4f}, IoU: {config['iou']:.4f}")
            print(f"     Sensitivity: {config['sensitivity']:.4f}, Specificity: {config['specificity']:.4f}")
    
    # Training dynamics analysis
    for exp in experiments:
        results = exp['results']
        val_history = results.get('val_history', [])
        if len(val_history) > 0:
            dice_history = [h.get('dice', 0) for h in val_history]
            sensitivity_history = [h.get('sensitivity', 0) for h in val_history]
            
            # Find convergence point (when dice stops improving significantly)
            if len(dice_history) > 10:
                # Look for plateau (last 10% of training)
                plateau_start = int(len(dice_history) * 0.9)
                plateau_dice = dice_history[plateau_start:]
                max_dice = max(dice_history)
                plateau_mean = np.mean(plateau_dice)
                
                # Find peak epoch
                peak_epoch = np.argmax(dice_history) + 1
                
                config_key = f"{exp['model']}_{exp['loss']}_{exp['optimizer']}"
                analysis['training_dynamics'][config_key] = {
                    'total_epochs': len(val_history),
                    'peak_epoch': int(peak_epoch),
                    'peak_dice': float(max_dice),
                    'final_dice': float(dice_history[-1]),
                    'plateau_dice': float(plateau_mean),
                    'convergence_epoch': int(peak_epoch) if abs(max_dice - plateau_mean) < 0.01 else len(val_history)
                }
    
    # Gamma-Optimizer interaction analysis
    if len(focal_experiments) > 0:
        analysis['gamma_optimizer_interaction'] = {}
        for gamma in focal_experiments['gamma'].unique():
            gamma_data = focal_experiments[focal_experiments['gamma'] == gamma]
            interaction = {}
            for opt in gamma_data['optimizer'].unique():
                opt_data = gamma_data[gamma_data['optimizer'] == opt]
                interaction[opt] = {
                    'mean_dice': opt_data['best_val_dice'].mean(),
                    'best_dice': opt_data['best_val_dice'].max()
                }
            analysis['gamma_optimizer_interaction'][int(gamma)] = interaction
        
        if verbose:
            print(f"\n[GAMMA-OPTIMIZER INTERACTION]")
            for gamma in sorted(analysis['gamma_optimizer_interaction'].keys()):
                print(f"  gamma={gamma}:")
                for opt, stats in analysis['gamma_optimizer_interaction'][gamma].items():
                    print(f"    {opt.upper()}: Mean Dice {stats['mean_dice']:.4f}, Best {stats['best_dice']:.4f}")
    
    return analysis


def print_detailed_report(analysis: Dict[str, Any]):
    """Print a formatted detailed report."""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS REPORT: {analysis['dataset']}")
    print(f"{'='*80}")
    
    # Summary table
    print(f"\n[METRIC SUMMARY TABLE]")
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print(f"{'-'*68}")
    for metric in ['best_val_dice', 'best_val_iou', 'best_val_sensitivity', 'best_val_specificity']:
        mean = analysis['summary']['mean'][metric]
        std = analysis['summary']['std'][metric]
        min_val = analysis['summary']['min'][metric]
        max_val = analysis['summary']['max'][metric]
        print(f"{metric:<20} {mean:<12.4f} {std:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")
    
    # Best configurations table
    print(f"\n[TOP CONFIGURATIONS TABLE]")
    print(f"{'Rank':<6} {'Model':<15} {'Loss':<12} {'Optimizer':<12} {'Dice':<8} {'Sens':<8} {'Spec':<8}")
    print(f"{'-'*80}")
    for config in analysis['best_configs']:
        print(f"{config['rank']:<6} {config['model']:<15} {config['loss']:<12} {config['optimizer']:<12} "
              f"{config['dice']:<8.4f} {config['sensitivity']:<8.4f} {config['specificity']:<8.4f}")


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results from JSON files')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory containing experiment JSON files')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='Specific JSON files to analyze (default: all DRIVE files)')
    parser.add_argument('--save-report', type=str, default=None,
                       help='Path to save detailed report as JSON')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed analysis')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Find JSON files
    if args.files:
        json_files = [output_dir / f for f in args.files]
    else:
        # Default: find DRIVE experiment files
        json_files = list(output_dir.glob("*DRIVE*.json"))
        if len(json_files) == 0:
            # Try common names
            common_names = ['segmentation_DRIVE.json', 'DRIVE_GAMMA_STUDY2.json']
            json_files = [output_dir / f for f in common_names if (output_dir / f).exists()]
    
    if len(json_files) == 0:
        print(f"[ERROR] No JSON files found in {output_dir}")
        print(f"   Looking for files matching: *DRIVE*.json")
        return
    
    print(f"[INFO] Found {len(json_files)} experiment file(s):")
    for f in json_files:
        print(f"   - {f.name}")
    
    # Analyze each file
    all_analyses = []
    for json_file in json_files:
        if not json_file.exists():
            print(f"⚠️  File not found: {json_file}")
            continue
        
        try:
            experiment_dict = load_experiment_json(json_file)
            analysis = analyze_dataset(experiment_dict, verbose=args.verbose)
            all_analyses.append(analysis)
            
            if args.verbose:
                print_detailed_report(analysis)
        except Exception as e:
            print(f"[ERROR] Error analyzing {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined report
    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(all_analyses, f, indent=2)
        print(f"\n[SAVED] Detailed report saved to: {report_path}")
    
    # Summary across all files
    if len(all_analyses) > 1:
        print(f"\n{'='*80}")
        print("CROSS-FILE SUMMARY")
        print(f"{'='*80}")
        for analysis in all_analyses:
            print(f"\n{analysis['dataset']}:")
            print(f"  Best Dice: {analysis['summary']['best_dice']:.4f}")
            print(f"  Best Config: {analysis['summary']['best_dice_config']['model']} + "
                  f"{analysis['summary']['best_dice_config']['loss']} + "
                  f"{analysis['summary']['best_dice_config']['optimizer']}")


if __name__ == '__main__':
    main()

