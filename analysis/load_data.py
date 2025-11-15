"""Functions to load and parse experiment JSON files."""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


def load_experiment_json(json_path: Path) -> Dict[str, Any]:
    """Load a single experiment JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_all_experiments(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all experiment JSON files from output directory."""
    json_files = list(output_dir.glob("*.json"))
    experiments = []
    for json_file in json_files:
        data = load_experiment_json(json_file)
        experiments.append(data)
    return experiments


def extract_experiment_data(experiment_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract experiment data into a pandas DataFrame for easy analysis.
    
    Each row represents one experiment configuration (model + loss + optimizer).
    """
    dataset_name = experiment_dict['dataset']
    experiments = experiment_dict['experiments']
    
    rows = []
    for exp in experiments:
        results = exp['results']
        
        # Extract best metrics
        best_val_metrics = results.get('best_val_metrics', {})
        final_train_metrics = results.get('final_train_metrics', {})
        final_val_metrics = results.get('final_val_metrics', {})
        
        row = {
            'dataset': dataset_name,
            'model': exp['model'],
            'loss': exp['loss'],
            'optimizer': exp['optimizer'],
            'config': exp.get('config', ''),
            
            # Best epoch metrics
            'best_epoch': results.get('best_epoch', 0),
            
            # Best validation metrics (from best_val_metrics dict)
            'best_val_dice': best_val_metrics.get('dice', results.get('best_val_dice', 0.0)),
            'best_val_iou': best_val_metrics.get('iou', 0.0),
            'best_val_accuracy': best_val_metrics.get('accuracy', 0.0),
            'best_val_sensitivity': best_val_metrics.get('sensitivity', 0.0),
            'best_val_specificity': best_val_metrics.get('specificity', 0.0),
            'best_val_loss': best_val_metrics.get('loss', results.get('best_val_loss', float('inf'))),
            
            # Final training metrics
            'final_train_loss': final_train_metrics.get('loss', 0.0),
            'final_train_dice': final_train_metrics.get('dice', 0.0),
            'final_train_iou': final_train_metrics.get('iou', 0.0),
            'final_train_accuracy': final_train_metrics.get('accuracy', 0.0),
            'final_train_sensitivity': final_train_metrics.get('sensitivity', 0.0),
            'final_train_specificity': final_train_metrics.get('specificity', 0.0),
            
            # Final validation metrics
            'final_val_loss': final_val_metrics.get('loss', 0.0),
            'final_val_dice': final_val_metrics.get('dice', 0.0),
            'final_val_iou': final_val_metrics.get('iou', 0.0),
            'final_val_accuracy': final_val_metrics.get('accuracy', 0.0),
            'final_val_sensitivity': final_val_metrics.get('sensitivity', 0.0),
            'final_val_specificity': final_val_metrics.get('specificity', 0.0),
            
            # Training info
            'total_epochs': results.get('total_epochs', 0),
            'total_time_seconds': results.get('total_time_seconds', 0),
            'learning_rate': results.get('parameters', {}).get('learning_rate', 0.0),
            'batch_size': results.get('parameters', {}).get('batch_size', 0),
            
            # Store full history for plotting
            'train_history': results.get('train_history', []),
            'val_history': results.get('val_history', [])
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def get_training_curves(experiment_dict: Dict[str, Any], 
                       model: Optional[str] = None,
                       loss: Optional[str] = None,
                       optimizer: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract training curves (per-epoch metrics) for specific experiment(s).
    
    Returns dict with keys: 'train_history', 'val_history', 'epochs'
    """
    experiments = experiment_dict['experiments']
    
    # Filter experiments
    filtered = []
    for exp in experiments:
        if model and exp['model'] != model:
            continue
        if loss and exp['loss'] != loss:
            continue
        if optimizer and exp['optimizer'] != optimizer:
            continue
        filtered.append(exp)
    
    if len(filtered) == 0:
        return {'train_history': [], 'val_history': [], 'epochs': []}
    
    # Get first matching experiment (or combine multiple if needed)
    exp = filtered[0]
    results = exp['results']
    
    train_history = results.get('train_history', [])
    val_history = results.get('val_history', [])
    epochs = list(range(1, len(train_history) + 1))
    
    return {
        'train_history': train_history,
        'val_history': val_history,
        'epochs': epochs,
        'model': exp['model'],
        'loss': exp['loss'],
        'optimizer': exp['optimizer'],
        'dataset': experiment_dict['dataset']
    }

