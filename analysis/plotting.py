"""Functions for creating various plots from experiment results."""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.patches as mpatches

from analysis.load_data import extract_experiment_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_training_curves(train_history: List[Dict], 
                        val_history: List[Dict],
                        epochs: List[int],
                        metrics: List[str],
                        save_path: Optional[Path] = None,
                        title: str = "Training Curves",
                        show: bool = True):
    """
    Plot training and validation curves for multiple metrics over epochs.
    
    Args:
        train_history: List of dicts with training metrics per epoch
        val_history: List of dicts with validation metrics per epoch
        epochs: List of epoch numbers
        metrics: List of metric names to plot (e.g., ['loss', 'dice', 'iou'])
        save_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Extract metric values
        train_values = [h.get(metric, 0) for h in train_history]
        val_values = [h.get(metric, 0) for h in val_history]
        
        # Plot
        ax.plot(epochs, train_values, label='Train', marker='o', markersize=3, linewidth=2)
        ax.plot(epochs, val_values, label='Validation', marker='s', markersize=3, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_comparison(df: pd.DataFrame,
                          metric: str,
                          group_by: str = 'model',
                          save_path: Optional[Path] = None,
                          title: Optional[str] = None,
                          show: bool = True):
    """
    Create line/scatter plot comparing a metric across different configurations.
    
    Args:
        df: DataFrame with experiment results
        metric: Metric name to compare (e.g., 'best_val_dice')
        group_by: How to group experiments ('model', 'loss', 'optimizer', or 'dataset')
        save_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group data
    grouped = df.groupby(group_by)[metric].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False)
    
    # Use line plot with error bars
    x_pos = np.arange(len(grouped))
    ax.errorbar(x_pos, grouped['mean'], yerr=grouped['std'], 
               marker='o', markersize=10, linewidth=2.5, capsize=5, capthick=2,
               color='steelblue', markerfacecolor='white', markeredgecolor='steelblue', 
               markeredgewidth=2, elinewidth=1.5)
    
    # Add scatter points for individual experiments
    for idx, (_, row) in enumerate(grouped.iterrows()):
        param_val = row[group_by]
        param_data = df[df[group_by] == param_val][metric].values
        ax.scatter([idx] * len(param_data), param_data, 
                  alpha=0.4, s=50, color='steelblue', zorder=3)
    
    # Add value labels
    for i, (x, mean_val) in enumerate(zip(x_pos, grouped['mean'])):
        ax.text(x, mean_val, f' {mean_val:.4f}',
               ha='left', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel(group_by.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped[group_by], rotation=45, ha='right')
    ax.set_title(title or f'{metric.replace("_", " ").title()} Comparison by {group_by.capitalize()}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_heatmap_comparison(df: pd.DataFrame,
                           metric: str,
                           row_var: str = 'model',
                           col_var: str = 'loss',
                           save_path: Optional[Path] = None,
                           title: Optional[str] = None,
                           show: bool = True):
    """
    Create heatmap comparing metric across two categorical variables.
    
    Args:
        df: DataFrame with experiment results
        metric: Metric name to compare
        row_var: Variable for rows (e.g., 'model')
        col_var: Variable for columns (e.g., 'loss')
        save_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    # Pivot table
    pivot = df.pivot_table(values=metric, index=row_var, columns=col_var, aggfunc='mean')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': metric.replace('_', ' ').title()},
                ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel(col_var.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel(row_var.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title(title or f'{metric.replace("_", " ").title()} Heatmap', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_metrics_comparison(df: pd.DataFrame,
                               metrics: List[str],
                               group_by: str = 'config',
                               save_path: Optional[Path] = None,
                               title: str = "All Metrics Comparison",
                               show: bool = True):
    """
    Create line plot comparing multiple metrics.
    
    Args:
        df: DataFrame with experiment results
        metrics: List of metric names to compare
        group_by: How to group experiments
        save_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    grouped = df.groupby(group_by)[metrics].mean()
    x_pos = np.arange(len(grouped))
    
    # Create line plots for each metric
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    for i, metric in enumerate(metrics):
        ax.plot(x_pos, grouped[metric], marker='o', markersize=8, linewidth=2.5,
               label=metric.replace('_', ' ').replace('best val ', '').title(), 
               color=colors[i], markerfacecolor='white', markeredgecolor=colors[i], 
               markeredgewidth=2)
    
    ax.set_xlabel(group_by.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index, rotation=45, ha='right')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curves_all_experiments(experiment_dict: Dict[str, Any],
                                        metric: str = 'dice',
                                        save_path: Optional[Path] = None,
                                        title: Optional[str] = None,
                                        show: bool = True):
    """
    Plot learning curves for all experiments in a dataset, grouped by configuration.
    
    Args:
        experiment_dict: Experiment data dictionary
        metric: Metric to plot
        save_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    experiments = experiment_dict['experiments']
    dataset_name = experiment_dict['dataset']
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, exp in enumerate(experiments):
        results = exp['results']
        train_history = results.get('train_history', [])
        val_history = results.get('val_history', [])
        
        if len(train_history) == 0:
            continue
        
        epochs = list(range(1, len(train_history) + 1))
        train_values = [h.get(metric, 0) for h in train_history]
        val_values = [h.get(metric, 0) for h in val_history]
        
        config_label = f"{exp['model']} + {exp['loss']} + {exp['optimizer']}"
        
        ax.plot(epochs, train_values, label=f'{config_label} (Train)', 
               color=colors[idx], linestyle='-', linewidth=2, alpha=0.7)
        ax.plot(epochs, val_values, label=f'{config_label} (Val)', 
               color=colors[idx], linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title(title or f'{metric.capitalize()} Learning Curves - {dataset_name}', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning curves to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_by_ablation_parameter(df: pd.DataFrame,
                                     metric: str,
                                     ablation_param: str,
                                     save_path: Optional[Path] = None,
                                     dataset_name: str = "",
                                     show: bool = True):
    """
    Plot a single metric grouped by an ablation parameter using line/scatter plot.
    
    Args:
        df: DataFrame with experiment results
        metric: Metric name to plot (e.g., 'best_val_dice', 'best_val_sensitivity')
        ablation_param: Parameter to group by ('model', 'loss', or 'optimizer')
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby(ablation_param)[metric].mean().sort_values(ascending=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))
    
    # Use line plot with markers instead of bars
    x_pos = range(len(grouped))
    ax.plot(x_pos, grouped.values, marker='o', markersize=10, linewidth=2.5, 
            color='steelblue', markerfacecolor='white', markeredgecolor='steelblue', 
            markeredgewidth=2, label='Mean')
    
    # Add scatter points for individual experiments
    for param_val in grouped.index:
        param_data = df[df[ablation_param] == param_val][metric].values
        x_idx = list(grouped.index).index(param_val)
        ax.scatter([x_idx] * len(param_data), param_data, 
                  alpha=0.4, s=50, color=colors[x_idx], zorder=3)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel(metric.replace('_', ' ').replace('best val ', '').title(), fontsize=12, fontweight='bold')
    ax.set_xlabel(ablation_param.capitalize(), fontsize=12, fontweight='bold')
    
    metric_clean = metric.replace('best_val_', '').replace('_', ' ').title()
    title = f'{metric_clean} by {ablation_param.capitalize()}'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on points
    for i, (x, val) in enumerate(zip(x_pos, grouped.values)):
        ax.text(x, val, f' {val:.4f}',
               ha='left', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_clean} by {ablation_param} plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dataset_metric_comparison(exp_dict: Dict[str, Any],
                                  metric: str = 'best_val_dice',
                                  save_path: Optional[Path] = None,
                                  show: bool = True):
    """
    Plot a single metric comparison for one dataset using line/scatter plot.
    
    Args:
        exp_dict: Experiment dictionary for one dataset
        metric: Metric to compare
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    df = extract_experiment_data(exp_dict)
    dataset_name = exp_dict['dataset']
    
    # Sort by metric value
    df_sorted = df.sort_values(metric, ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) * 0.4)))
    
    y_pos = range(len(df_sorted))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    
    # Use line plot with markers
    ax.plot(df_sorted[metric].values, y_pos, marker='o', markersize=8, 
            linewidth=2, color='steelblue', markerfacecolor='white', 
            markeredgecolor='steelblue', markeredgewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['config'].values, fontsize=9)
    ax.set_xlabel(metric.replace('_', ' ').replace('best val ', '').title(), 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_name} - {metric.replace("_", " ").replace("best val ", "").title()} by Configuration', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (y, val) in enumerate(zip(y_pos, df_sorted[metric].values)):
        ax.text(val, y, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        metric_clean = metric.replace('_', ' ').replace('best val ', '').title()
        print(f"Saved {dataset_name} {metric_clean} comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_focal_gamma_losses_from_json(
    json_path: Path,
    model: Optional[str] = None,
    optimizer: Optional[str] = None,
    bins: int = 50,
    show: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot empirical Focal Loss vs. probability p_t for different gamma values (one line per gamma).

    Uses the per-pixel samples stored as 'probability_samples' and 'focal_loss_samples'
    in the validation history (collected when using FocalLoss).

    Args:
        json_path: Path to the dataset-level experiment JSON (e.g. outputs/DRIVE_GAMMA_STUDY.json).
        model: Optional model name to filter by (e.g. 'Unet'). If None, use all models.
        optimizer: Optional optimizer name to filter by (e.g. 'adamW'). If None, use all.
        bins: Number of probability bins between 0 and 1 for averaging the loss.
        show: Whether to display the plot.
        save_path: Optional path to save the figure.
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        exp_dict = json.load(f)

    experiments = exp_dict.get("experiments", [])
    if not experiments:
        print("No experiments found in JSON; nothing to plot.")
        return

    # Collect bin-wise data per gamma (each experiment has complete bin data)
    gamma_to_data: Dict[int, Dict[str, np.ndarray]] = {}

    for exp in experiments:
        loss_name = exp.get("loss", "")
        if not str(loss_name).lower().startswith("focal"):
            continue

        if model is not None and exp.get("model", "").lower() != model.lower():
            continue

        if optimizer is not None and exp.get("optimizer", "").lower() != optimizer.lower():
            continue

        # Parse gamma from loss name (e.g. 'Focal1', 'Focal_3')
        loss_str = str(loss_name)
        digits = "".join(ch for ch in loss_str if ch.isdigit())
        gamma = int(digits) if digits else 2

        results = exp.get("results", {})
        val_history = results.get("val_history", [])
        if not val_history:
            continue

        # Use the last validation epoch as representative
        last_val = val_history[-1]
        
        # Check for new bin-wise format (preferred, exact averages)
        bin_centers = last_val.get("focal_bin_centers")
        loss_by_bin = last_val.get("focal_loss_by_bin")
        
        if bin_centers is not None and loss_by_bin is not None:
            # New format: directly use pre-computed bin-wise averages
            p_arr = np.asarray(bin_centers, dtype=np.float32)
            l_arr = np.asarray(loss_by_bin, dtype=np.float32)
            
            # Filter out NaN values (empty bins)
            valid = ~np.isnan(l_arr)
            p_arr = p_arr[valid]
            l_arr = l_arr[valid]
            
            if p_arr.size > 0:
                gamma_to_data[gamma] = {"probs": p_arr, "losses": l_arr}
            continue
        
        # Fallback: check for old subsampled format (for backwards compatibility)
        probs = last_val.get("probability_samples")
        losses = last_val.get("focal_loss_samples")
        
        if probs is not None and losses is not None:
            # Old format: need to bin the subsampled data
            p_arr = np.asarray(probs, dtype=np.float32)
            l_arr = np.asarray(losses, dtype=np.float32)

            # Filter to valid [0,1] range
            mask = (p_arr >= 0.0) & (p_arr <= 1.0)
            p_arr = p_arr[mask]
            l_arr = l_arr[mask]

            if p_arr.size > 0:
                # Bin the data
                bin_edges = np.linspace(0.0, 1.0, bins + 1)
                bin_centers_old = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                
                indices = np.digitize(p_arr, bin_edges) - 1
                indices = np.clip(indices, 0, bins - 1)
                
                mean_loss = np.zeros(bins, dtype=np.float32)
                counts = np.zeros(bins, dtype=np.int64)
                
                for i, loss_val in zip(indices, l_arr):
                    mean_loss[i] += loss_val
                    counts[i] += 1
                
                valid = counts > 0
                mean_loss[valid] /= counts[valid]
                mean_loss[~valid] = np.nan
                
                valid_bins = ~np.isnan(mean_loss)
                gamma_to_data[gamma] = {
                    "probs": bin_centers_old[valid_bins],
                    "losses": mean_loss[valid_bins]
                }

    if not gamma_to_data:
        print("No probability/loss data found for FocalLoss; did you run with the updated Trainer?")
        print("Expected fields: 'focal_bin_centers' and 'focal_loss_by_bin' in validation history")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for gamma in sorted(gamma_to_data.keys()):
        data = gamma_to_data[gamma]
        p_arr = data["probs"]
        l_arr = data["losses"]
        
        ax.plot(
            p_arr,
            l_arr,
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"$\\gamma={gamma}$",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Probability $p_t$", fontsize=12, fontweight="bold")
    ax.set_ylabel("Focal loss", fontsize=12, fontweight="bold")
    title_model = f" ({model})" if model is not None else ""
    dataset_name = exp_dict.get("dataset", "")
    dataset_suffix = f" - {dataset_name}" if dataset_name else ""
    ax.set_title(f"Empirical Focal Loss vs. Probability{title_model}{dataset_suffix}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Focal gamma")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved focal loss vs probability plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

