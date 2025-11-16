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
    Create bar plot comparing a metric across different configurations.
    
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
    
    # Create bar plot
    x_pos = np.arange(len(grouped))
    bars = ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
                  capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, grouped['mean'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
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
    Create grouped bar chart comparing multiple metrics.
    
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
    width = 0.8 / len(metrics)
    
    # Create bars for each metric
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2) * width + width/2
        bars = ax.bar(x_pos + offset, grouped[metric], width, 
                     label=metric.replace('_', ' ').title(), 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
    
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


def plot_ablation_study_summary(df: pd.DataFrame,
                               save_path: Optional[Path] = None,
                               show: bool = True):
    """
    Create comprehensive ablation study summary with multiple subplots.
    
    Args:
        df: DataFrame with all experiment results
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Best Dice by Model
    ax1 = fig.add_subplot(gs[0, 0])
    model_dice = df.groupby('model')['best_val_dice'].mean().sort_values(ascending=False)
    ax1.bar(range(len(model_dice)), model_dice.values, color='skyblue', edgecolor='black')
    ax1.set_xticks(range(len(model_dice)))
    ax1.set_xticklabels(model_dice.index, rotation=45, ha='right')
    ax1.set_ylabel('Dice Score', fontweight='bold')
    ax1.set_title('Best Dice by Model', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Best Dice by Loss
    ax2 = fig.add_subplot(gs[0, 1])
    loss_dice = df.groupby('loss')['best_val_dice'].mean().sort_values(ascending=False)
    ax2.bar(range(len(loss_dice)), loss_dice.values, color='lightcoral', edgecolor='black')
    ax2.set_xticks(range(len(loss_dice)))
    ax2.set_xticklabels(loss_dice.index, rotation=45, ha='right')
    ax2.set_ylabel('Dice Score', fontweight='bold')
    ax2.set_title('Best Dice by Loss Function', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Best Dice by Optimizer
    ax3 = fig.add_subplot(gs[0, 2])
    opt_dice = df.groupby('optimizer')['best_val_dice'].mean().sort_values(ascending=False)
    ax3.bar(range(len(opt_dice)), opt_dice.values, color='lightgreen', edgecolor='black')
    ax3.set_xticks(range(len(opt_dice)))
    ax3.set_xticklabels(opt_dice.index, rotation=45, ha='right')
    ax3.set_ylabel('Dice Score', fontweight='bold')
    ax3.set_title('Best Dice by Optimizer', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Heatmap: Model x Loss
    ax4 = fig.add_subplot(gs[1, 0])
    pivot1 = df.pivot_table(values='best_val_dice', index='model', columns='loss', aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'Dice'})
    ax4.set_title('Dice: Model × Loss', fontweight='bold')
    
    # 5. Heatmap: Model x Optimizer
    ax5 = fig.add_subplot(gs[1, 1])
    pivot2 = df.pivot_table(values='best_val_dice', index='model', columns='optimizer', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Dice'})
    ax5.set_title('Dice: Model × Optimizer', fontweight='bold')
    
    # 6. Heatmap: Loss x Optimizer
    ax6 = fig.add_subplot(gs[1, 2])
    pivot3 = df.pivot_table(values='best_val_dice', index='loss', columns='optimizer', aggfunc='mean')
    sns.heatmap(pivot3, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Dice'})
    ax6.set_title('Dice: Loss × Optimizer', fontweight='bold')
    
    # 7. All metrics comparison (best validation)
    ax7 = fig.add_subplot(gs[2, :])
    metrics_to_plot = ['best_val_dice', 'best_val_iou', 'best_val_accuracy', 
                      'best_val_sensitivity', 'best_val_specificity']
    df_melted = df.melt(id_vars=['config'], value_vars=metrics_to_plot,
                       var_name='metric', value_name='value')
    df_melted['metric'] = df_melted['metric'].str.replace('best_val_', '').str.capitalize()
    
    sns.boxplot(data=df_melted, x='metric', y='value', ax=ax7, palette='Set2')
    ax7.set_xlabel('Metric', fontweight='bold')
    ax7.set_ylabel('Value', fontweight='bold')
    ax7.set_title('Distribution of Best Validation Metrics', fontweight='bold')
    ax7.tick_params(axis='x', rotation=45)
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Ablation Study Summary', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ablation study summary to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_dataset_comparison(all_experiments: List[Dict[str, Any]],
                           metric: str = 'best_val_dice',
                           save_path: Optional[Path] = None,
                           show: bool = True):
    """
    Compare performance across datasets.
    
    Args:
        all_experiments: List of experiment dictionaries (one per dataset)
        metric: Metric to compare
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, exp_dict in enumerate(all_experiments):
        df = extract_experiment_data(exp_dict)
        dataset_name = exp_dict['dataset']
        
        # Group by configuration
        grouped = df.groupby('config')[metric].mean().sort_values(ascending=False)
        
        ax = axes[idx]
        bars = ax.barh(range(len(grouped)), grouped.values, color=plt.cm.viridis(np.linspace(0, 1, len(grouped))))
        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels(grouped.index, fontsize=9)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontweight='bold')
        ax.set_title(f'{dataset_name} - {metric.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, grouped.values)):
            ax.text(val, bar.get_y() + bar.get_height()/2, 
                   f' {val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Dataset Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dataset comparison to {save_path}")
    
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

    # Collect samples per gamma
    gamma_to_probs: Dict[int, List[float]] = {}
    gamma_to_losses: Dict[int, List[float]] = {}

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
        probs = last_val.get("probability_samples")
        losses = last_val.get("focal_loss_samples")

        if probs is None or losses is None:
            continue

        if gamma not in gamma_to_probs:
            gamma_to_probs[gamma] = []
            gamma_to_losses[gamma] = []

        gamma_to_probs[gamma].extend(probs)
        gamma_to_losses[gamma].extend(losses)

    if not gamma_to_probs:
        print("No probability/loss samples found for FocalLoss; did you run with the updated Trainer?")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for gamma, probs_list in sorted(gamma_to_probs.items()):
        losses_list = gamma_to_losses[gamma]
        p_arr = np.asarray(probs_list, dtype=np.float32)
        l_arr = np.asarray(losses_list, dtype=np.float32)

        # Filter to valid [0,1] range just in case
        mask = (p_arr >= 0.0) & (p_arr <= 1.0)
        p_arr = p_arr[mask]
        l_arr = l_arr[mask]

        if p_arr.size == 0:
            continue

        # Digitize probabilities into bins and compute mean loss per bin
        indices = np.digitize(p_arr, bin_edges) - 1  # 0..bins-1
        mean_loss = np.zeros(bins, dtype=np.float32)
        counts = np.zeros(bins, dtype=np.int64)

        for i, loss_val in zip(indices, l_arr):
            if 0 <= i < bins:
                mean_loss[i] += loss_val
                counts[i] += 1

        # Avoid division by zero
        valid = counts > 0
        mean_loss[valid] /= counts[valid]
        mean_loss[~valid] = np.nan

        ax.plot(
            bin_centers,
            mean_loss,
            marker="",
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

