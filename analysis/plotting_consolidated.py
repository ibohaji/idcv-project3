"""Consolidated, multi-dimensional plots for scientific reports."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from analysis.load_data import extract_experiment_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def plot_all_metrics_by_model(df: pd.DataFrame,
                              save_path: Optional[Path] = None,
                              dataset_name: str = "",
                              show: bool = True):
    """
    Single plot showing all key metrics (dice, iou, sensitivity, specificity) grouped by model.
    Uses grouped bars or line plot to show multiple metrics together.
    
    Args:
        df: DataFrame with experiment results
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    metrics = ['best_val_dice', 'best_val_iou', 'best_val_sensitivity', 'best_val_specificity']
    metric_labels = ['Dice', 'IoU', 'Sensitivity', 'Specificity']
    
    # Group by model and calculate mean for each metric
    grouped = df.groupby('model')[metrics].mean()
    models = grouped.index
    x_pos = np.arange(len(models))
    width = 0.2
    
    # Create grouped bars
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - len(metrics)/2) * width + width/2
        bars = ax.bar(x_pos + offset, grouped[metric], width, 
                     label=label, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, grouped[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    title = 'All Metrics by Model'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved all metrics by model to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_by_optimizer_and_loss(df: pd.DataFrame,
                                       save_path: Optional[Path] = None,
                                       dataset_name: str = "",
                                       show: bool = True):
    """
    Single plot showing dice and sensitivity grouped by optimizer, with different lines for each loss.
    Shows interaction between optimizer and loss.
    
    Args:
        df: DataFrame with experiment results
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    optimizers = sorted(df['optimizer'].unique())
    losses = sorted(df['loss'].unique())
    x_pos = np.arange(len(optimizers))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(losses)))
    
    # Plot Dice by optimizer, different lines for each loss (left y-axis)
    for loss, color in zip(losses, colors):
        loss_data = df[df['loss'] == loss]
        dice_by_opt = loss_data.groupby('optimizer')['best_val_dice'].mean()
        dice_values = [dice_by_opt.get(opt, 0) for opt in optimizers]
        ax.plot(x_pos, dice_values, marker='o', markersize=10, linewidth=2.5,
               label=f'Dice: {loss}', color=color, markerfacecolor='white', 
               markeredgecolor=color, markeredgewidth=2)
    
    ax.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold', color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(optimizers, fontsize=11)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.grid(True, alpha=0.3)
    
    # Plot Sensitivity on right y-axis
    ax2 = ax.twinx()
    for loss, color in zip(losses, colors):
        loss_data = df[df['loss'] == loss]
        sens_by_opt = loss_data.groupby('optimizer')['best_val_sensitivity'].mean()
        sens_values = [sens_by_opt.get(opt, 0) for opt in optimizers]
        ax2.plot(x_pos, sens_values, marker='s', markersize=10, linewidth=2.5,
                label=f'Sensitivity: {loss}', color=color, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=2, linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('Sensitivity', fontsize=12, fontweight='bold', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, ncol=2)
    
    title = 'Metrics by Optimizer and Loss Function'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics by optimizer and loss to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_best_config_per_metric(df: pd.DataFrame,
                                save_path: Optional[Path] = None,
                                dataset_name: str = "",
                                show: bool = True):
    """
    Single plot showing which configuration (model+loss+optimizer) performs best for each metric.
    Uses horizontal bars with metric on y-axis, value on x-axis, colored by model.
    
    Args:
        df: DataFrame with experiment results
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    metrics = ['best_val_dice', 'best_val_iou', 'best_val_sensitivity', 'best_val_specificity']
    metric_labels = ['Dice', 'IoU', 'Sensitivity', 'Specificity']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    y_positions = []
    bar_values = []
    bar_colors = []
    bar_labels = []
    
    # For each metric, find best config and its value
    for metric, label in zip(metrics, metric_labels):
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        best_value = best_row[metric]
        best_config = f"{best_row['model']} + {best_row['loss']} + {best_row['optimizer']}"
        
        y_positions.append(len(metric_labels) - metrics.index(metric) - 1)
        bar_values.append(best_value)
        bar_colors.append('steelblue' if 'Unet' in best_row['model'] else 'coral')
        bar_labels.append(best_config)
    
    # Create horizontal bars
    bars = ax.barh(y_positions, bar_values, color=bar_colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val, label) in enumerate(zip(bars, bar_values, bar_labels)):
        ax.text(val, bar.get_y() + bar.get_height()/2, 
               f' {val:.4f} ({label})', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(metric_labels, fontsize=12, fontweight='bold')
    ax.set_xlabel('Best Metric Value', fontsize=12, fontweight='bold')
    title = 'Best Configuration per Metric'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend for model colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Unet'),
        Patch(facecolor='coral', label='EncoderDecoder')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved best config per metric to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves_comparison(experiment_dict: Dict[str, Any],
                                   metrics: List[str] = ['dice', 'iou', 'loss'],
                                   save_path: Optional[Path] = None,
                                   dataset_name: str = "",
                                   show: bool = True):
    """
    Single plot showing training curves for best configuration across multiple metrics.
    Shows train and val curves for dice, iou, and loss together.
    
    Args:
        experiment_dict: Experiment data dictionary
        metrics: List of metrics to plot
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    # Find best configuration (by dice)
    df = extract_experiment_data(experiment_dict)
    best_idx = df['best_val_dice'].idxmax()
    best_config = df.loc[best_idx]
    
    # Find matching experiment
    best_exp = None
    for exp in experiment_dict['experiments']:
        if (exp['model'] == best_config['model'] and 
            exp['loss'] == best_config['loss'] and 
            exp['optimizer'] == best_config['optimizer']):
            best_exp = exp
            break
    
    if not best_exp:
        return
    
    results = best_exp['results']
    train_history = results.get('train_history', [])
    val_history = results.get('val_history', [])
    epochs = list(range(1, len(train_history) + 1))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot dice on left y-axis
    train_dice = [h.get('dice', 0) for h in train_history]
    val_dice = [h.get('dice', 0) for h in val_history]
    
    ax.plot(epochs, train_dice, label='Dice (Train)', color='steelblue', 
           linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    ax.plot(epochs, val_dice, label='Dice (Val)', color='steelblue', 
           linewidth=2.5, marker='s', markersize=5, alpha=0.8, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.grid(True, alpha=0.3)
    
    # Plot loss on right y-axis
    ax2 = ax.twinx()
    train_loss = [h.get('loss', 0) for h in train_history]
    val_loss = [h.get('loss', 0) for h in val_history]
    
    ax2.plot(epochs, train_loss, label='Loss (Train)', color='coral', 
            linewidth=2.5, marker='^', markersize=5, alpha=0.8)
    ax2.plot(epochs, val_loss, label='Loss (Val)', color='coral', 
            linewidth=2.5, marker='v', markersize=5, alpha=0.8, linestyle='--')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    config_label = f"{best_config['model']} + {best_config['loss']} + {best_config['optimizer']}"
    title = f'Training Curves - Best Configuration\n{config_label}'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ablation_summary_consolidated(df: pd.DataFrame,
                                      save_path: Optional[Path] = None,
                                      dataset_name: str = "",
                                      show: bool = True):
    """
    Single comprehensive plot showing ablation study results.
    Shows top configurations with all metrics, grouped by key parameters.
    
    Args:
        df: DataFrame with experiment results
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Show top configurations with all metrics
    metrics_to_plot = ['best_val_dice', 'best_val_iou', 'best_val_sensitivity', 'best_val_specificity']
    metric_labels = ['Dice', 'IoU', 'Sensitivity', 'Specificity']
    
    # Group by config and calculate mean, sort by dice
    grouped = df.groupby('config')[metrics_to_plot].mean().sort_values('best_val_dice', ascending=False)
    top_configs = grouped.head(10)  # Show top 10 configs
    
    x_pos = np.arange(len(top_configs))
    width = 0.2
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_to_plot)))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        offset = (i - len(metrics_to_plot)/2) * width + width/2
        bars = ax.bar(x_pos + offset, top_configs[metric], width,
                     label=label, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on top
        for bar, val in zip(bars, top_configs[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_xlabel('Configuration (Top 10 by Dice)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([cfg[:30] + '...' if len(cfg) > 30 else cfg for cfg in top_configs.index],
                      rotation=45, ha='right', fontsize=9)
    title = 'Ablation Study Summary: Top Configurations'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved consolidated ablation summary to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_key_metrics_over_epochs(experiment_dict: Dict[str, Any],
                                 top_n: int = 5,
                                 save_path: Optional[Path] = None,
                                 dataset_name: str = "",
                                 show: bool = True):
    """
    Plot dice, sensitivity, and specificity over epochs for top N configurations.
    Shows training dynamics - what a scientist would look for: convergence, stability, trade-offs.
    
    Args:
        experiment_dict: Experiment data dictionary
        top_n: Number of top configurations to show (by dice)
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    df = extract_experiment_data(experiment_dict)
    
    # Get top N configurations by dice
    top_configs = df.nlargest(top_n, 'best_val_dice')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['dice', 'sensitivity', 'specificity']
    metric_labels = ['Dice Score', 'Sensitivity (Recall)', 'Specificity']
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        for idx, (_, row) in enumerate(top_configs.iterrows()):
            # Find matching experiment
            exp = None
            for e in experiment_dict['experiments']:
                if (e['model'] == row['model'] and 
                    e['loss'] == row['loss'] and 
                    e['optimizer'] == row['optimizer']):
                    exp = e
                    break
            
            if not exp:
                continue
            
            results = exp['results']
            val_history = results.get('val_history', [])
            if len(val_history) == 0:
                continue
            
            epochs = list(range(1, len(val_history) + 1))
            values = [h.get(metric, 0) for h in val_history]
            
            config_label = f"{row['model'][:4]}+{row['loss'][:6]}+{row['optimizer']}"
            ax.plot(epochs, values, label=config_label, linewidth=2.5, 
                   color=colors[idx], alpha=0.8, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Over Training', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    title = f'Key Metrics Evolution: Top {top_n} Configurations'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved key metrics over epochs to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_tradeoffs(experiment_dict: Dict[str, Any],
                         save_path: Optional[Path] = None,
                         dataset_name: str = "",
                         show: bool = True):
    """
    Plot metric trade-offs: Dice vs Sensitivity, Dice vs Specificity.
    Shows what scientists care about: are we sacrificing one metric for another?
    
    Args:
        experiment_dict: Experiment data dictionary
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    df = extract_experiment_data(experiment_dict)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color by model
    models = df['model'].unique()
    model_colors = {'Unet': 'steelblue', 'EncoderDecoder': 'coral'}
    
    # Plot 1: Dice vs Sensitivity
    for model in models:
        model_data = df[df['model'] == model]
        ax1.scatter(model_data['best_val_dice'], model_data['best_val_sensitivity'],
                   s=100, alpha=0.7, label=model, color=model_colors.get(model, 'gray'),
                   edgecolors='black', linewidth=1.5)
        
        # Add config labels for top performers
        top_3 = model_data.nlargest(3, 'best_val_dice')
        for _, row in top_3.iterrows():
            label = f"{row['loss'][:6]}+{row['optimizer']}"
            ax1.annotate(label, (row['best_val_dice'], row['best_val_sensitivity']),
                        fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Dice Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sensitivity (Recall)', fontsize=12, fontweight='bold')
    ax1.set_title('Dice vs Sensitivity Trade-off', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Dice vs Specificity
    for model in models:
        model_data = df[df['model'] == model]
        ax2.scatter(model_data['best_val_dice'], model_data['best_val_specificity'],
                   s=100, alpha=0.7, label=model, color=model_colors.get(model, 'gray'),
                   edgecolors='black', linewidth=1.5, marker='s')
        
        # Add config labels for top performers
        top_3 = model_data.nlargest(3, 'best_val_dice')
        for _, row in top_3.iterrows():
            label = f"{row['loss'][:6]}+{row['optimizer']}"
            ax2.annotate(label, (row['best_val_dice'], row['best_val_specificity']),
                        fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Dice Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Specificity', fontsize=12, fontweight='bold')
    ax2.set_title('Dice vs Specificity Trade-off', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    title = 'Metric Trade-off Analysis'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metric tradeoffs to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comprehensive_metrics_comparison(df: pd.DataFrame,
                                         save_path: Optional[Path] = None,
                                         dataset_name: str = "",
                                         show: bool = True):
    """
    Single comprehensive plot showing dice, sensitivity, specificity across all configurations.
    Easy to read comparison - what scientists need to quickly assess performance.
    
    Args:
        df: DataFrame with experiment results
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by dice score
    df_sorted = df.sort_values('best_val_dice', ascending=True)
    
    y_pos = np.arange(len(df_sorted))
    
    # Create grouped horizontal bars
    width = 0.25
    colors = {'dice': 'steelblue', 'sensitivity': 'coral', 'specificity': 'green'}
    
    bars1 = ax.barh(y_pos - width, df_sorted['best_val_dice'], width,
                    label='Dice', color=colors['dice'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.barh(y_pos, df_sorted['best_val_sensitivity'], width,
                    label='Sensitivity', color=colors['sensitivity'], alpha=0.8, 
                    edgecolor='black', linewidth=1)
    bars3 = ax.barh(y_pos + width, df_sorted['best_val_specificity'], width,
                    label='Specificity', color=colors['specificity'], alpha=0.8,
                    edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            width_bar = bar.get_width()
            ax.text(width_bar, bar.get_y() + bar.get_height()/2,
                   f' {width_bar:.3f}', va='center', fontsize=7, fontweight='bold')
    
    # Set y-axis labels
    config_labels = [f"{row['model'][:4]}+{row['loss'][:6]}+{row['optimizer']}" 
                    for _, row in df_sorted.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(config_labels, fontsize=9)
    
    ax.set_xlabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    title = 'Comprehensive Metrics Comparison (Dice, Sensitivity, Specificity)'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive metrics comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_comparison(experiment_dict: Dict[str, Any],
                               metric: str = 'dice',
                               top_n: int = 6,
                               save_path: Optional[Path] = None,
                               dataset_name: str = "",
                               show: bool = True):
    """
    Compare convergence behavior across top configurations.
    Shows when each configuration reaches its peak and how stable it is.
    
    Args:
        experiment_dict: Experiment data dictionary
        metric: Metric to analyze (dice, sensitivity, specificity)
        top_n: Number of top configurations to compare
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    df = extract_experiment_data(experiment_dict)
    top_configs = df.nlargest(top_n, f'best_val_{metric}')
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for idx, (_, row) in enumerate(top_configs.iterrows()):
        # Find matching experiment
        exp = None
        for e in experiment_dict['experiments']:
            if (e['model'] == row['model'] and 
                e['loss'] == row['loss'] and 
                e['optimizer'] == row['optimizer']):
                exp = e
                break
        
        if not exp:
            continue
        
        results = exp['results']
        val_history = results.get('val_history', [])
        if len(val_history) == 0:
            continue
        
        epochs = list(range(1, len(val_history) + 1))
        values = [h.get(metric, 0) for h in val_history]
        
        # Find peak epoch
        peak_epoch = epochs[np.argmax(values)]
        peak_value = max(values)
        
        config_label = f"{row['model'][:4]}+{row['loss'][:6]}+{row['optimizer']}"
        ax.plot(epochs, values, label=config_label, linewidth=2.5, 
               color=colors[idx], alpha=0.8, marker='o', markersize=4)
        
        # Mark peak
        ax.plot(peak_epoch, peak_value, marker='*', markersize=15, 
               color=colors[idx], markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=12, fontweight='bold')
    title = f'{metric.capitalize()} Convergence: Top {top_n} Configurations'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

