"""Advanced visualization functions that leverage per-epoch data and all experiments."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from analysis.load_data import extract_experiment_data

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_parameter_interaction_surface(df: pd.DataFrame,
                                      metric: str,
                                      param1: str,
                                      param2: str,
                                      save_path: Optional[Path] = None,
                                      dataset_name: str = "",
                                      show: bool = True):
    """
    Create 2D heatmap showing metric performance across parameter interactions.
    More detailed than basic heatmap - shows all combinations.
    
    Args:
        df: DataFrame with experiment results
        metric: Metric to visualize
        param1: First parameter (e.g., 'model', 'loss')
        param2: Second parameter (e.g., 'optimizer', 'loss')
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table
    pivot = df.pivot_table(values=metric, index=param1, columns=param2, aggfunc='mean')
    
    # Create heatmap with annotations
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=pivot.values.min(), vmax=pivot.values.max())
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    metric_clean = metric.replace('_', ' ').replace('best val ', '').title()
    cbar.set_label(metric_clean, fontsize=11, fontweight='bold')
    
    ax.set_xlabel(param2.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel(param1.capitalize(), fontsize=12, fontweight='bold')
    title = f'{metric_clean}: {param1.capitalize()} × {param2.capitalize()} Interaction'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter interaction surface to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence_analysis(experiment_dict: Dict[str, Any],
                             metric: str = 'dice',
                             save_path: Optional[Path] = None,
                             dataset_name: str = "",
                             show: bool = True):
    """
    Analyze convergence speed and stability across all experiments.
    Shows: convergence epoch, final value, stability (variance in last N epochs).
    
    Args:
        experiment_dict: Experiment data dictionary
        metric: Metric to analyze
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    experiments = experiment_dict['experiments']
    dataset_name = dataset_name or experiment_dict.get('dataset', '')
    
    convergence_data = []
    for exp in experiments:
        results = exp['results']
        val_history = results.get('val_history', [])
        
        if len(val_history) < 5:
            continue
        
        # Get metric values over epochs
        metric_values = [h.get(metric, 0) for h in val_history]
        
        # Find convergence epoch (when metric stabilizes within 1% for 5 epochs)
        converged_epoch = len(metric_values)
        for i in range(5, len(metric_values)):
            recent_vals = metric_values[i-5:i]
            if max(recent_vals) - min(recent_vals) < 0.01 * max(recent_vals):
                converged_epoch = i
                break
        
        # Calculate stability (std of last 10 epochs)
        last_n = min(10, len(metric_values))
        stability = np.std(metric_values[-last_n:]) if last_n > 1 else 0
        
        # Final value
        final_value = metric_values[-1]
        
        convergence_data.append({
            'config': f"{exp['model']} + {exp['loss']} + {exp['optimizer']}",
            'converged_epoch': converged_epoch,
            'final_value': final_value,
            'stability': stability,
            'max_value': max(metric_values)
        })
    
    if not convergence_data:
        return
    
    df_conv = pd.DataFrame(convergence_data)
    df_conv = df_conv.sort_values('final_value', ascending=False)
    
    # Create singular plot: Convergence epoch vs final value
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(df_conv['converged_epoch'], df_conv['final_value'], 
                        s=100, c=df_conv['stability'], cmap='viridis_r',
                        edgecolors='black', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Convergence Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.capitalize()} (Final Value)', fontsize=12, fontweight='bold')
    title = f'Convergence Speed vs Final Performance ({metric.capitalize()})'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Stability (Lower = More Stable)', fontsize=10)
    
    # Add config labels for top performers
    top_n = min(3, len(df_conv))
    for idx in range(top_n):
        row = df_conv.iloc[idx]
        ax.annotate(row['config'].split(' + ')[0], 
                   (row['converged_epoch'], row['final_value']),
                   fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence analysis to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_correlation_matrix(df: pd.DataFrame,
                                   save_path: Optional[Path] = None,
                                   dataset_name: str = "",
                                   show: bool = True):
    """
    Create correlation matrix showing relationships between different metrics.
    
    Args:
        df: DataFrame with experiment results
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    metrics = ['best_val_dice', 'best_val_iou', 'best_val_accuracy', 
              'best_val_sensitivity', 'best_val_specificity', 'best_val_loss']
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        return
    
    corr_matrix = df[available_metrics].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    metric_labels = [m.replace('best_val_', '').replace('_', ' ').title() for m in available_metrics]
    ax.set_xticks(np.arange(len(available_metrics)))
    ax.set_yticks(np.arange(len(available_metrics)))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_yticklabels(metric_labels)
    
    # Add text annotations
    for i in range(len(available_metrics)):
        for j in range(len(available_metrics)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            text = ax.text(j, i, f'{val:.2f}',
                         ha="center", va="center", color=color, fontweight='bold', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')
    
    title = 'Metric Correlation Matrix'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_dynamics_heatmap(experiment_dict: Dict[str, Any],
                                  metric: str = 'dice',
                                  save_path: Optional[Path] = None,
                                  dataset_name: str = "",
                                  show: bool = True):
    """
    Create heatmap showing metric evolution over epochs for all experiments.
    Each row is an experiment, columns are epochs.
    
    Args:
        experiment_dict: Experiment data dictionary
        metric: Metric to visualize
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    experiments = experiment_dict['experiments']
    dataset_name = dataset_name or experiment_dict.get('dataset', '')
    
    # Collect all validation histories
    data_matrix = []
    config_labels = []
    max_epochs = 0
    
    for exp in experiments:
        results = exp['results']
        val_history = results.get('val_history', [])
        
        if len(val_history) == 0:
            continue
        
        metric_values = [h.get(metric, 0) for h in val_history]
        data_matrix.append(metric_values)
        config_labels.append(f"{exp['model'][:4]} + {exp['loss'][:6]} + {exp['optimizer']}")
        max_epochs = max(max_epochs, len(metric_values))
    
    if not data_matrix:
        return
    
    # Pad all rows to same length
    for row in data_matrix:
        while len(row) < max_epochs:
            row.append(row[-1] if row else 0)  # Extend with last value
    
    data_array = np.array(data_matrix)
    
    # Sort by final performance
    final_perf = data_array[:, -1]
    sort_idx = np.argsort(final_perf)[::-1]
    data_array = data_array[sort_idx]
    config_labels = [config_labels[i] for i in sort_idx]
    
    fig, ax = plt.subplots(figsize=(max(12, max_epochs * 0.3), max(8, len(data_array) * 0.4)))
    
    im = ax.imshow(data_array, cmap='YlOrRd', aspect='auto', 
                   vmin=data_array.min(), vmax=data_array.max())
    
    # Set ticks
    ax.set_yticks(np.arange(len(config_labels)))
    ax.set_yticklabels(config_labels, fontsize=8)
    
    # Show every 5th epoch or all if < 20
    if max_epochs <= 20:
        epoch_ticks = np.arange(max_epochs)
        epoch_labels = [str(i+1) for i in epoch_ticks]
    else:
        epoch_ticks = np.arange(0, max_epochs, max(1, max_epochs // 20))
        epoch_labels = [str(i+1) for i in epoch_ticks]
    
    ax.set_xticks(epoch_ticks)
    ax.set_xticklabels(epoch_labels, fontsize=8)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    metric_clean = metric.replace('_', ' ').title()
    cbar.set_label(f'Validation {metric_clean}', fontsize=11, fontweight='bold')
    
    title = f'Training Dynamics: {metric_clean} Evolution Over Epochs'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training dynamics heatmap to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_overfitting_analysis(experiment_dict: Dict[str, Any],
                             metric: str = 'dice',
                             save_path: Optional[Path] = None,
                             dataset_name: str = "",
                             show: bool = True):
    """
    Analyze overfitting by showing train-val gap over epochs for all experiments.
    
    Args:
        experiment_dict: Experiment data dictionary
        metric: Metric to analyze
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    experiments = experiment_dict['experiments']
    dataset_name = dataset_name or experiment_dict.get('dataset', '')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, exp in enumerate(experiments):
        results = exp['results']
        train_history = results.get('train_history', [])
        val_history = results.get('val_history', [])
        
        if len(train_history) == 0 or len(val_history) == 0:
            continue
        
        epochs = list(range(1, len(train_history) + 1))
        train_values = [h.get(metric, 0) for h in train_history]
        val_values = [h.get(metric, 0) for h in val_history]
        
        # Calculate gap (train - val)
        gap = [t - v for t, v in zip(train_values, val_values)]
        
        config_label = f"{exp['model']} + {exp['loss']} + {exp['optimizer']}"
        ax.plot(epochs, gap, label=config_label, linewidth=2, alpha=0.7, color=colors[idx])
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Train - Val {metric.capitalize()} Gap', fontsize=12, fontweight='bold')
    title = f'Overfitting Analysis: Train-Val Gap ({metric.capitalize()})'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overfitting analysis to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_performance_distribution(df: pd.DataFrame,
                                 metric: str,
                                 group_by: str = 'config',
                                 save_path: Optional[Path] = None,
                                 dataset_name: str = "",
                                 show: bool = True):
    """
    Show distribution of metric values across experiments using violin plots.
    More informative than box plots - shows full distribution shape.
    
    Args:
        df: DataFrame with experiment results
        metric: Metric to visualize
        save_path: Optional path to save the figure
        dataset_name: Dataset name for title
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for violin plot
    groups = df[group_by].unique()
    data_by_group = [df[df[group_by] == g][metric].values for g in groups]
    
    # Create violin plot
    parts = ax.violinplot(data_by_group, positions=range(len(groups)), 
                         showmeans=True, showmedians=True)
    
    # Customize violins
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric.replace('_', ' ').replace('best val ', '').title(), 
                  fontsize=12, fontweight='bold')
    ax.set_xlabel(group_by.capitalize(), fontsize=12, fontweight='bold')
    
    metric_clean = metric.replace('_', ' ').replace('best val ', '').title()
    title = f'{metric_clean} Distribution by {group_by.capitalize()}'
    if dataset_name:
        title = f'{dataset_name} - {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance distribution to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

