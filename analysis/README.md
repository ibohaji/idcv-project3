# Experiment Analysis Module

This module provides comprehensive analysis and visualization tools for experiment results.

## Structure

- `load_data.py`: Functions to load and parse experiment JSON files
- `plotting.py`: Functions to create various plots (training curves, comparisons, heatmaps, etc.)
- `main.py`: Main script to generate all analysis plots

## Usage

### Basic Usage

Run the main script to generate all plots from experiment JSON files:

```bash
python analysis/main.py
```

This will:
1. Load all JSON files from `./outputs/` directory
2. Generate comprehensive plots for each dataset
3. Save all plots to `./analysis/plots/`

### Command Line Options

```bash
python analysis/main.py --output-dir ./outputs --save-dir ./analysis/plots --show
```

- `--output-dir`: Directory containing experiment JSON files (default: `./outputs`)
- `--save-dir`: Directory to save generated plots (default: `./analysis/plots`)
- `--show`: Display plots interactively (default: False, saves only)

## Generated Plots

For each dataset, the following plots are generated:

1. **Training Curves** (`{dataset}_best_config_training_curves.png`): Loss, Dice, IoU, and Accuracy over epochs for the best configuration
2. **Metric Comparisons** (`{dataset}_dice_by_{factor}.png`): Bar charts comparing Dice score by model, loss, or optimizer
3. **Heatmaps** (`{dataset}_dice_heatmap_{vars}.png`): Heatmaps showing Dice scores across different combinations
4. **All Metrics Comparison** (`{dataset}_all_metrics_comparison.png`): Grouped bar chart comparing all metrics
5. **Learning Curves** (`{dataset}_learning_curves_{metric}.png`): Learning curves for all experiments
6. **Ablation Study Summary** (`{dataset}_ablation_summary.png`): Comprehensive summary with multiple subplots

Cross-dataset comparisons are also generated when multiple datasets are available.

## Output Files

- **Plots**: Saved as PNG files in `{save_dir}/{dataset_name}/`
- **Summary CSV**: `experiment_summary.csv` with all experiment results in tabular format

## Example

```bash
# Generate all plots (saves only, no display)
python analysis/main.py

# Generate plots and display them
python analysis/main.py --show

# Use custom directories
python analysis/main.py --output-dir ./my_outputs --save-dir ./my_plots
```

## Visualization Scripts

### Single Model Visualization

Visualize predictions from a single model checkpoint:

```bash
python -m analysis.visualize_predictions --checkpoint ./checkpoints/segmentation_ablation_PH2_best.pth --num-samples 5
```

### Multiple Models Comparison

Compare predictions from multiple models side-by-side:

```bash
python -m analysis.visualize_multiple_models --checkpoints ./checkpoints/segmentation_ablation_PH2_best.pth ./checkpoints/segmentation_ablation_DRIVE_best.pth --num-samples 3
```

**Options:**
- `--checkpoint` / `--checkpoints`: Path(s) to model checkpoint(s)
- `--num-samples`: Number of test samples to visualize (default: 3-5)
- `--save-dir`: Directory to save visualizations (default: `./analysis/visualizations`)
- `--device`: Device to use (default: cuda if available)

