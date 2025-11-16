"""Create theoretical Focal Loss plot showing different gamma values."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_theoretical_focal_loss_plot(save_path: Path = None):
    """Create the theoretical Focal Loss plot from losses.ipynb."""
    # Create p_t values
    p_t = np.linspace(0.01, 0.99, 200)
    
    # Cross-Entropy
    ce_loss = -np.log(p_t)
    
    # Focal Loss with DIFFERENT GAMMA values (using α=0.75 for vessels)
    alpha = 0.75
    focal_gamma_0 = alpha * ce_loss  # Just weighted CE
    focal_gamma_1 = alpha * (1 - p_t)**1 * ce_loss
    focal_gamma_2 = alpha * (1 - p_t)**2 * ce_loss
    focal_gamma_5 = alpha * (1 - p_t)**5 * ce_loss
    
    plt.figure(figsize=(13, 7))
    
    # Plot
    plt.plot(p_t, ce_loss, 'k-', linewidth=3.5, label='Cross-Entropy', alpha=0.8)
    plt.plot(p_t, focal_gamma_0, color='purple', linewidth=2.5, label='Focal (α=0.75, γ=0)', alpha=0.7, linestyle='--')
    plt.plot(p_t, focal_gamma_1, color='blue', linewidth=2.5, label='Focal (α=0.75, γ=1)', alpha=0.8)
    plt.plot(p_t, focal_gamma_2, color='orange', linewidth=3, label='Focal (α=0.75, γ=2)', alpha=0.8)
    plt.plot(p_t, focal_gamma_5, color='red', linewidth=2.5, label='Focal (α=0.75, γ=5)', alpha=0.8)
    
    # Shade regions
    plt.axvspan(0.8, 1.0, alpha=0.1, color='green', label='High confidence')
    plt.axvspan(0.5, 0.8, alpha=0.1, color='yellow', label='Medium confidence')
    plt.axvspan(0, 0.5, alpha=0.1, color='red', label='Low confidence')
    
    # Add horizontal line at a reasonable loss threshold
    plt.axhline(y=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    plt.text(0.02, 0.15, 'Loss ≈ 0.1\n(weak signal)', fontsize=9, color='gray')
    
    plt.xlabel('$p_t$ (Probability of True Class)', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Theoretical Focal Loss Curves for Different $\\gamma$ Values\n(α=0.75, for vessel class)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved theoretical Focal Loss plot to {save_path}")
    
    plt.close()

if __name__ == '__main__':
    output_dir = Path('analysis/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / 'theoretical_focal_loss_curves.png'
    create_theoretical_focal_loss_plot(save_path)

