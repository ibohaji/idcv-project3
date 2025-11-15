import torch 


class FocalLoss(torch.nn.Module): 
    """
    Focal Loss for binary segmentation with class-specific alpha weighting.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - alpha_t = alpha if y_true = 1 (positive class), else (1 - alpha) for negative class
    - p_t is the predicted probability for the true class
    - gamma controls the focusing strength (higher = more focus on hard examples)
    
    This helps address class imbalance by:
    1. Downweighting easy examples (high confidence predictions)
    2. Upweighting hard examples (low confidence predictions)
    3. Class-specific alpha balancing positive vs negative class
    """
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_pred, y_true): 
        # Get probabilities
        p = torch.sigmoid(y_pred)
        
        # Calculate p_t: probability of the true class
        # p_t = p if y_true = 1, else (1 - p) if y_true = 0
        p_t = p * y_true + (1 - p) * (1 - y_true)
        
        # Class-specific alpha: alpha for positive class, (1-alpha) for negative class
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        
        # Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        # Using log(p_t) = -log(1 - p_t) for numerical stability when p_t is small
        # But we can use: -log(p_t) directly
        ce_loss = -torch.log(p_t + 1e-8)  # Add small epsilon for numerical stability
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class BCE(torch.nn.Module): 
    def __init__(self): 
        super().__init__()

    def forward(self, y_pred, y_true): 
        return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)


class WeightedBCE(torch.nn.Module):
    """Apply weight alpha_i where i is the inverse frequency of the class pixel i
    so that observations of the minority class are upweighted and observations of the majority class are downweighted"""
    
    def __init__(self): 
        super().__init__()

    def forward(self, y_pred, y_true):
        # Calculate per-sample loss (not averaged)
        bce_per_pixel = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        # Calculate weight per sample based on positive class frequency
        alpha_i = 1 / (y_true.sum(dim=[1,2,3]) + 1e-6)
        # Average loss per sample, then weight
        loss_per_sample = bce_per_pixel.mean(dim=[1,2,3])  # [batch_size]
        weighted_loss = alpha_i * loss_per_sample
        return weighted_loss.mean()
