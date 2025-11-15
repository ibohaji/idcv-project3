import torch 


class FocalLoss(torch.nn.Module): 
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_pred, y_true): 
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
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
        self.bce = BCE()

    def forward(self, y_pred, y_true): 
        alpha_i = 1 / (y_true.sum(dim=[1,2,3]) + 1e-6)
        loss = self.bce(y_pred, y_true)
        # Expand alpha_i to match spatial dimensions
        alpha_i = alpha_i.view(-1, 1, 1, 1)
        weighted_loss = alpha_i * loss
        return weighted_loss.mean()
