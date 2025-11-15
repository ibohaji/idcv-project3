import torch 


def dice_score(y_pred, y_true):
    """Dice metric for evaluation"""
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()  # Binarize and convert to float
    intersection = (y_pred * y_true).sum()
    return (2 * intersection) / (y_pred.sum() + y_true.sum() + 1e-6)

def iou_score(y_pred, y_true):
    """IoU metric for evaluation"""
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()  # Binarize and convert to float
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    return intersection / (union + 1e-6)

def accuracy(y_pred, y_true):
    """Accuracy metric for evaluation"""
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()  # Binarize and convert to float
    return (y_pred == y_true).float().mean()

def precission(y_pred, y_true):
    """Precission metric for evaluation"""
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()  # Binarize and convert to float
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    return tp / (tp + fp + 1e-6)


def specificity(y_pred, y_true):
    """Specificity (True Negative Rate) metric for evaluation"""
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()  # Binarize and convert to float
    tn = ((1 - y_pred) * (1 - y_true)).sum()
    fp = (y_pred * (1 - y_true)).sum()
    return tn / (tn + fp + 1e-6)

def sensitivity(y_pred, y_true):
    """Sensitivity (Recall) metric for evaluation"""
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()  # Binarize and convert to float
    tp = (y_pred * y_true).sum()
    fn = ((1 - y_pred) * y_true).sum()  # False negatives
    return tp / (tp + fn + 1e-6)
    