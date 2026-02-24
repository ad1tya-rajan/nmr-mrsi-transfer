"""
Weighted Huber loss for parameter regression.

Huber loss is more robust to outliers than MSE while remaining smooth.
"""

from typing import Optional
import torch
import torch.nn as nn


class WeightedHuberLoss(nn.Module):
    """
    Weighted Huber loss.
    
    Huber loss combines L1 (for large errors) and L2 (for small errors)
    with a delta parameter controlling the transition point.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "mean",
        weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            delta: Threshold for transition from L2 to L1 loss
            reduction: Reduction method ("mean", "sum", or "none")
            weights: Optional per-parameter weights, shape (D,)
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.register_buffer("weights", weights)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted Huber loss.
        
        Args:
            pred: Predicted values, shape (batch_size, D)
            target: Target values, shape (batch_size, D)
        
        Returns:
            Loss value
        """
        error = pred - target
        abs_error = torch.abs(error)
        
        # Huber loss: L2 for small errors, L1 for large errors
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        
        # Apply weights if provided
        if self.weights is not None:
            loss = loss * self.weights.unsqueeze(0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def create_huber_loss(
    delta: float = 1.0,
    weights: Optional[torch.Tensor] = None
) -> WeightedHuberLoss:
    """
    Factory function to create a weighted Huber loss.
    
    Args:
        delta: Threshold parameter
        weights: Optional per-parameter weights
    
    Returns:
        WeightedHuberLoss instance
    """
    return WeightedHuberLoss(delta=delta, weights=weights)
