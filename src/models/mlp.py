"""
Residual MLP model for learning cross-scanner transfer functions.

Implements a residual MLP with LayerNorm and GELU activations.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class ResidualMLP(nn.Module):
    """
    Residual MLP for parameter transfer.
    
    Architecture:
    - Input projection
    - N residual blocks with LayerNorm + GELU
    - Output projection
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
        
        Returns:
            Output tensor, shape (batch_size, output_dim)
        """
        x = self.input_proj(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_proj(x)
        
        return x


def create_model(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    n_layers: int = 4,
    dropout: float = 0.1
) -> ResidualMLP:
    """
    Factory function to create a ResidualMLP model.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension
        n_layers: Number of residual layers
        dropout: Dropout rate
    
    Returns:
        ResidualMLP model
    """
    return ResidualMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout
    )

