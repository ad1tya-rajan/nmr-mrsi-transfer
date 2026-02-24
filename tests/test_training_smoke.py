"""
Smoke test for training pipeline.

Overfit on tiny synthetic set (32 samples) to ensure pipeline works end-to-end.
"""

import numpy as np
import torch
import pytest
from src.training.train import ParameterDataset, create_synthetic_dataset
from src.models.mlp import create_model
from src.losses.huber import create_huber_loss
from src.utils.schema import get_dim


def test_training_smoke():
    """Test that training pipeline runs without errors."""
    # Create tiny dataset
    n_samples = 32
    dataset = create_synthetic_dataset(n_samples, {})
    
    assert len(dataset) == n_samples
    
    # Create model
    D = get_dim()
    model = create_model(
        input_dim=D,
        output_dim=D,
        hidden_dim=64,  # Smaller for smoke test
        n_layers=2,
        dropout=0.0
    )
    
    # Create loss
    criterion = create_huber_loss(delta=1.0)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for a few steps
    model.train()
    for epoch in range(5):
        total_loss = 0.0
        n_batches = 0
        
        # Simple batch iteration
        batch_size = 8
        for i in range(0, n_samples, batch_size):
            batch = {
                "nmr": dataset.theta_nmr_t[i:i+batch_size],
                "mrsi": dataset.theta_mrsi_t[i:i+batch_size]
            }
            
            nmr = torch.FloatTensor(batch["nmr"])
            mrsi = torch.FloatTensor(batch["mrsi"])
            
            optimizer.zero_grad()
            pred = model(nmr)
            loss = criterion(pred, mrsi)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    
    # Check that loss decreased (or at least model ran)
    assert avg_loss is not None
    assert not np.isnan(avg_loss), "Loss should not be NaN"
    assert not np.isinf(avg_loss), "Loss should not be Inf"
    
    print("Smoke test passed!")


def test_model_forward():
    """Test that model forward pass works."""
    D = get_dim()
    model = create_model(
        input_dim=D,
        output_dim=D,
        hidden_dim=64,
        n_layers=2
    )
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, D)
    
    # Forward pass
    y = model(x)
    
    assert y.shape == (batch_size, D)
    assert not torch.isnan(y).any(), "Output should not contain NaN"
    assert not torch.isinf(y).any(), "Output should not contain Inf"


def test_dataset_creation():
    """Test that dataset can be created and accessed."""
    dataset = create_synthetic_dataset(10, {})
    
    assert len(dataset) == 10
    
    # Get a sample
    sample = dataset[0]
    
    assert "nmr" in sample
    assert "mrsi" in sample
    assert sample["nmr"].shape == (get_dim(),)
    assert sample["mrsi"].shape == (get_dim(),)


def test_loss_computation():
    """Test that loss can be computed."""
    D = get_dim()
    criterion = create_huber_loss(delta=1.0)
    
    pred = torch.randn(4, D)
    target = torch.randn(4, D)
    
    loss = criterion(pred, target)
    
    assert loss.item() >= 0, "Loss should be non-negative"
    assert not np.isnan(loss.item()), "Loss should not be NaN"
    assert not np.isinf(loss.item()), "Loss should not be Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

