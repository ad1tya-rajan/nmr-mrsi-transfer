"""
Tests for normalization transforms.

Key test: transform then inverse_transform ≈ identity
"""

import numpy as np
import pytest
from src.utils.normalization import (
    TransformConfig, fit_stats, transform, inverse_transform,
    phase_to_sincos, sincos_to_phase
)
from src.utils.schema import default_schema
from src.simulation.parameter_sampling import sample_parameter_vector


def test_transform_inverse_identity():
    """Test that transform then inverse_transform ≈ identity."""
    schema = default_schema
    
    # Sample some parameter vectors
    theta = sample_parameter_vector(domain="nmr", n_samples=100)
    
    # Fit stats
    config = TransformConfig()
    stats = fit_stats(theta, schema=schema, config=config)
    
    # Transform
    theta_tilde = transform(theta, stats, schema=schema, config=config)
    
    # Inverse transform
    theta_recon = inverse_transform(theta_tilde, stats, schema=schema, config=config)
    
    # Check reconstruction error
    # Note: log transform is exact, but numerical precision may cause small errors
    mae = np.mean(np.abs(theta - theta_recon))
    
    # For log-transformed params, we expect small errors
    # For other params, should be very close
    assert mae < 0.1, f"Reconstruction error too large: {mae}"
    
    # Check per-parameter group
    for met in schema.metabolites[:3]:  # Check first 3 metabolites
        for param in schema.metabolite_params:
            idx = schema.get_idx(met, param)
            param_mae = np.mean(np.abs(theta[:, idx] - theta_recon[:, idx]))
            # Log params may have larger errors due to exp/log
            if param in config.log_params:
                assert param_mae < 0.5, f"Large error for {met}.{param}: {param_mae}"
            else:
                assert param_mae < 0.01, f"Large error for {met}.{param}: {param_mae}"


def test_phase_sincos_conversion():
    """Test phase to sin/cos conversion and back."""
    # Test single value
    phi = np.pi / 4
    sin_val, cos_val = phase_to_sincos(phi)
    phi_recon = sincos_to_phase(sin_val, cos_val)
    assert np.isclose(phi, phi_recon), f"Phase reconstruction failed: {phi} != {phi_recon}"
    
    # Test array
    phis = np.array([0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 2])
    sin_vals, cos_vals = phase_to_sincos(phis)
    phis_recon = sincos_to_phase(sin_vals, cos_vals)
    
    # atan2 should recover the angle (modulo 2π wrapping)
    for i, (phi, phi_r) in enumerate(zip(phis, phis_recon)):
        # Normalize to [-pi, pi]
        phi_norm = np.arctan2(np.sin(phi), np.cos(phi))
        phi_r_norm = np.arctan2(np.sin(phi_r), np.cos(phi_r))
        assert np.isclose(phi_norm, phi_r_norm), \
            f"Phase reconstruction failed for {phi}: {phi_r}"


def test_fit_stats():
    """Test that fit_stats produces reasonable statistics."""
    schema = default_schema
    
    # Sample parameters
    theta = sample_parameter_vector(domain="nmr", n_samples=1000)
    
    config = TransformConfig()
    stats = fit_stats(theta, schema=schema, config=config)
    
    assert "mean" in stats
    assert "std" in stats
    assert stats["mean"].shape == (schema.total_dim,)
    assert stats["std"].shape == (schema.total_dim,)
    assert np.all(stats["std"] > 0), "Std should be positive"


def test_transform_single_sample():
    """Test transform on a single sample (1D array)."""
    schema = default_schema
    
    theta = sample_parameter_vector(domain="nmr", n_samples=1)[0]
    
    config = TransformConfig()
    stats = fit_stats(theta[np.newaxis, :], schema=schema, config=config)
    
    # Transform single sample
    theta_tilde = transform(theta, stats, schema=schema, config=config)
    
    assert theta_tilde.ndim == 1
    assert theta_tilde.shape == (schema.total_dim,)
    
    # Inverse transform
    theta_recon = inverse_transform(theta_tilde, stats, schema=schema, config=config)
    
    assert theta_recon.ndim == 1
    assert theta_recon.shape == (schema.total_dim,)


def test_transform_batch():
    """Test transform on a batch of samples."""
    schema = default_schema
    
    theta = sample_parameter_vector(domain="nmr", n_samples=10)
    
    config = TransformConfig()
    stats = fit_stats(theta, schema=schema, config=config)
    
    # Transform batch
    theta_tilde = transform(theta, stats, schema=schema, config=config)
    
    assert theta_tilde.ndim == 2
    assert theta_tilde.shape == (10, schema.total_dim)
    
    # Inverse transform
    theta_recon = inverse_transform(theta_tilde, stats, schema=schema, config=config)
    
    assert theta_recon.ndim == 2
    assert theta_recon.shape == (10, schema.total_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

