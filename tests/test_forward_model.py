"""Tests for the forward model implementation."""

import numpy as np
import pytest

from src.utils.schema import default_schema
from src.simulation.forward_model import simulate_fid, fid_to_spectrum


def make_dummy_theta():
    """Create a dummy parameter vector with reasonable values."""
    theta = np.zeros(default_schema.total_dim)
    for met in default_schema.metabolites:
        theta[default_schema.get_idx(met, 'concentration')] = 0.1
        theta[default_schema.get_idx(met, 'T2')] = 100.0  # ms
        theta[default_schema.get_idx(met, 'T2p')] = 50.0   # ms
        theta[default_schema.get_idx(met, 'freq_shift')] = 0.0
    theta[default_schema.get_global_param_idx('phi')] = 0.0
    theta[default_schema.get_global_param_idx('linewidth')] = 0.0
    return theta


def test_simulate_fid_60dim_input():
    """Test that 60-dimensional input (metabolite params only) works."""
    # Create 60-dim input (metabolite params only)
    theta_60 = np.zeros(60)  # 10 metabolites × 6 params each
    for i, met in enumerate(default_schema.metabolites):
        base_idx = i * 6
        theta_60[base_idx + 0] = 0.1  # concentration
        theta_60[base_idx + 1] = 100.0  # T2 (ms)
        theta_60[base_idx + 2] = 50.0   # T2p (ms)
        theta_60[base_idx + 3] = 0.0    # phase
        theta_60[base_idx + 4] = 0.0    # freq_shift
        theta_60[base_idx + 5] = 0.0    # linewidth
    
    # Should work with 60-dim input
    fid_60 = simulate_fid(theta_60)
    assert fid_60.shape == (2048,)
    assert fid_60.dtype == np.complex128
    
    # Should produce same result as 62-dim with zero globals
    theta_62 = np.zeros(62)
    theta_62[:60] = theta_60
    fid_62 = simulate_fid(theta_62)
    
    np.testing.assert_array_almost_equal(fid_60, fid_62)


def test_simulate_fid_no_nan_inf():
    """Ensure no NaN or Inf in FID output."""
    theta = make_dummy_theta()
    fid = simulate_fid(theta)
    
    has_nan = np.isnan(fid).any()
    has_inf = np.isinf(fid).any()
    print(f"FID has NaN: {has_nan}, has Inf: {has_inf}, max abs: {np.max(np.abs(fid)):.6f}")
    assert not has_nan
    assert not has_inf


def test_zero_amplitude_zero_contribution():
    """Zero concentration should give zero signal."""
    theta = make_dummy_theta()
    # Set all concentrations to zero
    for met in default_schema.metabolites:
        theta[default_schema.get_idx(met, 'concentration')] = 0.0
    
    fid = simulate_fid(theta)
    max_abs = np.max(np.abs(fid))
    print(f"With zero concentrations, max |FID| = {max_abs:.2e}")
    assert np.allclose(fid, 0.0, atol=1e-10)


def test_freq_shift_affects_oscillation():
    """Different freq_shift should change the signal phase/frequency."""
    theta1 = make_dummy_theta()
    theta1[default_schema.get_idx('met_1', 'freq_shift')] = 10.0  # Hz
    
    theta2 = make_dummy_theta()
    theta2[default_schema.get_idx('met_1', 'freq_shift')] = 20.0  # Hz
    
    fid1 = simulate_fid(theta1)
    fid2 = simulate_fid(theta2)
    
    # Should be different
    diff_max = np.max(np.abs(fid1 - fid2))
    print(f"Max difference between freq_shift 10Hz vs 20Hz: {diff_max:.6f}")
    assert not np.allclose(fid1, fid2, atol=1e-3)


def test_linewidth_affects_damping():
    """Larger linewidth should increase Gaussian damping."""
    theta1 = make_dummy_theta()
    theta1[default_schema.get_global_param_idx('linewidth')] = 0.0
    
    theta2 = make_dummy_theta()
    theta2[default_schema.get_global_param_idx('linewidth')] = 100.0
    
    fid1 = simulate_fid(theta1)
    fid2 = simulate_fid(theta2)
    
    # Should be different due to damping
    diff_max = np.max(np.abs(fid1 - fid2))
    print(f"Max difference between linewidth 0 vs 100: {diff_max:.6f}")
    assert not np.allclose(fid1, fid2, atol=1e-3)


def test_batch_vs_single_consistency():
    """Batch and single-sample should give same results."""
    theta = make_dummy_theta()
    
    fid_single = simulate_fid(theta)
    fid_batch = simulate_fid(np.stack([theta, theta]))[0]
    
    max_diff = np.max(np.abs(fid_single - fid_batch))
    print(f"Max difference between single and batch[0]: {max_diff:.2e}")
    assert np.allclose(fid_single, fid_batch, atol=1e-10)


def test_fid_to_spectrum():
    """Test spectrum conversion."""
    theta = make_dummy_theta()
    fid = simulate_fid(theta)
    
    freqs, spectrum = fid_to_spectrum(fid)
    
    print(f"Spectrum shape: {spectrum.shape}, freq range: {freqs.min():.1f} to {freqs.max():.1f} Hz")
    assert freqs.shape == (2048,)
    assert spectrum.shape == (2048,)
    assert spectrum.dtype == np.complex128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
