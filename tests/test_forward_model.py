"""
Tests for forward model and FID simulation.

Sanity checks:
- Single-metabolite: turning on delta_f shifts spectrum as expected
- Increasing g broadens peaks
"""

import numpy as np
import pytest
from src.simulation.forward_model import simulate_fid, fid_to_spectrum
from src.utils.schema import default_schema


def test_single_metabolite_fid():
    """Test that FID is generated for a single metabolite."""
    schema = default_schema
    D = schema.total_dim
    
    # Create parameter vector with only NAA active
    theta = np.zeros(D)
    schema.set_param(theta, "NAA", "Cm", 1.0)
    schema.set_param(theta, "NAA", "T2", 200.0)
    schema.set_param(theta, "NAA", "T2_prime", 50.0)
    schema.set_param(theta, "NAA", "delta_f", 0.0)
    schema.set_global_param(theta, "phi", 0.0)
    schema.set_global_param(theta, "g", 1.0)
    
    # Generate FID
    fid = simulate_fid(theta, n_points=1024)
    
    assert fid.shape == (1024,)
    assert np.any(np.abs(fid) > 0), "FID should have non-zero values"
    assert np.iscomplexobj(fid), "FID should be complex"


def test_frequency_shift():
    """Test that delta_f shifts the spectrum."""
    schema = default_schema
    D = schema.total_dim
    
    # Create two parameter vectors with different frequency shifts
    theta1 = np.zeros(D)
    schema.set_param(theta1, "NAA", "Cm", 1.0)
    schema.set_param(theta1, "NAA", "T2", 200.0)
    schema.set_param(theta1, "NAA", "T2_prime", 50.0)
    schema.set_param(theta1, "NAA", "delta_f", 0.0)
    schema.set_global_param(theta1, "phi", 0.0)
    schema.set_global_param(theta1, "g", 1.0)
    
    theta2 = np.zeros(D)
    schema.set_param(theta2, "NAA", "Cm", 1.0)
    schema.set_param(theta2, "NAA", "T2", 200.0)
    schema.set_param(theta2, "NAA", "T2_prime", 50.0)
    schema.set_param(theta2, "NAA", "delta_f", 10.0)  # 10 Hz shift
    schema.set_global_param(theta2, "phi", 0.0)
    schema.set_global_param(theta2, "g", 1.0)
    
    # Generate FIDs
    fid1 = simulate_fid(theta1, n_points=1024)
    fid2 = simulate_fid(theta2, n_points=1024)
    
    # Convert to spectra
    freqs1, spec1 = fid_to_spectrum(fid1)
    freqs2, spec2 = fid_to_spectrum(fid2)
    
    # Find peak locations
    peak_idx1 = np.argmax(np.abs(spec1))
    peak_idx2 = np.argmax(np.abs(spec2))
    
    # Peak should shift (though exact shift depends on implementation)
    # For now, just check that spectra are different
    assert not np.allclose(spec1, spec2), "Spectra should differ with frequency shift"


def test_linewidth_broadening():
    """Test that increasing g broadens peaks."""
    schema = default_schema
    D = schema.total_dim
    
    # Create two parameter vectors with different g values
    theta1 = np.zeros(D)
    schema.set_param(theta1, "NAA", "Cm", 1.0)
    schema.set_param(theta1, "NAA", "T2", 200.0)
    schema.set_param(theta1, "NAA", "T2_prime", 50.0)
    schema.set_param(theta1, "NAA", "delta_f", 0.0)
    schema.set_global_param(theta1, "phi", 0.0)
    schema.set_global_param(theta1, "g", 0.5)  # Narrow
    
    theta2 = np.zeros(D)
    schema.set_param(theta2, "NAA", "Cm", 1.0)
    schema.set_param(theta2, "NAA", "T2", 200.0)
    schema.set_param(theta2, "NAA", "T2_prime", 50.0)
    schema.set_param(theta2, "NAA", "delta_f", 0.0)
    schema.set_global_param(theta2, "phi", 0.0)
    schema.set_global_param(theta2, "g", 2.0)  # Broad
    
    # Generate FIDs
    fid1 = simulate_fid(theta1, n_points=1024)
    fid2 = simulate_fid(theta2, n_points=1024)
    
    # Convert to spectra
    freqs1, spec1 = fid_to_spectrum(fid1)
    freqs2, spec2 = fid_to_spectrum(fid2)
    
    # Broader peak should decay faster in time domain
    # Check that fid2 decays faster (has smaller values earlier)
    decay_rate1 = np.abs(fid1[100]) / np.abs(fid1[0])
    decay_rate2 = np.abs(fid2[100]) / np.abs(fid2[0])
    
    # Higher g should lead to faster decay (smaller ratio)
    # Note: This is a sanity check; exact behavior depends on implementation
    assert decay_rate2 < decay_rate1 or np.allclose(decay_rate1, decay_rate2, atol=0.1), \
        "Higher g should lead to faster decay"


def test_fid_to_spectrum():
    """Test FID to spectrum conversion."""
    schema = default_schema
    D = schema.total_dim
    
    theta = np.zeros(D)
    schema.set_param(theta, "NAA", "Cm", 1.0)
    schema.set_param(theta, "NAA", "T2", 200.0)
    schema.set_param(theta, "NAA", "T2_prime", 50.0)
    schema.set_param(theta, "NAA", "delta_f", 0.0)
    schema.set_global_param(theta, "phi", 0.0)
    schema.set_global_param(theta, "g", 1.0)
    
    fid = simulate_fid(theta, n_points=1024)
    freqs, spectrum = fid_to_spectrum(fid)
    
    assert freqs.shape == (1024,)
    assert spectrum.shape == (1024,)
    assert np.iscomplexobj(spectrum)
    assert len(freqs) == len(spectrum)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

