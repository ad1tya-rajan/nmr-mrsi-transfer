"""
Synthetic transfer rules for creating paired NMR -> MRSI parameter sets.

This creates supervised pairs (theta_nmr, theta_mrsi) using domain-dependent
transformation rules, enabling end-to-end pipeline testing.
"""

from typing import Optional
import numpy as np
from ..utils.schema import default_schema, ParameterSchema


def transfer(
    theta_nmr: np.ndarray,
    schema: Optional[ParameterSchema] = None,
    noise_level: float = 0.1,
    amplitude_scale: float = 0.8,
    linewidth_broadening: float = 1.3,
    frequency_jitter: float = 1.5
) -> np.ndarray:
    """
    Transfer NMR parameters to MRSI domain using synthetic rules.
    
    This creates a synthetic teacher mapping that approximates the
    cross-scanner transfer function.
    
    Args:
        theta_nmr: NMR parameter vector, shape (D,) or (n_samples, D)
        schema: Parameter schema
        noise_level: Noise level for transfer
        amplitude_scale: Scaling factor for amplitudes (typically < 1.0)
        linewidth_broadening: Factor for increasing linewidths
        frequency_jitter: Factor for increasing frequency jitter
    
    Returns:
        MRSI parameter vector, shape matching theta_nmr
    """
    if schema is None:
        schema = default_schema
    
    theta_nmr = np.asarray(theta_nmr)
    is_1d = theta_nmr.ndim == 1
    if is_1d:
        theta_nmr = theta_nmr[np.newaxis, :]
    
    n_samples = theta_nmr.shape[0]
    theta_mrsi = np.copy(theta_nmr)
    
    # Transfer metabolite parameters
    for i in range(n_samples):
        for met in schema.metabolites:
            # Amplitude scaling (typically reduced in MRSI)
            Cm_nmr = schema.get_param(theta_nmr[i], met, "Cm")
            Cm_mrsi = Cm_nmr * amplitude_scale
            # Add noise
            Cm_mrsi *= (1.0 + np.random.normal(0, noise_level * 0.1))
            Cm_mrsi = np.maximum(Cm_mrsi, 0.01)  # Ensure positive
            schema.set_param(theta_mrsi[i], met, "Cm", Cm_mrsi)
            
            # T2: typically shorter in MRSI (increased relaxation)
            T2_nmr = schema.get_param(theta_nmr[i], met, "T2")
            T2_mrsi = T2_nmr / linewidth_broadening
            # Add noise
            T2_mrsi *= (1.0 + np.random.normal(0, noise_level * 0.15))
            T2_mrsi = np.clip(T2_mrsi, 10.0, 1000.0)  # Physical bounds
            schema.set_param(theta_mrsi[i], met, "T2", T2_mrsi)
            
            # T2': also shorter in MRSI
            T2p_nmr = schema.get_param(theta_nmr[i], met, "T2_prime")
            T2p_mrsi = T2p_nmr / linewidth_broadening
            T2p_mrsi *= (1.0 + np.random.normal(0, noise_level * 0.15))
            T2p_mrsi = np.clip(T2p_mrsi, 5.0, 200.0)
            schema.set_param(theta_mrsi[i], met, "T2_prime", T2p_mrsi)
            
            # Frequency shift: increased jitter in MRSI
            df_nmr = schema.get_param(theta_nmr[i], met, "delta_f")
            df_mrsi = df_nmr * frequency_jitter
            # Add additional jitter
            df_mrsi += np.random.normal(0, noise_level * 2.0)
            schema.set_param(theta_mrsi[i], met, "delta_f", df_mrsi)
        
        # Global phase: add noise
        phi_nmr = schema.get_global_param(theta_nmr[i], "phi")
        phi_mrsi = phi_nmr + np.random.normal(0, noise_level * 0.2)
        # Wrap to [-pi, pi]
        phi_mrsi = np.arctan2(np.sin(phi_mrsi), np.cos(phi_mrsi))
        schema.set_global_param(theta_mrsi[i], "phi", phi_mrsi)
        
        # Global linewidth: increased in MRSI
        g_nmr = schema.get_global_param(theta_nmr[i], "g")
        g_mrsi = g_nmr * linewidth_broadening
        g_mrsi *= (1.0 + np.random.normal(0, noise_level * 0.1))
        g_mrsi = np.clip(g_mrsi, 0.5, 3.0)
        schema.set_global_param(theta_mrsi[i], "g", g_mrsi)
    
    if is_1d:
        theta_mrsi = theta_mrsi[0]
    
    return theta_mrsi


def create_paired_dataset(
    n_samples: int,
    schema: Optional[ParameterSchema] = None,
    transfer_kwargs: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a paired dataset of (NMR, MRSI) parameter vectors.
    
    Args:
        n_samples: Number of sample pairs to generate
        schema: Parameter schema
        transfer_kwargs: Additional kwargs for transfer function
    
    Returns:
        Tuple of (theta_nmr, theta_mrsi), each shape (n_samples, D)
    """
    from .parameter_sampling import sample_parameter_vector
    
    if schema is None:
        schema = default_schema
    
    if transfer_kwargs is None:
        transfer_kwargs = {}
    
    # Sample NMR parameters
    theta_nmr = sample_parameter_vector(
        domain="nmr",
        n_samples=n_samples,
        schema=schema
    )
    
    # Transfer to MRSI
    theta_mrsi = transfer(theta_nmr, schema=schema, **transfer_kwargs)
    
    return theta_nmr, theta_mrsi

