"""
Parameter sampling for biologically plausible metabolite parameters.

Supports different domains (NMR vs MRSI) with domain-specific distributions.
"""

from typing import Dict, Optional, List
import numpy as np
from dataclasses import dataclass
from ..utils.schema import default_schema, ParameterSchema


@dataclass
class DomainConfig:
    """Distribution parameters for a specific domain (NMR or MRSI)."""
    # Concentration/amplitude ranges (arbitrary units, will be normalized)
    Cm_range: tuple = (0.1, 10.0)
    
    # T2 relaxation times (ms)
    T2_range: tuple = (50.0, 500.0)
    
    # T2' relaxation times (ms)
    T2_prime_range: tuple = (10.0, 100.0)
    
    # Frequency shift ranges (Hz)
    delta_f_range: tuple = (-5.0, 5.0)
    
    # Global phase (radians)
    phi_range: tuple = (-np.pi, np.pi)
    
    # Global linewidth factor
    g_range: tuple = (0.5, 2.0)
    
    # Noise level for sampling
    noise_level: float = 0.1


# Domain-specific configurations
DOMAIN_CONFIGS = {
    "nmr": DomainConfig(
        Cm_range=(0.5, 10.0),
        T2_range=(100.0, 500.0),
        T2_prime_range=(20.0, 100.0),
        delta_f_range=(-2.0, 2.0),
        phi_range=(-np.pi, np.pi),
        g_range=(0.5, 1.5),
        noise_level=0.05,
    ),
    "mrsi": DomainConfig(
        Cm_range=(0.1, 8.0),
        T2_range=(50.0, 300.0),  # Shorter T2 in MRSI
        T2_prime_range=(10.0, 80.0),  # Shorter T2' in MRSI
        delta_f_range=(-5.0, 5.0),  # Larger frequency jitter
        phi_range=(-np.pi, np.pi),
        g_range=(1.0, 2.5),  # Broader linewidths
        noise_level=0.15,
    ),
}


def sample_metabolite_params(
    metabolite: str,
    domain: str = "nmr",
    n_samples: int = 1,
    schema: Optional[ParameterSchema] = None,
    config: Optional[DomainConfig] = None
) -> np.ndarray:
    """
    Sample parameters for a single metabolite.
    
    Args:
        metabolite: Metabolite name
        domain: Domain name ("nmr" or "mrsi")
        n_samples: Number of samples to generate
        schema: Parameter schema
        config: Domain configuration (defaults to DOMAIN_CONFIGS[domain])
    
    Returns:
        Array of shape (n_samples, n_metabolite_params)
    """
    if schema is None:
        schema = default_schema
    
    if config is None:
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_CONFIGS.keys())}")
        config = DOMAIN_CONFIGS[domain]
    
    params = np.zeros((n_samples, schema.n_metabolite_params))
    
    # Sample concentration (amplitude)
    Cm_min, Cm_max = config.Cm_range
    params[:, schema.get_param_idx("concentration")] = np.random.uniform(
        Cm_min, Cm_max, size=n_samples
    )
    
    # Sample T2 (half-normal distribution, clipped)
    T2_min, T2_max = config.T2_range
    T2_mean = (T2_min + T2_max) / 2
    T2_std = (T2_max - T2_min) / 4
    params[:, schema.get_param_idx("T2")] = np.clip(
        np.abs(np.random.normal(T2_mean, T2_std, size=n_samples)),
        T2_min, T2_max
    )
    
    # Sample T2' (half-normal distribution, clipped)
    T2p_min, T2p_max = config.T2_prime_range
    T2p_mean = (T2p_min + T2p_max) / 2
    T2p_std = (T2p_max - T2p_min) / 4
    params[:, schema.get_param_idx("T2p")] = np.clip(
        np.abs(np.random.normal(T2p_mean, T2p_std, size=n_samples)),
        T2p_min, T2p_max
    )
    
    # Sample phase (uniform in [-pi, pi])
    params[:, schema.get_param_idx("phase")] = np.random.uniform(
        -np.pi, np.pi, size=n_samples
    )
    
    # Sample freq_shift (frequency shift)
    df_min, df_max = config.delta_f_range
    params[:, schema.get_param_idx("freq_shift")] = np.random.uniform(
        df_min, df_max, size=n_samples
    )
    
    # Sample linewidth (metabolite-specific linewidth, uniform)
    # Use a reasonable range for metabolite linewidths
    params[:, schema.get_param_idx("linewidth")] = np.random.uniform(
        0.1, 2.0, size=n_samples
    )
    
    return params


def sample_global_params(
    domain: str = "nmr",
    n_samples: int = 1,
    schema: Optional[ParameterSchema] = None,
    config: Optional[DomainConfig] = None
) -> np.ndarray:
    """
    Sample global parameters.
    
    Args:
        domain: Domain name ("nmr" or "mrsi")
        n_samples: Number of samples to generate
        schema: Parameter schema
        config: Domain configuration
    
    Returns:
        Array of shape (n_samples, n_global_params)
    """
    if schema is None:
        schema = default_schema
    
    if config is None:
        if domain not in DOMAIN_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}")
        config = DOMAIN_CONFIGS[domain]
    
    params = np.zeros((n_samples, schema.n_global_params))
    
    # Sample phi (global phase)
    phi_min, phi_max = config.phi_range
    params[:, schema.global_params.index("phi")] = np.random.uniform(
        phi_min, phi_max, size=n_samples
    )
    
    # Sample g (global linewidth factor)
    g_min, g_max = config.g_range
    params[:, schema.global_params.index("linewidth")] = np.random.uniform(
        g_min, g_max, size=n_samples
    )
    
    return params


def sample_parameter_vector(
    domain: str = "nmr",
    n_samples: int = 1,
    schema: Optional[ParameterSchema] = None,
    add_noise: bool = True
) -> np.ndarray:
    """
    Sample a complete parameter vector (all metabolites + global params).
    
    Args:
        domain: Domain name ("nmr" or "mrsi")
        n_samples: Number of samples to generate
        schema: Parameter schema
        add_noise: Whether to add small noise to parameters
    
    Returns:
        Array of shape (n_samples, D) where D is total parameter dimension
    """
    if schema is None:
        schema = default_schema
    
    if domain not in DOMAIN_CONFIGS:
        raise ValueError(f"Unknown domain: {domain}")
    config = DOMAIN_CONFIGS[domain]
    
    # Sample parameters for each metabolite
    theta = np.zeros((n_samples, schema.total_dim))
    
    for i, met in enumerate(schema.metabolites):
        met_params = sample_metabolite_params(
            met, domain=domain, n_samples=n_samples, schema=schema, config=config
        )
        start_idx = schema.get_idx(met)
        end_idx = start_idx + schema.n_metabolite_params
        theta[:, start_idx:end_idx] = met_params
    
    # Sample global parameters
    global_params = sample_global_params(
        domain=domain, n_samples=n_samples, schema=schema, config=config
    )
    global_start = schema.n_metabolites * schema.n_metabolite_params
    theta[:, global_start:] = global_params
    
    # Add small noise if requested
    if add_noise:
        noise = np.random.normal(0, config.noise_level, size=theta.shape)
        theta += noise
        # Clip to ensure physical plausibility
        theta = np.clip(theta, 0.01, None)  # Ensure positive values
    
    return theta

