"""
Normalization and transformation utilities for parameter vectors.

Transforms parameters to a numerically friendly representation for training,
with inverse transforms to map back to physical units.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass, field
from .schema import default_schema, ParameterSchema


@dataclass
class TransformConfig:
    """
    Configuration for parameter transformations.
    
    Specifies which fields are log-transformed, z-scored, or use
    sin/cos representation (for phase).
    """
    # Parameters that should be log-transformed (typically positive values)
    log_params: List[str] = field(default_factory=lambda: ["Cm", "T2", "T2_prime", "g"])
    
    # Parameters that should be z-scored (centered and scaled)
    zscore_params: List[str] = field(default_factory=lambda: ["Cm", "T2", "T2_prime", "delta_f", "g"])
    
    # Parameters that use sin/cos representation (phase)
    phase_params: List[str] = field(default_factory=lambda: ["phi"])
    
    # Whether to apply transforms to global params
    transform_global: bool = True


def fit_stats(
    X: np.ndarray,
    schema: Optional[ParameterSchema] = None,
    config: Optional[TransformConfig] = None
) -> Dict[str, np.ndarray]:
    """
    Compute mean and std statistics from a dataset for normalization.
    
    Args:
        X: Parameter vectors, shape (n_samples, D)
        schema: Parameter schema
        config: Transform configuration
    
    Returns:
        Dictionary with 'mean' and 'std' arrays, shape (D,)
    """
    if schema is None:
        schema = default_schema
    if config is None:
        config = TransformConfig()
    
    # Compute statistics in log space for log-transformed params
    X_transformed = np.copy(X)
    
    # Apply log transform to log_params before computing stats
    for met in schema.metabolites:
        for param in schema.metabolite_params:
            if param in config.log_params:
                idx = schema.get_idx(met, param)
                # Only compute stats on positive values
                X_transformed[:, idx] = np.log(np.maximum(X[:, idx], 1e-10))
    
    # Handle global params
    if config.transform_global:
        for param in schema.global_params:
            if param in config.log_params:
                idx = schema.get_global_idx(param)
                X_transformed[:, idx] = np.log(np.maximum(X[:, idx], 1e-10))
    
    mean = np.mean(X_transformed, axis=0)
    std = np.std(X_transformed, axis=0)
    # Avoid division by zero
    std = np.maximum(std, 1e-8)
    
    return {"mean": mean, "std": std}


def transform(
    theta: Union[np.ndarray, List[float]],
    stats: Dict[str, np.ndarray],
    schema: Optional[ParameterSchema] = None,
    config: Optional[TransformConfig] = None
) -> np.ndarray:
    """
    Transform parameter vector to normalized representation.
    
    Args:
        theta: Parameter vector, shape (D,) or (n_samples, D)
        stats: Statistics dict with 'mean' and 'std' arrays
        schema: Parameter schema
        config: Transform configuration
    
    Returns:
        Transformed parameter vector
    """
    if schema is None:
        schema = default_schema
    if config is None:
        config = TransformConfig()
    
    theta = np.asarray(theta)
    is_1d = theta.ndim == 1
    if is_1d:
        theta = theta[np.newaxis, :]
    
    theta_tilde = np.copy(theta)
    mean = stats["mean"]
    std = stats["std"]
    
    # Transform metabolite parameters
    for met in schema.metabolites:
        for param in schema.metabolite_params:
            idx = schema.get_idx(met, param)
            
            if param in config.phase_params:
                # Phase: convert to sin/cos representation
                sin_val = np.sin(theta[:, idx])
                cos_val = np.cos(theta[:, idx])
                # Store as two values (will need to adjust dimension handling)
                # For now, keep as single value but note this needs expansion
                theta_tilde[:, idx] = np.sin(theta[:, idx])
            elif param in config.log_params:
                # Log transform
                theta_tilde[:, idx] = np.log(np.maximum(theta[:, idx], 1e-10))
            else:
                theta_tilde[:, idx] = theta[:, idx]
            
            # Apply z-scoring if configured
            if param in config.zscore_params:
                theta_tilde[:, idx] = (theta_tilde[:, idx] - mean[idx]) / std[idx]
    
    # Transform global parameters
    if config.transform_global:
        for param in schema.global_params:
            idx = schema.get_global_idx(param)
            
            if param in config.phase_params:
                theta_tilde[:, idx] = np.sin(theta[:, idx])
            elif param in config.log_params:
                theta_tilde[:, idx] = np.log(np.maximum(theta[:, idx], 1e-10))
            else:
                theta_tilde[:, idx] = theta[:, idx]
            
            # Apply z-scoring if configured
            if param in config.zscore_params:
                theta_tilde[:, idx] = (theta_tilde[:, idx] - mean[idx]) / std[idx]
    
    if is_1d:
        theta_tilde = theta_tilde[0]
    
    return theta_tilde


def inverse_transform(
    theta_tilde: Union[np.ndarray, List[float]],
    stats: Dict[str, np.ndarray],
    schema: Optional[ParameterSchema] = None,
    config: Optional[TransformConfig] = None
) -> np.ndarray:
    """
    Inverse transform from normalized representation back to physical units.
    
    Args:
        theta_tilde: Transformed parameter vector, shape (D,) or (n_samples, D)
        stats: Statistics dict with 'mean' and 'std' arrays
        schema: Parameter schema
        config: Transform configuration
    
    Returns:
        Parameter vector in physical units
    """
    if schema is None:
        schema = default_schema
    if config is None:
        config = TransformConfig()
    
    theta_tilde = np.asarray(theta_tilde)
    is_1d = theta_tilde.ndim == 1
    if is_1d:
        theta_tilde = theta_tilde[np.newaxis, :]
    
    theta = np.copy(theta_tilde)
    mean = stats["mean"]
    std = stats["std"]
    
    # Inverse transform metabolite parameters
    for met in schema.metabolites:
        for param in schema.metabolite_params:
            idx = schema.get_idx(met, param)
            
            # Reverse z-scoring if configured
            if param in config.zscore_params:
                theta[:, idx] = theta[:, idx] * std[idx] + mean[idx]
            
            # Reverse log transform
            if param in config.log_params:
                theta[:, idx] = np.exp(theta[:, idx])
            elif param in config.phase_params:
                # Phase: convert from sin back to angle
                # Note: this loses quadrant information, may need sin/cos pair
                theta[:, idx] = np.arcsin(np.clip(theta[:, idx], -1, 1))
    
    # Inverse transform global parameters
    if config.transform_global:
        for param in schema.global_params:
            idx = schema.get_global_idx(param)
            
            if param in config.zscore_params:
                theta[:, idx] = theta[:, idx] * std[idx] + mean[idx]
            
            if param in config.log_params:
                theta[:, idx] = np.exp(theta[:, idx])
            elif param in config.phase_params:
                theta[:, idx] = np.arcsin(np.clip(theta[:, idx], -1, 1))
    
    if is_1d:
        theta = theta[0]
    
    return theta


# Phase utilities

def phase_to_sincos(phi: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert phase angle to (sin, cos) representation.
    
    Args:
        phi: Phase angle in radians
    
    Returns:
        Tuple of (sin(phi), cos(phi))
    """
    return np.sin(phi), np.cos(phi)


def sincos_to_phase(sin_val: Union[float, np.ndarray], cos_val: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert (sin, cos) representation back to phase angle.
    
    Args:
        sin_val: sin(phi)
        cos_val: cos(phi)
    
    Returns:
        Phase angle phi in radians (using atan2)
    """
    return np.arctan2(sin_val, cos_val)

