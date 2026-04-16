"""
Metrics for evaluating parameter prediction performance.

Provides MAE per parameter group and physical plausibility checks.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from .schema import default_schema, ParameterSchema


def compute_mae_per_group(
    pred: np.ndarray,
    target: np.ndarray,
    schema: Optional[ParameterSchema] = None
) -> Dict[str, float]:
    """
    Compute MAE per parameter group.
    
    Groups:
    - amplitudes: All concentration values
    - T2: All T2 values
    - T2p: All T2' values
    - freq_shift: All frequency shifts
    - global: Global parameters (phi, linewidth)
    
    Args:
        pred: Predicted parameters, shape (n_samples, D)
        target: Target parameters, shape (n_samples, D)
        schema: Parameter schema
    
    Returns:
        Dictionary of MAE values per group
    """
    if schema is None:
        schema = default_schema
    
    pred = np.asarray(pred)
    target = np.asarray(target)
    
    metrics = {}
    
    # Amplitudes (concentration)
    cm_indices = []
    for met in schema.metabolites:
        cm_indices.append(schema.get_idx(met, "concentration"))
    cm_indices = np.array(cm_indices)
    metrics["mae_amplitudes"] = np.mean(np.abs(pred[:, cm_indices] - target[:, cm_indices]))
    
    # T2
    t2_indices = []
    for met in schema.metabolites:
        t2_indices.append(schema.get_idx(met, "T2"))
    t2_indices = np.array(t2_indices)
    metrics["mae_T2"] = np.mean(np.abs(pred[:, t2_indices] - target[:, t2_indices]))
    
    # T2'
    t2p_indices = []
    for met in schema.metabolites:
        t2p_indices.append(schema.get_idx(met, "T2p"))
    t2p_indices = np.array(t2p_indices)
    metrics["mae_T2p"] = np.mean(np.abs(pred[:, t2p_indices] - target[:, t2p_indices]))
    
    # Frequency shifts
    df_indices = []
    for met in schema.metabolites:
        df_indices.append(schema.get_idx(met, "freq_shift"))
    df_indices = np.array(df_indices)
    metrics["mae_freq_shift"] = np.mean(np.abs(pred[:, df_indices] - target[:, df_indices]))
    
    # Global parameters
    global_start = schema.n_metabolites * schema.n_metabolite_params
    metrics["mae_global"] = np.mean(np.abs(pred[:, global_start:] - target[:, global_start:]))
    
    # Overall MAE
    metrics["mae_overall"] = np.mean(np.abs(pred - target))
    
    return metrics


def check_physical_plausibility(
    theta: np.ndarray,
    schema: Optional[ParameterSchema] = None,
    return_violations: bool = False
) -> Dict[str, any]:
    """
    Check physical plausibility of parameter vectors.
    
    Checks:
    - Positive values for concentration, T2, T2', linewidth
    - T2 and T2' within reasonable ranges
    - Phase within [-pi, pi]
    
    Args:
        theta: Parameter vectors, shape (n_samples, D) or (D,)
        schema: Parameter schema
        return_violations: Whether to return detailed violation info
    
    Returns:
        Dictionary with plausibility metrics
    """
    if schema is None:
        schema = default_schema
    
    theta = np.asarray(theta)
    is_1d = theta.ndim == 1
    if is_1d:
        theta = theta[np.newaxis, :]
    
    n_samples = theta.shape[0]
    violations = []
    
    for i in range(n_samples):
        # Check metabolite parameters
        for met in schema.metabolites:
            Cm = schema.get_param(theta[i], met, "concentration")
            T2 = schema.get_param(theta[i], met, "T2")
            T2p = schema.get_param(theta[i], met, "T2p")
            
            if Cm <= 0:
                violations.append(f"Sample {i}, {met}: concentration <= 0 ({Cm:.4f})")
            if T2 <= 0:
                violations.append(f"Sample {i}, {met}: T2 <= 0 ({T2:.4f})")
            if T2p <= 0:
                violations.append(f"Sample {i}, {met}: T2' <= 0 ({T2p:.4f})")
            if T2 < 10 or T2 > 1000:
                violations.append(f"Sample {i}, {met}: T2 out of range ({T2:.4f})")
            if T2p < 5 or T2p > 200:
                violations.append(f"Sample {i}, {met}: T2' out of range ({T2p:.4f})")
        
        # Check global parameters
        g = schema.get_global_param(theta[i], "linewidth")
        if g <= 0:
            violations.append(f"Sample {i}: g <= 0 ({g:.4f})")
    
    n_violations = len(violations)
    plausibility_ratio = 1.0 - (n_violations / (n_samples * (schema.n_metabolites * 3 + 1)))
    
    result = {
        "n_violations": n_violations,
        "plausibility_ratio": plausibility_ratio,
        "is_plausible": n_violations == 0
    }
    
    if return_violations:
        result["violations"] = violations[:10]  # Limit to first 10
    
    return result


def compute_all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    schema: Optional[ParameterSchema] = None
) -> Dict[str, any]:
    """
    Compute all metrics (MAE per group + plausibility).
    
    Args:
        pred: Predicted parameters
        target: Target parameters
        schema: Parameter schema
    
    Returns:
        Dictionary of all metrics
    """
    metrics = compute_mae_per_group(pred, target, schema=schema)
    plausibility = check_physical_plausibility(pred, schema=schema)
    metrics.update(plausibility)
    return metrics

