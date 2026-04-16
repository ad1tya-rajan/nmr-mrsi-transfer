"""
TODO: Implement the forward model EXACTLY according to the current parametric formulation.

Required model:
    s(n, TE) = exp(i * phi_TE) *
               sum_{m=1}^M [
                   C_m * phi_{m,TE}(n * dt)
                   * exp(-(TE + n*dt) / T2_m)
                   * exp(-(n*dt) / T2p_m)
                   * exp(-i * 2*pi * delta_f_m * n*dt)
                   * exp(-(n*dt)^2 * g_TE)
               ]

IMPORTANT:
- Use EXACTLY this model for now.
- Do NOT replace it with a simplified decay-only model.
- Do NOT collapse terms or reinterpret the parameterization.
- Do NOT introduce alternative architectures or different signal models yet.

Parameterization per metabolite:
    theta_m = {f, phi, sigma, T2, T2p, A}
which corresponds in our current code/data to:
    - concentration / amplitude  -> A or C_m
    - T2                        -> T2_m
    - T2p                       -> T2p_m
    - phase                     -> phi
    - freq_shift                -> delta_f_m
    - linewidth                 -> sigma / Gaussian linewidth term

Implementation requirements:
[1] Replace the placeholder signal generation in simulate_fid().
    - Remove the current simplified exponential placeholder.
    - Implement the exact multiplicative terms from the formula.

[2] Respect the current schema and raw data layout.
    - Current raw sample shape corresponds to 10 metabolites × 6 params.
    - Use schema-aware indexing for:
        concentration
        T2
        T2p
        phase
        freq_shift
        linewidth
    - Do NOT assume global phi/g unless explicitly derived from the model/data.
    - Match the actual naming convention used in the schema.

[3] Basis function handling.
    - Include phi_{m,TE}(n*dt) explicitly in the forward model.
    - If basis is provided, use metabolite-specific basis functions from it.
    - If basis is not yet available, leave a clearly marked placeholder for phi_{m,TE}(t),
      but keep the rest of the model exactly as written.

[4] Time axis.
    - Define t_n = n * dt for n = 0, ..., n_points-1.
    - Use TE explicitly in the exp(-(TE + t_n)/T2_m) term.
    - Use t_n explicitly in all other terms.

[5] Complex phase/frequency terms.
    - Global phase term:
        exp(i * phi_TE)
      should be applied exactly as written in the current model.
    - Frequency shift term:
        exp(-i * 2*pi * delta_f_m * t_n)

[6] Gaussian linewidth term.
    - Implement exactly as:
        exp(-(t_n**2) * g_TE)
    - Do NOT replace with linear exponential damping.

[7] T2 and T2p handling.
    - Implement both relaxation terms separately:
        exp(-(TE + t_n)/T2_m)
        exp(-t_n/T2p_m)
    - Do NOT merge them into a single decay term.

[8] Units.
    - Verify whether T2 and T2p are stored in seconds or milliseconds.
    - Convert consistently before simulation.
    - Verify freq_shift units are consistent with Hz.
    - Keep unit assumptions explicit in comments/docstring.

[9] Batch support.
    - simulate_fid() must support both:
        theta shape (D,)
        theta shape (N, D)
    - Output must preserve batch behavior:
        (n_points,) for single sample
        (N, n_points) for batch input

[10] Update fid_to_spectrum().
    - Remove hardcoded dt=0.0005 inside fid_to_spectrum().
    - Pass dt as an argument so the spectrum uses the same sampling interval as simulate_fid().
    - Keep FFT behavior unchanged unless explicitly needed.

[11] Keep preprocessing separate.
    - Forward model must take RAW physical parameters, not standardized/log/sin-cos features.
    - Any model outputs must be inverse-transformed before calling simulate_fid().

[12] Add tests after implementation.
    - shape correctness
    - no NaN / no Inf
    - zero amplitude -> zero contribution
    - larger freq_shift changes oscillation frequency
    - larger linewidth increases Gaussian damping
    - batch and single-sample consistency

Goal:
- Make forward_model.py physically consistent with the exact current parametric model,
  so it can later be used for reconstruction, validation, and physics-based checking
  of predicted parameter vectors.
"""

from typing import Optional, Tuple
import numpy as np
from ..utils.schema import default_schema, ParameterSchema


def simulate_fid(
    theta: np.ndarray,
    basis: Optional[np.ndarray] = None,
    te: float = 0.0,
    dt: float = 0.0005,  # 0.5 ms sampling interval
    n_points: int = 2048,
    schema: Optional[ParameterSchema] = None
) -> np.ndarray:
    """
    Generate synthetic FID signal from parameter vector using the exact parametric model.
    
    Model: s(n, TE) = exp(i * phi_TE) * sum_{m=1}^M [
        C_m * phi_{m,TE}(n * dt) * exp(-(TE + n*dt) / T2_m) * exp(-(n*dt) / T2p_m) *
        exp(-i * 2*pi * delta_f_m * n*dt) * exp(-(n*dt)^2 * g_TE)
    ]
    
    Args:
        theta: Parameter vector, shape (D,) or (n_samples, D) where D is 60 or 62
        basis: Basis set (optional, for future use with phi_{m,TE})
        te: Echo time (seconds)
        dt: Sampling interval (seconds)
        n_points: Number of time points
        schema: Parameter schema
    
    Returns:
        FID signal, shape (n_points,) or (n_samples, n_points)
    """
    if schema is None:
        schema = default_schema
    
    theta = np.asarray(theta)
    is_1d = theta.ndim == 1
    if is_1d:
        theta = theta[np.newaxis, :]
    
    n_samples, d = theta.shape
    
    # Handle both 60-dim (metabolite only) and 62-dim (with globals) input
    if d == 60:
        # Add default global parameters
        theta_full = np.zeros((n_samples, 62))
        theta_full[:, :60] = theta
        theta_full[:, 60] = 0.0  # phi (global phase)
        theta_full[:, 61] = 0.0  # linewidth (global linewidth)
        theta = theta_full
    elif d != 62:
        raise ValueError(f"Parameter vector dimension {d} not supported. Expected 60 or 62.")
    
    t = np.arange(n_points) * dt
    
    fid = np.zeros((n_samples, n_points), dtype=complex)
    
    for i in range(n_samples):
        # Global parameters
        phi_TE = schema.get_global_param(theta[i], "phi")
        g_TE = schema.get_global_param(theta[i], "linewidth")
        
        signal = np.zeros(n_points, dtype=complex)
        
        for met in schema.metabolites:
            # Metabolite-specific parameters
            C_m = schema.get_param(theta[i], met, "concentration")
            T2_m = max(schema.get_param(theta[i], met, "T2") / 1000.0, 1e-6)  # Convert ms to s, avoid zero
            T2p_m = max(schema.get_param(theta[i], met, "T2p") / 1000.0, 1e-6)
            delta_f_m = schema.get_param(theta[i], met, "freq_shift")
            
            # Basis function placeholder: phi_{m,TE}(t) = 1 for now
            phi_m_TE_t = 1.0
            
            # Compute the metabolite contribution
            term = (
                C_m * phi_m_TE_t *
                np.exp(-(te + t) / T2_m) *
                np.exp(-t / T2p_m) *
                np.exp(-1j * 2 * np.pi * delta_f_m * t) *
                np.exp(-t**2 * g_TE)
            )
            signal += term
        
        # Apply global phase
        fid[i] = np.exp(1j * phi_TE) * signal
    
    if is_1d:
        fid = fid[0]
    
    return fid


def fid_to_spectrum(
    fid: np.ndarray,
    dt: float = 0.0005,
    apply_apodization: bool = True,
    apodization_factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert FID signal to frequency domain spectrum.
    
    Args:
        fid: FID signal, shape (n_points,) or (n_samples, n_points)
        dt: Sampling interval (seconds)
        apply_apodization: Whether to apply exponential apodization
        apodization_factor: Apodization line broadening factor (Hz)
    
    Returns:
        Tuple of (frequencies, spectrum)
        - frequencies: Frequency array in Hz
        - spectrum: Complex spectrum, shape matching fid
    """
    fid = np.asarray(fid)
    is_1d = fid.ndim == 1
    if is_1d:
        fid = fid[np.newaxis, :]
    
    n_samples, n_points = fid.shape
    
    # Apply apodization if requested
    if apply_apodization:
        t = np.arange(n_points) * dt
        apodization = np.exp(-t * apodization_factor * np.pi)
        fid = fid * apodization[np.newaxis, :]
    
    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1)
    
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(n_points, dt))
    
    if is_1d:
        spectrum = spectrum[0]
    
    return freqs, spectrum

