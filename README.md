# Physics-Guided Cross-Instrument MRS Parameter Transfer

This project develops a physics-informed neural network to learn cross-scanner transfer functions between high-field NMR and 9.4T MRSI parameter spaces.

Rather than mapping raw spectra, we operate in a structured parametric domain representing metabolite-specific amplitudes, relaxation constants, frequency shifts, and linewidth parameters. The objective is to learn a low-dimensional instrument transfer operator that enables scanner-invariant spectral modeling.

---

## Problem Statement

![NMR_vs_MRSI](assets/spectra.png)

High-field NMR (left) and in vivo 9.4T MRSI (right) measure the same metabolites but produce domain-shifted spectral representations due to:

- Field strength differences  
- Relaxation and linewidth variations  
- Magnetic field inhomogeneity  
- Voxel size and hardware effects  

Direct spectrum-to-spectrum translation is data-intensive and poorly constrained.

We instead:

1. Represent signals using a physics-based parametric forward model.
2. Learn a neural network mapping between parameter spaces.
3. Validate predictions via forward synthesis.

---

## Method Overview

Pipeline:

1. **Parameter Sampling**
   - Sample biologically realistic metabolite parameters.
2. **Forward Simulation**
   - Generate synthetic FID signals via parametric model.
3. **Paired Dataset Construction**
   - Create paired parameter sets: θ_NMR → θ_9.4T.
4. **Neural Network Training**
   - Train residual MLP regression model.
5. **Physics-Based Validation**
   - Reconstruct signals from predicted parameters.

---
---

## Installation

```bash
git clone https://github.com/yourusername/nmr-mrsi-transfer.git
cd nmr-mrsi-transfer
pip install -r requirements.txt

