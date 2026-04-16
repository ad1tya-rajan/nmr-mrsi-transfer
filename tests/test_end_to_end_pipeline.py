"""End-to-end pipeline test: real raw data -> preprocessing -> inverse -> physics simulation."""

import numpy as np
import h5py
import pytest
from pathlib import Path

from src.utils.preprocessing import load_and_preprocess_mat_file, normalize_pars_shape
from src.simulation.forward_model import simulate_fid, fid_to_spectrum


BOX_DATA_DIR = Path("~/Library/CloudStorage/Box-Box/NMR-MRSI-Data/trainingData").expanduser()
REAL_MAT_FILE = BOX_DATA_DIR / "trdata_141T_dt160us.mat"


@pytest.mark.skipif(
    not REAL_MAT_FILE.exists(),
    reason="Real Box .mat file not found"
)
def test_end_to_end_pipeline_real_data():
    print("Starting real-data end-to-end pipeline test...")
    print(f"Using file: {REAL_MAT_FILE}")

    # 1. Load and preprocess a small real subset
    features, preprocessor = load_and_preprocess_mat_file(str(REAL_MAT_FILE), sample_limit=3)
    print(f"Features shape: {features.shape}")
    assert features.shape == (3, 70)

    # 2. Inverse transform back to raw parameters
    recovered_pars = preprocessor.inverse_transform(features)
    print(f"Recovered pars shape: {recovered_pars.shape}")
    assert recovered_pars.shape == (3, 10, 6)

    # 3. Load original raw subset for comparison
    with h5py.File(REAL_MAT_FILE, "r") as f:
        original_pars = normalize_pars_shape(np.array(f["pars"]))

    original_subset = original_pars[:3]

    mae = np.mean(np.abs(recovered_pars - original_subset))
    print(f"Reconstruction MAE: {mae:.2e}")
    assert mae < 1e-5

    # 4. Flatten raw recovered parameters for forward model (60-dim metabolite only)
    theta = recovered_pars.reshape(recovered_pars.shape[0], -1)
    print(f"Theta shape for forward model: {theta.shape}")
    assert theta.shape == (3, 60)

    # 5. Simulate FID
    fid = simulate_fid(theta)
    print(f"FID shape: {fid.shape}")
    assert fid.shape == (3, 2048)
    assert np.iscomplexobj(fid)

    # 6. Convert to spectrum
    freqs, spectrum = fid_to_spectrum(fid)
    print(f"Spectrum shape: {spectrum.shape}")
    assert spectrum.shape == (3, 2048)
    assert freqs.shape == (2048,)

    # 7. Basic numerical checks
    assert not np.isnan(fid).any()
    assert not np.isinf(fid).any()
    assert not np.isnan(spectrum).any()
    assert not np.isinf(spectrum).any()

    print("Real-data pipeline test passed.")