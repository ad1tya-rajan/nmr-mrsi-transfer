"""Tests for the preprocessing pipeline.

This file validates raw pars shape normalization, feature extraction,
standardization, inverse reconstruction, and batch / single-sample consistency.
"""

import numpy as np
import pytest

from src.utils.preprocessing import NMRPreprocessor, normalize_pars_shape
from src.utils.schema import default_schema


def make_random_raw_pars(n_samples: int = 8, random_seed: int = 0) -> np.ndarray:
    """Build a small raw pars batch compatible with the default schema."""
    rng = np.random.default_rng(random_seed)
    n_met = default_schema.n_metabolites

    concentration = rng.uniform(0.01, 10.0, size=(n_samples, n_met))
    T2 = rng.uniform(0.01, 2.0, size=(n_samples, n_met))
    T2p = rng.uniform(0.01, 2.0, size=(n_samples, n_met))
    phase = rng.uniform(-np.pi, np.pi, size=(n_samples, n_met))
    freq_shift = rng.normal(loc=0.0, scale=3.0, size=(n_samples, n_met))
    linewidth = rng.uniform(0.001, 3.0, size=(n_samples, n_met))

    return np.stack(
        [concentration, T2, T2p, phase, freq_shift, linewidth],
        axis=-1,
    )


def test_normalize_pars_shape_permutations():
    """normalize_pars_shape should convert common permutations to (N, 10, 6)."""
    raw = np.arange(2 * default_schema.n_metabolites * default_schema.n_metabolite_params, dtype=float)
    raw = raw.reshape(2, default_schema.n_metabolites, default_schema.n_metabolite_params)
    permuted = np.transpose(raw, (2, 1, 0))

    normalized = normalize_pars_shape(permuted)

    assert normalized.shape == (2, default_schema.n_metabolites, default_schema.n_metabolite_params)
    assert np.array_equal(normalized, raw)


def test_preprocessor_shape_and_standardization():
    """fit_transform returns (N, 70) and standardized features."""
    pars_batch = make_random_raw_pars(n_samples=20)
    preprocessor = NMRPreprocessor().fit(pars_batch)

    features = preprocessor.transform(pars_batch)

    assert features.shape == (20, 70)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0, ddof=0)

    assert np.allclose(means, 0.0, atol=1e-7)
    assert np.allclose(stds, 1.0, atol=1e-7)


def test_feature_name_consistency():
    """get_feature_names should return 70 ordered feature labels."""
    preprocessor = NMRPreprocessor()
    names = preprocessor.get_feature_names()

    assert len(names) == 70
    assert names[0].endswith("_log_concentration")
    assert names[1].endswith("_log_T2")
    assert names[2].endswith("_log_T2p")
    assert names[3].endswith("_freq_shift")
    assert names[4].endswith("_log_linewidth")
    assert names[5].endswith("_sin_phase")
    assert names[6].endswith("_cos_phase")
    assert names[-1].endswith("_cos_phase")


def test_transform_sample_manual_feature_verification():
    """transform_sample should apply the same raw feature mapping as manual computation."""
    pars = make_random_raw_pars(n_samples=1, random_seed=7)[0]
    preprocessor = NMRPreprocessor().fit(pars[np.newaxis, ...])

    sample_features = preprocessor.transform_sample(pars)
    assert sample_features.shape == (70,)

    raw_features = preprocessor._feature_batch(pars[np.newaxis, ...])[0]

    cm, t2, t2p, phi, df, g = pars[0]
    eps = preprocessor.log_eps
    expected_first_met = np.array(
        [
            np.log(cm + eps),
            np.log(t2 + eps),
            np.log(t2p + eps),
            df,
            np.log(g + eps),
            np.sin(phi),
            np.cos(phi),
        ],
        dtype=float,
    )

    actual_first_met = raw_features[:7]
    assert np.allclose(actual_first_met, expected_first_met, atol=1e-9)
    assert np.all(actual_first_met[5:] >= -1.0)
    assert np.all(actual_first_met[5:] <= 1.0)


def test_inverse_transform_round_trip():
    """Inverse transform should recover the original raw pars within tolerance."""
    pars_batch = make_random_raw_pars(n_samples=12, random_seed=3)
    preprocessor = NMRPreprocessor().fit(pars_batch)

    features = preprocessor.transform(pars_batch)
    recovered = preprocessor.inverse_transform(features)

    assert recovered.shape == pars_batch.shape
    assert not np.isnan(recovered).any()
    assert not np.isinf(recovered).any()

    max_error = np.max(np.abs(recovered - pars_batch))
    mean_error = np.mean(np.abs(recovered - pars_batch))

    assert mean_error < 1e-5, f"Mean reconstruction error too large: {mean_error}"
    assert max_error < 1e-4, f"Max reconstruction error too large: {max_error}"


def test_batch_vs_single_consistency():
    """Batch transform should match stacked single-sample transforms."""
    pars_batch = make_random_raw_pars(n_samples=5, random_seed=11)
    preprocessor = NMRPreprocessor().fit(pars_batch)

    batch_features = preprocessor.transform(pars_batch)
    sample_features = np.stack([preprocessor.transform_sample(sample) for sample in pars_batch], axis=0)

    assert batch_features.shape == sample_features.shape
    assert np.allclose(batch_features, sample_features, atol=1e-8)


def test_invalid_shapes_raise_value_error():
    """Invalid raw shapes should raise a ValueError in preprocessing helpers."""
    preprocessor = NMRPreprocessor().fit(make_random_raw_pars(n_samples=1))

    with pytest.raises(ValueError):
        preprocessor.transform_sample(np.zeros((10, 5), dtype=float))

    with pytest.raises(ValueError):
        normalize_pars_shape(np.zeros((10, 6), dtype=float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
