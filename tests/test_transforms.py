"""
Tests for the new NMR preprocessing pipeline.

This file validates raw shape normalization, feature extraction,
standardization, and inverse reconstruction using the current schema.
"""

import numpy as np
import pytest

from src.utils.preprocessing import NMRPreprocessor, normalize_pars_shape
from src.utils.schema import default_schema


def make_random_raw_pars(n_samples: int = 4, random_seed: int = 0) -> np.ndarray:
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
    raw = np.arange(2 * default_schema.n_metabolites * default_schema.n_metabolite_params)
    raw = raw.reshape(2, default_schema.n_metabolites, default_schema.n_metabolite_params).astype(float)
    permuted = np.transpose(raw, (2, 1, 0))

    normalized = normalize_pars_shape(permuted)

    assert normalized.shape == (2, default_schema.n_metabolites, default_schema.n_metabolite_params)
    assert np.array_equal(normalized, raw)


def test_preprocessor_feature_shape_and_names():
    """Preprocessor returns 70 features for a batch and exposes names."""
    pars_batch = make_random_raw_pars(n_samples=3)
    preprocessor = NMRPreprocessor().fit(pars_batch)

    features = preprocessor.transform(pars_batch)
    feature_names = preprocessor.get_feature_names()

    assert features.shape == (3, 70)
    assert len(feature_names) == 70
    assert feature_names[0].endswith("_log_concentration")
    assert feature_names[-1].endswith("_cos_phase")


def test_preprocessor_inverse_transform_round_trip():
    """Inverse transform should recover the original raw pars within tolerance."""
    pars_batch = make_random_raw_pars(n_samples=5, random_seed=1)
    preprocessor = NMRPreprocessor().fit(pars_batch)

    features = preprocessor.transform(pars_batch)
    recovered = preprocessor.inverse_transform(features)

    assert recovered.shape == pars_batch.shape
    assert np.all(recovered[..., 0] >= 0)
    assert np.allclose(np.sin(recovered[..., 3]), np.sin(pars_batch[..., 3]), atol=1e-6)
    assert np.mean(np.abs(recovered - pars_batch)) < 1e-5


def test_transform_sample_works_for_single_sample():
    """transform_sample should match batch transform for one sample."""
    sample = make_random_raw_pars(n_samples=1, random_seed=2)[0]
    preprocessor = NMRPreprocessor().fit(sample[np.newaxis, ...])

    sample_features = preprocessor.transform_sample(sample)
    batch_features = preprocessor.transform(sample[np.newaxis, ...])[0]

    assert sample_features.shape == (70,)
    assert np.allclose(sample_features, batch_features, atol=1e-8)


def test_invalid_shapes_raise_value_error():
    """Invalid raw shapes should raise a ValueError in preprocessing helpers."""
    preprocessor = NMRPreprocessor().fit(make_random_raw_pars(n_samples=1))

    with pytest.raises(ValueError):
        preprocessor.transform_sample(np.zeros((10, 5)))

    with pytest.raises(ValueError):
        normalize_pars_shape(np.zeros((10, 6)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

