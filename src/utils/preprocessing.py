"""
Preprocessing utilities for NMR-MRSI training data.

Handles transformation from raw MATLAB pars arrays to model-ready features.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler

from .schema import default_schema


def normalize_pars_shape(pars: np.ndarray) -> np.ndarray:
    """
    Normalize raw pars arrays to shape (N, 10, 6).

    Supports common layouts such as:
    - (N, 10, 6)
    - (6, 10, N)
    - (10, 6, N)
    - (N, 6, 10)
    - (10, N, 6)
    - (6, N, 10)
    """
    if pars.ndim != 3:
        raise ValueError(f"Expected raw pars with 3 dimensions, got {pars.ndim}D")

    if pars.shape[1:] == (default_schema.n_metabolites, default_schema.n_metabolite_params):
        return pars

    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        candidate = np.transpose(pars, perm)
        if candidate.shape[1:] == (default_schema.n_metabolites, default_schema.n_metabolite_params):
            return candidate

    raise ValueError(
        f"Could not normalize pars shape {pars.shape} to (N, {default_schema.n_metabolites}, {default_schema.n_metabolite_params})"
    )


class NMRPreprocessor:
    """
    Preprocessing pipeline for NMR-MRSI data.

    Transforms raw pars arrays (N, 10, 6) to standardized features (N, 70)
    and recovers raw pars from standardized features.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.schema = default_schema
        self.log_eps = 1e-6

    def _log_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply stable log transform to non-negative inputs."""
        x = np.asarray(x, dtype=np.float64)
        if np.any(x < 0):
            x = np.clip(x, 0.0, None)
        return np.log(x + self.log_eps)

    def _feature_batch(self, pars_batch: np.ndarray) -> np.ndarray:
        """
        Convert raw pars array to feature matrix before standardization.

        Args:
            pars_batch: Raw pars of shape (N, 10, 6)

        Returns:
            Array of shape (N, 70)
        """
        pars_batch = np.asarray(pars_batch, dtype=np.float64)
        if pars_batch.ndim != 3:
            raise ValueError(f"Expected 3D array, got {pars_batch.ndim}D")
        if pars_batch.shape[1:] != (self.schema.n_metabolites, self.schema.n_metabolite_params):
            raise ValueError(
                f"Expected raw pars shape (N, {self.schema.n_metabolites}, {self.schema.n_metabolite_params}), got {pars_batch.shape}"
            )

        log_concentration = self._log_transform(pars_batch[..., 0])
        log_T2 = self._log_transform(pars_batch[..., 1])
        log_T2p = self._log_transform(pars_batch[..., 2])
        phase = pars_batch[..., 3]
        freq_shift = pars_batch[..., 4]
        log_linewidth = self._log_transform(pars_batch[..., 5])

        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)

        features = np.stack(
            [
                log_concentration,
                log_T2,
                log_T2p,
                freq_shift,
                log_linewidth,
                sin_phase,
                cos_phase,
            ],
            axis=-1,
        )

        return features.reshape(pars_batch.shape[0], -1)

    def transform_sample(self, pars: np.ndarray) -> np.ndarray:
        """
        Transform a single sample from raw pars to standardized features.

        Args:
            pars: Raw parameter array of shape (10, 6)

        Returns:
            features: Array of shape (70,)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform_sample")

        pars = np.asarray(pars, dtype=np.float64)
        if pars.ndim != 2:
            raise ValueError(f"Expected 2D sample, got {pars.ndim}D")

        features = self._feature_batch(pars[np.newaxis, ...])
        return self.scaler.transform(features)[0]

    def fit(self, pars_batch: np.ndarray) -> 'NMRPreprocessor':
        """
        Fit the preprocessor on a batch of raw pars data.
        """
        pars_batch = np.asarray(pars_batch, dtype=np.float64)
        features_batch = self._feature_batch(pars_batch)
        self.scaler.fit(features_batch)
        self.is_fitted = True
        return self

    def transform(self, pars_batch: np.ndarray) -> np.ndarray:
        """
        Transform a batch of raw pars data to standardized features.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        pars_batch = np.asarray(pars_batch, dtype=np.float64)
        features_batch = self._feature_batch(pars_batch)
        return self.scaler.transform(features_batch)

    def fit_transform(self, pars_batch: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        return self.fit(pars_batch).transform(pars_batch)

    def inverse_transform(self, features_batch: np.ndarray) -> np.ndarray:
        """
        Inverse transform standardized features back to raw pars.

        Returns:
            Raw parameter array of shape (N, 10, 6)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")

        features = np.asarray(features_batch, dtype=np.float64)
        if features.ndim == 1:
            features = features[np.newaxis, :]
        if features.ndim != 2 or features.shape[1] != 70:
            raise ValueError(f"Expected features shape (N, 70), got {features.shape}")

        unscaled = self.scaler.inverse_transform(features)
        metabolite_features = unscaled.reshape(-1, self.schema.n_metabolites, 7)

        log_concentration = metabolite_features[..., 0]
        log_T2 = metabolite_features[..., 1]
        log_T2p = metabolite_features[..., 2]
        freq_shift = metabolite_features[..., 3]
        log_linewidth = metabolite_features[..., 4]
        sin_phase = metabolite_features[..., 5]
        cos_phase = metabolite_features[..., 6]

        phase = np.arctan2(sin_phase, cos_phase)

        concentration = np.exp(log_concentration) - self.log_eps
        T2 = np.exp(log_T2) - self.log_eps
        T2p = np.exp(log_T2p) - self.log_eps
        linewidth = np.exp(log_linewidth) - self.log_eps

        concentration = np.clip(concentration, 0.0, None)
        T2 = np.clip(T2, 0.0, None)
        T2p = np.clip(T2p, 0.0, None)
        linewidth = np.clip(linewidth, 0.0, None)

        raw_pars = np.stack(
            [
                concentration,
                T2,
                T2p,
                phase,
                freq_shift,
                linewidth,
            ],
            axis=-1,
        )

        return raw_pars

    def get_feature_names(self) -> list:
        """
        Get names of the 70 features.
        """
        feature_names = []
        for met in self.schema.metabolites:
            feature_names.extend([
                f"{met}_log_concentration",
                f"{met}_log_T2",
                f"{met}_log_T2p",
                f"{met}_freq_shift",
                f"{met}_log_linewidth",
                f"{met}_sin_phase",
                f"{met}_cos_phase",
            ])
        return feature_names


def create_preprocessor() -> NMRPreprocessor:
    """Factory function to create a configured NMR preprocessor."""
    return NMRPreprocessor()


def preprocess_pars_batch(pars_batch: np.ndarray) -> Tuple[np.ndarray, NMRPreprocessor]:
    """
    Preprocess a batch of pars data and return both features and fitted preprocessor.
    """
    preprocessor = NMRPreprocessor()
    features = preprocessor.fit_transform(pars_batch)
    return features, preprocessor


def load_and_preprocess_mat_file(mat_file_path: str, sample_limit: Optional[int] = None) -> Tuple[np.ndarray, NMRPreprocessor]:
    """
    Load a MATLAB file and preprocess its pars data.
    """
    import h5py

    with h5py.File(mat_file_path, 'r') as f:
        pars = np.array(f['pars'])

    pars = normalize_pars_shape(pars)

    if sample_limit is not None:
        pars = pars[:sample_limit]

    return preprocess_pars_batch(pars)
