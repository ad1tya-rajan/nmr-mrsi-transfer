"""
Parameter schema for metabolite-specific parameters from MATLAB training data.

This schema is built directly from the MATLAB .mat files, using the
trdata_xxT_dtxxxus.Name field when available.

Expected MATLAB pars layout:
    pars.shape = (6, 10, N)

Parameter order:
    0: concentration
    1: T2
    2: T2p
    3: phase
    4: freq_shift
    5: linewidth
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
from scipy.io import loadmat
import h5py


# Per-metabolite parameter names (matches MATLAB pars array)
METABOLITE_PARAMS = [
    "concentration",
    "T2",
    "T2p",
    "phase",
    "freq_shift",
    "linewidth",
]

# Global parameters for the forward model
GLOBAL_PARAMS: List[str] = ["phi", "linewidth"]


@dataclass
class ParameterSchema:
    """Schema defining the flattened parameter vector structure."""

    metabolites: List[str]
    metabolite_params: List[str]
    global_params: List[str]

    def __post_init__(self):
        self.n_metabolites = len(self.metabolites)
        self.n_metabolite_params = len(self.metabolite_params)
        self.n_global_params = len(self.global_params)
        self.n_per_metabolite = self.n_metabolite_params
        self.total_dim = (
            self.n_metabolites * self.n_metabolite_params
            + self.n_global_params
        )

    def get_metabolite_idx(self, metabolite: str) -> int:
        if metabolite not in self.metabolites:
            raise ValueError(f"Unknown metabolite: {metabolite}")
        return self.metabolites.index(metabolite)

    def get_param_idx(self, param_name: str) -> int:
        if param_name not in self.metabolite_params:
            raise ValueError(f"Unknown metabolite parameter: {param_name}")
        return self.metabolite_params.index(param_name)

    def get_idx(self, metabolite: str, param_name: Optional[str] = None) -> int:
        met_idx = self.get_metabolite_idx(metabolite)
        base_idx = met_idx * self.n_metabolite_params
        if param_name is None:
            return base_idx
        param_idx = self.get_param_idx(param_name)
        return base_idx + param_idx

    def slice_metabolite(self, theta, metabolite: str):
        start = self.get_idx(metabolite)
        end = start + self.n_metabolite_params
        return theta[start:end]

    def get_param(self, theta, metabolite: str, param_name: str):
        return theta[self.get_idx(metabolite, param_name)]

    def set_param(self, theta, metabolite: str, param_name: str, value: float):
        theta[self.get_idx(metabolite, param_name)] = value

    def get_global_param_idx(self, param_name: str) -> int:
        if param_name not in self.global_params:
            raise ValueError(f"Unknown global parameter: {param_name}")
        base_idx = self.n_metabolites * self.n_metabolite_params
        return base_idx + self.global_params.index(param_name)

    def get_global_param(self, theta, param_name: str):
        return theta[self.get_global_param_idx(param_name)]

    def set_global_param(self, theta, param_name: str, value: float):
        theta[self.get_global_param_idx(param_name)] = value


def _decode_matlab_string_array(arr) -> List[str]:
    """
    Convert common MATLAB string/cellstr representations into a Python list[str].
    """
    arr = np.array(arr)

    # Object/cell arrays
    if arr.dtype == object:
        out = []
        for x in arr.ravel():
            if isinstance(x, np.ndarray):
                if x.dtype.kind in {"U", "S"}:
                    out.append("".join(x.tolist()).strip())
                else:
                    out.append(str(np.array(x).squeeze()).strip())
            else:
                out.append(str(x).strip())
        return out

    # Char arrays
    if arr.dtype.kind in {"U", "S"}:
        if arr.ndim == 2:
            return ["".join(row).strip() for row in arr]
        return ["".join(arr.tolist()).strip()]

    return [str(x).strip() for x in arr.ravel()]


def load_metabolite_names_from_mat(filepath: str) -> List[str]:
    """
    Load metabolite names from a MATLAB file.

    Looks for a field named 'Name' first, since your data uses
    trdata_xxT_dtxxxus.Name.
    """
    try:
        mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)

        # Direct field
        if "Name" in mat:
            names = _decode_matlab_string_array(mat["Name"])
            if names:
                return names

        # Look through structs
        for value in mat.values():
            if hasattr(value, "Name"):
                names = _decode_matlab_string_array(value.Name)
                if names:
                    return names

    except NotImplementedError:
        # MATLAB v7.3 / HDF5
        with h5py.File(filepath, "r") as f:
            if "Name" in f:
                raw = np.array(f["Name"])
                names = _decode_matlab_string_array(raw)
                if names:
                    return names

    raise ValueError(
        f"Could not find metabolite names in {filepath}. "
        "Please inspect the file structure and confirm where the Name field is stored."
    )


def create_schema_from_mat(filepath: str) -> ParameterSchema:
    """
    Build a schema directly from the MATLAB file metadata.
    """
    metabolites = load_metabolite_names_from_mat(filepath)
    return ParameterSchema(
        metabolites=metabolites,
        metabolite_params=METABOLITE_PARAMS,
        global_params=GLOBAL_PARAMS,
    )


# Optional fallback if you want something usable before metadata loading
default_schema = ParameterSchema(
    metabolites=[f"met_{i}" for i in range(1, 11)],
    metabolite_params=METABOLITE_PARAMS,
    global_params=GLOBAL_PARAMS,
)


def get_dim(schema: Optional[ParameterSchema] = None) -> int:
    if schema is None:
        schema = default_schema
    return schema.total_dim