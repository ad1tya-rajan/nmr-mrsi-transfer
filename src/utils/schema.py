"""
Parameter schema for metabolite-specific and global parameters.

This module defines the canonical parameter vector format, providing
a contract that prevents constant refactors later.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# Ordered list of metabolites
METABOLITES = [
    "NAA",      # N-acetylaspartate
    "Cr",       # Creatine
    "Cho",      # Choline
    "Glu",      # Glutamate
    "Gln",      # Glutamine
    "Ins",      # Myo-inositol
    "Lac",      # Lactate
    "Ala",      # Alanine
    "Tau",      # Taurine
    "Asp",      # Aspartate
    "GABA",     # Gamma-aminobutyric acid
    "GSH",      # Glutathione
    "PE",       # Phosphoethanolamine
    "PCr",      # Phosphocreatine
    "GPC",      # Glycerophosphocholine
    "PCh",      # Phosphocholine
]

# Per-metabolite parameter names
METABOLITE_PARAMS = [
    "Cm",       # Concentration/amplitude
    "T2",       # Transverse relaxation time
    "T2_prime", # T2' (additional relaxation)
    "delta_f",  # Frequency shift (Hz)
]

# Global parameter names
GLOBAL_PARAMS = [
    "phi",      # Global phase (radians)
    "g",        # Global linewidth/broadening factor
]


@dataclass
class ParameterSchema:
    """Schema defining the parameter vector structure."""
    
    metabolites: List[str]
    metabolite_params: List[str]
    global_params: List[str]
    
    def __post_init__(self):
        """Validate and compute dimensions."""
        self.n_metabolites = len(self.metabolites)
        self.n_metabolite_params = len(self.metabolite_params)
        self.n_global_params = len(self.global_params)
        self.n_per_metabolite = self.n_metabolite_params
        self.total_dim = (
            self.n_metabolites * self.n_metabolite_params + 
            self.n_global_params
        )
    
    def get_metabolite_idx(self, metabolite: str) -> int:
        """Get index of a metabolite in the ordered list."""
        if metabolite not in self.metabolites:
            raise ValueError(f"Unknown metabolite: {metabolite}")
        return self.metabolites.index(metabolite)
    
    def get_param_idx(self, param_name: str) -> int:
        """Get index of a parameter within metabolite params."""
        if param_name not in self.metabolite_params:
            raise ValueError(f"Unknown metabolite parameter: {param_name}")
        return self.metabolite_params.index(param_name)
    
    def get_global_param_idx(self, param_name: str) -> int:
        """Get index of a global parameter."""
        if param_name not in self.global_params:
            raise ValueError(f"Unknown global parameter: {param_name}")
        return self.global_params.index(param_name)
    
    def get_idx(self, metabolite: str, param_name: Optional[str] = None) -> int:
        """
        Get the index in the parameter vector for a metabolite parameter.
        
        Args:
            metabolite: Metabolite name
            param_name: Parameter name (e.g., "T2", "Cm"). If None, returns
                       the starting index for the metabolite.
        
        Returns:
            Index in the flattened parameter vector.
        """
        met_idx = self.get_metabolite_idx(metabolite)
        base_idx = met_idx * self.n_metabolite_params
        
        if param_name is None:
            return base_idx
        
        param_idx = self.get_param_idx(param_name)
        return base_idx + param_idx
    
    def get_global_idx(self, param_name: str) -> int:
        """
        Get the index in the parameter vector for a global parameter.
        
        Args:
            param_name: Global parameter name (e.g., "phi", "g")
        
        Returns:
            Index in the flattened parameter vector.
        """
        global_idx = self.get_global_param_idx(param_name)
        return self.n_metabolites * self.n_metabolite_params + global_idx
    
    def slice_metabolite(self, theta: List[float], metabolite: str) -> List[float]:
        """Extract all parameters for a specific metabolite."""
        start_idx = self.get_idx(metabolite)
        end_idx = start_idx + self.n_metabolite_params
        return theta[start_idx:end_idx]
    
    def slice_global(self, theta: List[float]) -> List[float]:
        """Extract all global parameters."""
        start_idx = self.n_metabolites * self.n_metabolite_params
        return theta[start_idx:]
    
    def get_param(self, theta: List[float], metabolite: str, param_name: str) -> float:
        """Get a specific metabolite parameter value."""
        idx = self.get_idx(metabolite, param_name)
        return theta[idx]
    
    def get_global_param(self, theta: List[float], param_name: str) -> float:
        """Get a specific global parameter value."""
        idx = self.get_global_idx(param_name)
        return theta[idx]
    
    def set_param(self, theta: List[float], metabolite: str, param_name: str, value: float) -> None:
        """Set a specific metabolite parameter value (in-place)."""
        idx = self.get_idx(metabolite, param_name)
        theta[idx] = value
    
    def set_global_param(self, theta: List[float], param_name: str, value: float) -> None:
        """Set a specific global parameter value (in-place)."""
        idx = self.get_global_idx(param_name)
        theta[idx] = value


# Default schema instance
default_schema = ParameterSchema(
    metabolites=METABOLITES,
    metabolite_params=METABOLITE_PARAMS,
    global_params=GLOBAL_PARAMS,
)


def get_dim(schema: Optional[ParameterSchema] = None) -> int:
    """
    Compute the total dimension D of the parameter vector.
    
    Args:
        schema: Parameter schema (defaults to default_schema)
    
    Returns:
        Total dimension D
    """
    if schema is None:
        schema = default_schema
    return schema.total_dim

