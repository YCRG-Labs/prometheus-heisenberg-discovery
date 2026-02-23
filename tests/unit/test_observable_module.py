"""Unit tests for Observable Computation Module"""

import pytest
import numpy as np
import pandas as pd
from src.observable_module import (
    Observable,
    Energy,
    EnergyDensity,
    StaggeredMagnetization,
    StripeOrder,
    PlaquetteOrder,
    StructureFactor,
    EntanglementEntropy,
    NematicOrder,
    DimerOrder,
    ObservableModule
)
from src.ed_module import J1J2Hamiltonian, GroundState
from src.config import Config


class TestObservableBase:
    """Test Observable base class"""
    
    def test_abstract_base(self):
        """Test that Observable is abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            Observable()


class TestEnergy:
    """Test Energy observable"""
    
    def test_compute(self):
        """Test energy computation returns ground state energy"""
        # Create a simple ground state
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        energy_obs = Energy()
        energy_value = energy_obs.compute(state)
        
        assert isinstance(energy_value, float)
        assert energy_value == state.energy
        assert np.isfinite(energy_value)


class TestEnergyDensity:
    """Test EnergyDensity observable"""
    
    def test_compute(self):
        """Test energy density computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        energy_density_obs = EnergyDensity()
        energy_density = energy_density_obs.compute(state)
        
        assert isinstance(energy_density, float)
        assert energy_density == state.energy / 16  # L=4, N=16
        assert np.isfinite(energy_density)


class TestStaggeredMagnetization:
    """Test StaggeredMagnetization observable"""
    
    def test_compute(self):
        """Test staggered magnetization computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        stag_mag_obs = StaggeredMagnetization()
        stag_mag = stag_mag_obs.compute(state)
        
        assert isinstance(stag_mag, float)
        assert np.isfinite(stag_mag)
        # Staggered magnetization should be non-negative
        assert stag_mag >= 0.0
    
    def test_neel_regime(self):
        """Test staggered magnetization in Néel regime (J2=0)"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.0)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        stag_mag_obs = StaggeredMagnetization()
        stag_mag = stag_mag_obs.compute(state)
        
        # In Néel regime, staggered magnetization should be significant
        assert stag_mag > 0.1


class TestStripeOrder:
    """Test StripeOrder observable"""
    
    def test_compute(self):
        """Test stripe order computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        stripe_obs = StripeOrder()
        stripe_order = stripe_obs.compute(state)
        
        assert isinstance(stripe_order, float)
        assert np.isfinite(stripe_order)
        assert stripe_order >= 0.0


class TestPlaquetteOrder:
    """Test PlaquetteOrder observable"""
    
    def test_compute(self):
        """Test plaquette order computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        plaq_obs = PlaquetteOrder()
        plaq_order = plaq_obs.compute(state)
        
        assert isinstance(plaq_order, float)
        assert np.isfinite(plaq_order)


class TestStructureFactor:
    """Test StructureFactor observable"""
    
    def test_compute_pi_pi(self):
        """Test structure factor at (π, π)"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        sf_obs = StructureFactor((np.pi, np.pi))
        sf_value = sf_obs.compute(state)
        
        assert isinstance(sf_value, float)
        assert np.isfinite(sf_value)
    
    def test_compute_pi_0(self):
        """Test structure factor at (π, 0)"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        sf_obs = StructureFactor((np.pi, 0.0))
        sf_value = sf_obs.compute(state)
        
        assert isinstance(sf_value, float)
        assert np.isfinite(sf_value)
    
    def test_name_includes_wavevector(self):
        """Test that observable name includes wavevector"""
        sf_obs = StructureFactor((np.pi, np.pi))
        assert "StructureFactor" in sf_obs.name
        assert "3.14" in sf_obs.name  # π ≈ 3.14


class TestEntanglementEntropy:
    """Test EntanglementEntropy observable"""
    
    def test_compute(self):
        """Test entanglement entropy computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        ee_obs = EntanglementEntropy()
        ee_value = ee_obs.compute(state)
        
        assert isinstance(ee_value, float)
        # Entanglement entropy should be non-negative
        assert ee_value >= 0.0


class TestNematicOrder:
    """Test NematicOrder observable"""
    
    def test_compute(self):
        """Test nematic order computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        nematic_obs = NematicOrder()
        nematic_value = nematic_obs.compute(state)
        
        assert isinstance(nematic_value, float)
        assert np.isfinite(nematic_value)
        assert nematic_value >= 0.0


class TestDimerOrder:
    """Test DimerOrder observable"""
    
    def test_compute(self):
        """Test dimer order computation"""
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        dimer_obs = DimerOrder()
        dimer_value = dimer_obs.compute(state)
        
        assert isinstance(dimer_value, float)
        assert np.isfinite(dimer_value)
        assert dimer_value >= 0.0


class TestObservableModule:
    """Test ObservableModule class"""
    
    def test_initialization(self):
        """Test ObservableModule initialization"""
        config = Config.from_yaml("configs/default_config.yaml")
        obs_module = ObservableModule(config)
        
        assert obs_module.config is config
        assert len(obs_module.observables) > 0
        
        # Check that all expected observables are present
        expected_obs = [
            'energy',
            'energy_density',
            'staggered_mag',
            'stripe_order',
            'plaquette_order',
            'structure_factor_pi_pi',
            'structure_factor_pi_0',
            'structure_factor_0_pi',
            'entanglement_entropy',
            'nematic_order',
            'dimer_order'
        ]
        
        for obs_name in expected_obs:
            assert obs_name in obs_module.observables
    
    def test_compute_all(self):
        """Test computing all observables for a single state"""
        config = Config.from_yaml("configs/default_config.yaml")
        obs_module = ObservableModule(config)
        
        # Create a ground state
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        # Compute all observables
        results = obs_module.compute_all(state)
        
        # Check that we got results for all observables
        assert len(results) == len(obs_module.observables)
        
        # Check that all values are finite (or NaN if computation failed)
        for obs_name, value in results.items():
            assert isinstance(value, float)
            # Value should be finite or NaN (but not Inf)
            assert np.isfinite(value) or np.isnan(value)
    
    def test_compute_for_sweep(self):
        """Test computing observables for parameter sweep"""
        config = Config.from_yaml("configs/default_config.yaml")
        obs_module = ObservableModule(config)
        
        # Create a small set of ground states
        states = {}
        for j2_j1 in [0.4, 0.5]:
            for L in [4]:
                ham = J1J2Hamiltonian(L=L, j2_j1=j2_j1)
                ham.build_hamiltonian()
                state = ham.compute_ground_state()
                states[(j2_j1, L)] = state
        
        # Compute observables for all states
        df = obs_module.compute_for_sweep(states)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert 'j2_j1' in df.columns
        assert 'L' in df.columns
        assert 'observable' in df.columns
        assert 'value' in df.columns
        
        # Check that we have the right number of rows
        expected_rows = len(states) * len(obs_module.observables)
        assert len(df) == expected_rows
        
        # Check that all parameter points are present
        for j2_j1, L in states.keys():
            subset = df[(df['j2_j1'] == j2_j1) & (df['L'] == L)]
            assert len(subset) == len(obs_module.observables)
    
    def test_observable_determinism(self):
        """Test that computing observables multiple times gives identical results"""
        config = Config.from_yaml("configs/default_config.yaml")
        obs_module = ObservableModule(config)
        
        # Create a ground state
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        # Compute observables twice
        results1 = obs_module.compute_all(state)
        results2 = obs_module.compute_all(state)
        
        # Check that results are identical
        for obs_name in results1.keys():
            if np.isnan(results1[obs_name]):
                assert np.isnan(results2[obs_name])
            else:
                assert results1[obs_name] == results2[obs_name]


class TestObservablePhysicalBounds:
    """Test that observables satisfy physical bounds"""
    
    def test_staggered_magnetization_bounds(self):
        """Test that staggered magnetization is non-negative and finite"""
        config = Config.from_yaml("configs/default_config.yaml")
        obs_module = ObservableModule(config)
        
        # Test multiple parameter points
        for j2_j1 in [0.0, 0.3, 0.5, 0.7]:
            ham = J1J2Hamiltonian(L=4, j2_j1=j2_j1)
            ham.build_hamiltonian()
            state = ham.compute_ground_state()
            
            results = obs_module.compute_all(state)
            stag_mag = results['staggered_mag']
            
            # Staggered magnetization should be non-negative and finite
            assert stag_mag >= 0.0, \
                f"Staggered magnetization {stag_mag} is negative for j2_j1={j2_j1}"
            assert np.isfinite(stag_mag), \
                f"Staggered magnetization {stag_mag} is not finite for j2_j1={j2_j1}"
    
    def test_energy_density_negative(self):
        """Test that energy density is negative (antiferromagnetic)"""
        config = Config.from_yaml("configs/default_config.yaml")
        obs_module = ObservableModule(config)
        
        # For antiferromagnetic interactions, energy should be negative
        ham = J1J2Hamiltonian(L=4, j2_j1=0.5)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        results = obs_module.compute_all(state)
        energy_density = results['energy_density']
        
        # Energy density should be negative for antiferromagnetic systems
        assert energy_density < 0.0


class TestObservableKnownPhases:
    """Test observable behavior in known phases"""
    
    def test_neel_phase_staggered_mag(self):
        """Test that staggered magnetization is large in Néel phase"""
        config = Config.from_yaml("configs/default_config.yaml")
        obs_module = ObservableModule(config)
        
        # J2=0 should be in Néel phase
        ham = J1J2Hamiltonian(L=4, j2_j1=0.0)
        ham.build_hamiltonian()
        state = ham.compute_ground_state()
        
        results = obs_module.compute_all(state)
        stag_mag = results['staggered_mag']
        stripe_order = results['stripe_order']
        
        # In Néel phase, staggered magnetization should dominate
        assert stag_mag > 0.1
        # Staggered mag should be larger than stripe order in Néel phase
        assert stag_mag > stripe_order
