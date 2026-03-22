"""
Alloy Calculator Module
Provides calculations for creating alloys from constituent elements.
Uses physics-based formulas for property estimation.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ==================== Physical Constants ====================

class AlloyConstants:
    """Constants for alloy calculations"""

    # Lattice constants for pure elements (pm) at room temperature
    LATTICE_CONSTANTS = {
        'Fe': {'structure': 'BCC', 'a': 286.65},
        'Al': {'structure': 'FCC', 'a': 404.95},
        'Cu': {'structure': 'FCC', 'a': 361.49},
        'Ni': {'structure': 'FCC', 'a': 352.4},
        'Cr': {'structure': 'BCC', 'a': 288.46},
        'Ti': {'structure': 'HCP', 'a': 295.08, 'c': 468.55},
        'Zn': {'structure': 'HCP', 'a': 266.49, 'c': 494.68},
        'Sn': {'structure': 'BCT', 'a': 583.18, 'c': 318.18},
        'Mn': {'structure': 'BCC', 'a': 891.39},
        'Mo': {'structure': 'BCC', 'a': 314.7},
        'W': {'structure': 'BCC', 'a': 316.52},
        'V': {'structure': 'BCC', 'a': 302.4},
        'Co': {'structure': 'HCP', 'a': 250.71, 'c': 406.95},
        'Nb': {'structure': 'BCC', 'a': 330.04},
        'Si': {'structure': 'Diamond', 'a': 543.09},
        'Ag': {'structure': 'FCC', 'a': 408.53},
        'Au': {'structure': 'FCC', 'a': 407.82},
        'Pb': {'structure': 'FCC', 'a': 495.02},
    }

    # Densities of pure elements (g/cm³)
    ELEMENT_DENSITIES = {
        'Fe': 7.874, 'Al': 2.70, 'Cu': 8.96, 'Ni': 8.908, 'Cr': 7.19,
        'Ti': 4.506, 'Zn': 7.14, 'Sn': 7.265, 'Mn': 7.21, 'Mo': 10.28,
        'W': 19.25, 'V': 6.11, 'Co': 8.90, 'Nb': 8.57, 'Si': 2.33,
        'Ag': 10.49, 'Au': 19.30, 'Pb': 11.34, 'C': 2.267, 'N': 1.251,
        'P': 1.823, 'S': 2.07, 'B': 2.34, 'Mg': 1.738
    }

    # Melting points of pure elements (K)
    ELEMENT_MELTING_POINTS = {
        'Fe': 1811, 'Al': 933.5, 'Cu': 1357.8, 'Ni': 1728, 'Cr': 2180,
        'Ti': 1941, 'Zn': 692.7, 'Sn': 505.1, 'Mn': 1519, 'Mo': 2896,
        'W': 3695, 'V': 2183, 'Co': 1768, 'Nb': 2750, 'Si': 1687,
        'Ag': 1234.9, 'Au': 1337.3, 'Pb': 600.6, 'C': 3915, 'N': 63.15,
        'P': 317.3, 'S': 388.4, 'B': 2349, 'Mg': 923
    }

    # Thermal conductivities (W/m·K)
    ELEMENT_THERMAL_CONDUCTIVITY = {
        'Fe': 80.4, 'Al': 237, 'Cu': 401, 'Ni': 90.9, 'Cr': 93.9,
        'Ti': 21.9, 'Zn': 116, 'Sn': 66.8, 'Mn': 7.81, 'Mo': 138,
        'W': 173, 'V': 30.7, 'Co': 100, 'Nb': 53.7, 'Si': 149,
        'Ag': 429, 'Au': 317, 'Pb': 35.3, 'C': 140, 'Mg': 156
    }

    # Electrical resistivities (Ω·m × 10^-8)
    ELEMENT_RESISTIVITY = {
        'Fe': 9.71, 'Al': 2.65, 'Cu': 1.68, 'Ni': 6.99, 'Cr': 12.7,
        'Ti': 42.0, 'Zn': 5.92, 'Sn': 11.5, 'Mn': 144, 'Mo': 5.34,
        'W': 5.28, 'V': 20.1, 'Co': 6.24, 'Nb': 15.2, 'Si': 2300,
        'Ag': 1.59, 'Au': 2.44, 'Pb': 20.6, 'C': 3500
    }

    # Atomic masses (g/mol)
    ATOMIC_MASSES = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'Al': 26.982,
        'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Ti': 47.867, 'V': 50.942,
        'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693,
        'Cu': 63.546, 'Zn': 65.38, 'Nb': 92.906, 'Mo': 95.95, 'Ag': 107.868,
        'Sn': 118.71, 'W': 183.84, 'Au': 196.967, 'Pb': 207.2, 'B': 10.81,
        'Mg': 24.305
    }


# ==================== Alloy Calculator ====================

class AlloyCalculator:
    """
    Calculate alloy properties from constituent elements.
    Uses physics-based formulas including rule of mixtures and Vegard's law.
    """

    @classmethod
    def create_alloy_from_components(
        cls,
        component_data: List[Dict],
        weight_fractions: List[float],
        lattice_type: str = "FCC",
        name: str = None
    ) -> Dict:
        """
        Calculate alloy properties from constituent elements.

        Args:
            component_data: List of element data dictionaries
                           Each should have at least 'symbol' or 'Element'
            weight_fractions: Weight fractions for each component (should sum to 1.0)
            lattice_type: Crystal structure type (FCC, BCC, HCP, etc.)
            name: Optional name for the alloy

        Returns:
            Dictionary containing calculated alloy properties
        """
        if not component_data or not weight_fractions:
            return {}

        if len(component_data) != len(weight_fractions):
            raise ValueError("Component data and weight fractions must have same length")

        # Normalize weight fractions
        total = sum(weight_fractions)
        if total <= 0:
            raise ValueError("Weight fractions must sum to a positive value")
        weight_fractions = [w / total for w in weight_fractions]

        # Extract element symbols
        elements = []
        for comp in component_data:
            sym = comp.get('symbol') or comp.get('Element') or comp.get('Symbol', 'Unknown')
            elements.append(sym)

        # Calculate properties
        density = cls._calculate_density(elements, weight_fractions)
        melting_point = cls._calculate_melting_point(elements, weight_fractions)
        thermal_conductivity = cls._calculate_thermal_conductivity(elements, weight_fractions)
        electrical_resistivity = cls._calculate_electrical_resistivity(elements, weight_fractions)
        lattice_param = cls._calculate_lattice_parameter(elements, weight_fractions, lattice_type)
        estimated_strength = cls._estimate_strength(elements, weight_fractions, density)

        # Build components list
        components = []
        for elem, wf in zip(elements, weight_fractions):
            role = cls._determine_role(elem, wf, elements)
            components.append({
                'Element': elem,
                'MinPercent': wf * 100 * 0.95,  # Allow 5% variation
                'MaxPercent': wf * 100 * 1.05,
                'Role': role
            })

        # Determine primary element (highest weight fraction)
        max_idx = weight_fractions.index(max(weight_fractions))
        primary_element = elements[max_idx]

        # Generate formula
        formula = cls._generate_formula(elements, weight_fractions)

        # Determine category based on primary element
        category = cls._determine_category(primary_element, elements)

        # Build alloy data structure
        alloy_data = {
            'Name': name or f"Custom {category} Alloy",
            'Formula': formula,
            'Category': category,
            'SubCategory': 'Custom',
            'Description': f"Custom alloy created from {', '.join(elements)}",

            'Components': components,

            'PhysicalProperties': {
                'Density_g_cm3': round(density, 3),
                'MeltingPoint_K': round(melting_point, 1),
                'ThermalConductivity_W_mK': round(thermal_conductivity, 1),
                'ThermalExpansion_per_K': cls._calculate_thermal_expansion(elements, weight_fractions) * 1e-6,
                'ElectricalResistivity_Ohm_m': electrical_resistivity,
                'SpecificHeat_J_kgK': cls._calculate_specific_heat(elements, weight_fractions),
                'YoungsModulus_GPa': round(estimated_strength['youngs_modulus'], 1),
                'ShearModulus_GPa': round(estimated_strength['youngs_modulus'] / 2.6, 1),
                'PoissonsRatio': 0.30,
                'BrinellHardness_HB': round(estimated_strength['hardness'])
            },

            'MechanicalProperties': {
                'TensileStrength_MPa': round(estimated_strength['tensile_strength']),
                'YieldStrength_MPa': round(estimated_strength['yield_strength']),
                'Elongation_percent': round(estimated_strength['elongation']),
                'ReductionOfArea_percent': 50,
                'ImpactStrength_J': 100,
                'FatigueStrength_MPa': round(estimated_strength['tensile_strength'] * 0.45)
            },

            'LatticeProperties': {
                'PrimaryStructure': lattice_type,
                'SecondaryStructures': [],
                'LatticeParameters': {
                    'a_pm': round(lattice_param, 2),
                    'b_pm': round(lattice_param, 2),
                    'c_pm': round(lattice_param * (1.633 if lattice_type == 'HCP' else 1.0), 2),
                    'alpha_deg': 90,
                    'beta_deg': 90,
                    'gamma_deg': 120 if lattice_type == 'HCP' else 90
                },
                'AtomicPackingFactor': cls._get_packing_factor(lattice_type),
                'CoordinationNumber': cls._get_coordination_number(lattice_type)
            },

            'PhaseComposition': cls._estimate_phase_composition(elements, weight_fractions, lattice_type),

            'Microstructure': {
                'GrainStructure': {
                    'AverageGrainSize_um': 50,
                    'GrainSizeDistribution': 'LogNormal',
                    'GrainSizeStdDev': 0.35,
                    'ASTMGrainSizeNumber': 5,
                    'VoronoiSeedDensity_per_mm2': 400
                },
                'PhaseDistribution': {
                    'NoiseType': 'Simplex',
                    'NoiseScale': 0.1,
                    'NoiseOctaves': 3,
                    'NoisePersistence': 0.5
                }
            },

            'CorrosionResistance': cls._calculate_corrosion_resistance(elements, weight_fractions),

            'Applications': [],
            'ProcessingMethods': [],
            'Color': cls._get_alloy_color(primary_element)
        }

        # === Calculate Atom Positions in Lattice ===
        atom_positions = cls._calculate_atom_positions_in_lattice(elements, weight_fractions, lattice_type, lattice_param)

        # === Calculate Defect Concentrations ===
        defect_data = cls._calculate_defect_concentrations(elements, weight_fractions, melting_point)

        # === Calculate Grain Boundary Data ===
        grain_boundary_data = cls._calculate_grain_boundary_data(elements, weight_fractions, density)

        # === Preserve ALL Input Element Properties ===
        preserved_elements = []
        for comp, wf in zip(component_data, weight_fractions):
            preserved_elements.append({
                'original_data': comp.copy(),  # Preserve ALL input properties
                'symbol': comp.get('symbol') or comp.get('Element') or comp.get('Symbol', 'Unknown'),
                'weight_fraction': wf,
                'atomic_fraction': cls._weight_to_atomic_fractions(elements, weight_fractions)[elements.index(comp.get('symbol') or comp.get('Element') or comp.get('Symbol', 'Unknown'))],
                'role': cls._determine_role(comp.get('symbol') or comp.get('Element') or comp.get('Symbol', 'Unknown'), wf, elements)
            })

        # Add comprehensive simulation data
        alloy_data['SimulationData'] = {
            'PreservedElements': preserved_elements,
            'AtomPositions': atom_positions,
            'DefectConcentrations': defect_data,
            'GrainBoundaryData': grain_boundary_data,
            'PhaseData': alloy_data['PhaseComposition'],
            'LatticeData': {
                'lattice_type': lattice_type,
                'lattice_parameter_pm': round(lattice_param, 2),
                'packing_factor': cls._get_packing_factor(lattice_type),
                'coordination_number': cls._get_coordination_number(lattice_type),
                'unit_cell_volume_pm3': round(lattice_param ** 3, 1) if lattice_type in ['FCC', 'BCC'] else round(lattice_param ** 2 * lattice_param * 1.633 * 0.866, 1),
                'atomic_volume_pm3': round(lattice_param ** 3 / cls._get_atoms_per_unit_cell(lattice_type), 1),
            },
            'StrengtheningMechanisms': {
                'solid_solution': cls._calculate_solid_solution_strengthening(elements, weight_fractions),
                'precipitation': cls._calculate_precipitation_strengthening(elements, weight_fractions),
                'grain_boundary': cls._calculate_grain_boundary_strengthening(50),  # 50 um default grain size
                'dislocation': {'density_m-2': 1e12, 'strengthening_MPa': 50}
            },
            'Uncertainties': {
                'density_percent': 2.0,
                'strength_percent': 10.0,
                'thermal_conductivity_percent': 15.0,
                'method': 'rule_of_mixtures_with_empirical_corrections'
            }
        }

        # Add derived properties for easy access
        alloy_data['name'] = alloy_data['Name']
        alloy_data['category'] = alloy_data['Category']
        alloy_data['density'] = alloy_data['PhysicalProperties']['Density_g_cm3']
        alloy_data['melting_point'] = alloy_data['PhysicalProperties']['MeltingPoint_K']
        alloy_data['tensile_strength'] = alloy_data['MechanicalProperties']['TensileStrength_MPa']
        alloy_data['yield_strength'] = alloy_data['MechanicalProperties']['YieldStrength_MPa']
        alloy_data['crystal_structure'] = lattice_type
        alloy_data['primary_element'] = primary_element

        return alloy_data

    @classmethod
    def _calculate_density(cls, elements: List[str], weight_fractions: List[float]) -> float:
        """
        Calculate alloy density using rule of mixtures.
        1/ρ_alloy = Σ(w_i / ρ_i)
        """
        inv_density_sum = 0
        for elem, wf in zip(elements, weight_fractions):
            elem_density = AlloyConstants.ELEMENT_DENSITIES.get(elem, 7.0)
            if elem_density > 0:
                inv_density_sum += wf / elem_density

        return 1.0 / inv_density_sum if inv_density_sum > 0 else 7.0

    @classmethod
    def _calculate_melting_point(cls, elements: List[str], weight_fractions: List[float]) -> float:
        """
        Calculate approximate melting point.
        Uses weighted average with depression factor for multi-component alloys.
        """
        weighted_mp = 0
        for elem, wf in zip(elements, weight_fractions):
            mp = AlloyConstants.ELEMENT_MELTING_POINTS.get(elem, 1500)
            weighted_mp += wf * mp

        # Apply melting point depression for alloys (typically 5-15%)
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        depression_factor = 1.0 - 0.03 * (num_components - 1)
        depression_factor = max(0.85, min(1.0, depression_factor))

        return weighted_mp * depression_factor

    @classmethod
    def _calculate_thermal_conductivity(cls, elements: List[str], weight_fractions: List[float]) -> float:
        """
        Calculate thermal conductivity.
        Alloys typically have lower thermal conductivity than pure metals.
        """
        weighted_tc = 0
        for elem, wf in zip(elements, weight_fractions):
            tc = AlloyConstants.ELEMENT_THERMAL_CONDUCTIVITY.get(elem, 50)
            weighted_tc += wf * tc

        # Apply reduction factor for alloying (phonon scattering)
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        reduction = 0.7 ** (num_components - 1)
        reduction = max(0.3, min(1.0, reduction))

        return weighted_tc * reduction

    @classmethod
    def _calculate_electrical_resistivity(cls, elements: List[str], weight_fractions: List[float]) -> float:
        """
        Calculate electrical resistivity.
        Alloying increases resistivity due to electron scattering.
        """
        weighted_res = 0
        for elem, wf in zip(elements, weight_fractions):
            res = AlloyConstants.ELEMENT_RESISTIVITY.get(elem, 10)
            weighted_res += wf * res

        # Matthiessen's rule addition for alloying
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        alloying_addition = 5 * (num_components - 1)  # Additional resistivity

        return (weighted_res + alloying_addition) * 1e-8

    @classmethod
    def _calculate_lattice_parameter(cls, elements: List[str], weight_fractions: List[float],
                                      lattice_type: str) -> float:
        """
        Calculate lattice parameter using Vegard's law.
        a_alloy = Σ(x_i * a_i)
        """
        # Convert weight fractions to atomic fractions
        atomic_fracs = cls._weight_to_atomic_fractions(elements, weight_fractions)

        weighted_a = 0
        total_frac = 0
        for elem, af in zip(elements, atomic_fracs):
            lattice_info = AlloyConstants.LATTICE_CONSTANTS.get(elem, {})
            a = lattice_info.get('a', 350)  # Default lattice constant
            weighted_a += af * a
            total_frac += af

        return weighted_a / total_frac if total_frac > 0 else 350

    @classmethod
    def _get_element_data(cls, elements: List[str]) -> List[Dict]:
        """Get element data dictionaries for use with predictive physics."""
        element_data = []
        for elem in elements:
            element_data.append({
                'symbol': elem,
                'Symbol': elem,
                'Element': elem,
                'density': AlloyConstants.ELEMENT_DENSITIES.get(elem, 7.0),
                'melting_point': AlloyConstants.ELEMENT_MELTING_POINTS.get(elem, 1500),
                'atomic_mass': AlloyConstants.ATOMIC_MASSES.get(elem, 50),
            })
        return element_data

    @classmethod
    def _estimate_strength(cls, elements: List[str], weight_fractions: List[float],
                           density: float) -> Dict:
        """
        Estimate strength using all element sub-properties.

        Uses the predictive physics engine for comprehensive property prediction
        including:
        - Solid solution strengthening
        - Precipitation hardening estimates
        - Hall-Petch grain size effects
        - Element position predictions in crystal lattice

        Improved model includes:
        - Base strength from alloy category
        - Solid solution strengthening
        - Precipitation/phase strengthening
        - Alloy-specific empirical factors
        """
        # Use predictive physics engine for comprehensive calculation
        from periodica.utils.predictive_physics import UniversalPredictor

        # Get full element data
        element_data = cls._get_element_data(elements)

        predictor = UniversalPredictor()
        result = predictor.predict_alloy_properties(element_data, weight_fractions)

        # Return in expected format with enhanced details
        return {
            'tensile_strength': result['tensile_strength'],
            'yield_strength': result['yield_strength'],
            'elongation': result['elongation'],
            'hardness': result['hardness'],
            'youngs_modulus': result['youngs_modulus'],
            'element_positions': result.get('element_positions', []),
            'uncertainty': result.get('uncertainty', {}),
            'method': result.get('method', 'predictive_physics'),
            'details': result.get('details', {})
        }

        # Legacy calculation follows for backward compatibility reference
        # Get element percentages for calculations
        elem_pct = {elem: wf * 100 for elem, wf in zip(elements, weight_fractions)}

        # Determine primary element and alloy category
        max_idx = weight_fractions.index(max(weight_fractions))
        primary = elements[max_idx]

        # Base strength varies by alloy system (MPa)
        # Reference values for annealed condition
        base_strengths = {
            'Fe': 280,   # Low carbon steel base
            'Al': 90,    # Pure Al base
            'Cu': 220,   # Pure Cu base
            'Ti': 450,   # CP Ti grade 2
            'Ni': 450,   # Pure Ni base
            'Co': 500,   # Pure Co base
            'Mg': 130,   # Pure Mg base
        }
        base_strength = base_strengths.get(primary, 200)

        # Solid solution strengthening
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        ss_strengthening = 30 * (num_components - 1)

        # Element-specific strengthening contributions
        for elem, wf in zip(elements, weight_fractions):
            pct = wf * 100
            if elem == 'C' and wf > 0:
                # Carbon: very effective in steels (Hall-Petch + pearlite)
                ss_strengthening += pct * 800
            elif elem == 'N' and wf > 0:
                ss_strengthening += pct * 700
            elif elem == 'Mo' and wf > 0:
                ss_strengthening += pct * 35
            elif elem == 'W' and wf > 0:
                ss_strengthening += pct * 30
            elif elem == 'V' and wf > 0:
                # V: beta stabilizer in Ti, carbide former in steel
                if primary == 'Ti':
                    ss_strengthening += pct * 80  # Alpha-beta strengthening
                else:
                    ss_strengthening += pct * 40
            elif elem == 'Nb' and wf > 0:
                ss_strengthening += pct * 35
            elif elem == 'Cr' and wf > 0:
                ss_strengthening += pct * 15
            elif elem == 'Ni' and wf > 0:
                ss_strengthening += pct * 12
            elif elem == 'Mn' and wf > 0:
                ss_strengthening += pct * 25
            elif elem == 'Si' and wf > 0:
                ss_strengthening += pct * 60
            elif elem == 'Cu' and wf > 0 and primary != 'Cu':
                ss_strengthening += pct * 30  # Precipitation strengthening
            elif elem == 'Al' and wf > 0:
                if primary == 'Ti':
                    ss_strengthening += pct * 60  # Alpha stabilizer
                elif primary == 'Ni':
                    ss_strengthening += pct * 100  # Gamma prime former
                else:
                    ss_strengthening += pct * 20
            elif elem == 'Zn' and wf > 0 and primary in ['Cu', 'Al']:
                ss_strengthening += pct * 8  # Brass/Al-Zn strengthening

        # Alloy-category-specific strengthening bonuses
        category_bonus = 0

        # Titanium alpha-beta alloys (Ti-6Al-4V type)
        if primary == 'Ti' and elem_pct.get('Al', 0) > 3 and elem_pct.get('V', 0) > 2:
            category_bonus += 350  # Alpha-beta transformation strengthening

        # Nickel superalloys
        if primary == 'Ni' and elem_pct.get('Cr', 0) > 10:
            category_bonus += 200  # Gamma/gamma-prime strengthening

        # Stainless steels
        if primary == 'Fe' and elem_pct.get('Cr', 0) > 10:
            if elem_pct.get('Ni', 0) > 6:
                category_bonus += 100  # Austenitic SS
            elif elem_pct.get('C', 0) > 0.1:
                category_bonus += 200  # Martensitic SS

        # Aluminum aerospace alloys (2xxx, 7xxx series)
        if primary == 'Al':
            if elem_pct.get('Cu', 0) > 2:
                category_bonus += 250  # 2xxx precipitation hardening
            elif elem_pct.get('Zn', 0) > 3:
                category_bonus += 300  # 7xxx precipitation hardening

        tensile_strength = base_strength + ss_strengthening + category_bonus

        # Yield strength ratio varies by alloy type
        if primary == 'Ti':
            ys_ratio = 0.9  # Ti alloys have high YS/UTS ratio
        elif primary == 'Al':
            ys_ratio = 0.85
        else:
            ys_ratio = 0.65  # Steels and Cu alloys

        yield_strength = tensile_strength * ys_ratio

        # Elongation decreases with strength (inverse relationship)
        # But varies by alloy system
        if primary == 'Ti':
            elongation = max(8, 25 - tensile_strength / 100)
        elif primary == 'Al':
            elongation = max(3, 30 - tensile_strength / 50)
        else:
            elongation = max(5, 50 - tensile_strength / 25)

        # Young's modulus estimate (rule of mixtures, roughly)
        weighted_E = 0
        E_values = {'Fe': 210, 'Al': 69, 'Cu': 130, 'Ni': 200, 'Ti': 116,
                    'Cr': 279, 'Mo': 329, 'W': 411, 'Co': 209, 'Mg': 45,
                    'Zn': 108, 'Sn': 50, 'Ag': 83, 'Au': 78, 'Pb': 16}
        for elem, wf in zip(elements, weight_fractions):
            E = E_values.get(elem, 150)
            weighted_E += wf * E

        # Hardness estimate (Brinell, correlated with UTS)
        # HB ≈ UTS/3.45 for steels, varies for other alloys
        if primary == 'Fe':
            hardness = tensile_strength / 3.45
        elif primary == 'Al':
            hardness = tensile_strength / 4.0
        elif primary == 'Ti':
            hardness = tensile_strength / 3.0
        else:
            hardness = tensile_strength / 3.5

        return {
            'tensile_strength': tensile_strength,
            'yield_strength': yield_strength,
            'elongation': elongation,
            'youngs_modulus': weighted_E,
            'hardness': hardness
        }

    @classmethod
    def _weight_to_atomic_fractions(cls, elements: List[str], weight_fractions: List[float]) -> List[float]:
        """Convert weight fractions to atomic fractions"""
        molar_fractions = []
        for elem, wf in zip(elements, weight_fractions):
            atomic_mass = AlloyConstants.ATOMIC_MASSES.get(elem, 50)
            molar_fractions.append(wf / atomic_mass)

        total = sum(molar_fractions)
        return [mf / total for mf in molar_fractions] if total > 0 else weight_fractions

    @classmethod
    def _determine_role(cls, element: str, weight_frac: float, all_elements: List[str]) -> str:
        """Determine the role of an element in the alloy"""
        if weight_frac >= max(0.5, max(weight_frac for e in all_elements)):
            return "Base"

        roles = {
            'C': 'Strengthening',
            'N': 'Strengthening',
            'Cr': 'Corrosion Resistance',
            'Ni': 'Stabilizer',
            'Mo': 'Strengthening',
            'V': 'Grain Refiner',
            'Ti': 'Grain Refiner',
            'Mn': 'Deoxidizer',
            'Si': 'Deoxidizer',
            'W': 'Hardening',
            'Co': 'Strengthening',
            'Al': 'Deoxidizer',
            'P': 'Impurity',
            'S': 'Impurity',
            'Cu': 'Corrosion Resistance'
        }
        return roles.get(element, 'Other')

    @classmethod
    def _determine_category(cls, primary_element: str, elements: List[str]) -> str:
        """Determine alloy category from composition"""
        categories = {
            'Fe': 'Steel',
            'Al': 'Aluminum',
            'Cu': 'Copper',
            'Ti': 'Titanium',
            'Ni': 'Nickel',
            'Zn': 'Zinc',
            'Sn': 'Tin',
            'Ag': 'Precious',
            'Au': 'Precious',
            'Pb': 'Lead'
        }

        base_category = categories.get(primary_element, 'Other')

        # Special cases
        if primary_element == 'Cu':
            if 'Zn' in elements:
                return 'Brass'
            if 'Sn' in elements:
                return 'Bronze'

        return base_category

    @classmethod
    def _generate_formula(cls, elements: List[str], weight_fractions: List[float]) -> str:
        """Generate a formula string for the alloy"""
        # Sort by weight fraction (descending)
        sorted_pairs = sorted(zip(elements, weight_fractions), key=lambda x: -x[1])

        # Take top elements with significant fractions
        significant = [(e, wf) for e, wf in sorted_pairs if wf > 0.005]

        if len(significant) <= 3:
            return '-'.join(e for e, _ in significant)
        else:
            return '-'.join(e for e, _ in significant[:3]) + '...'

    @classmethod
    def _get_packing_factor(cls, lattice_type: str) -> float:
        """Get atomic packing factor for a lattice type"""
        factors = {
            'FCC': 0.74,
            'HCP': 0.74,
            'BCC': 0.68,
            'BCT': 0.70,
            'Diamond': 0.34
        }
        return factors.get(lattice_type, 0.68)

    @classmethod
    def _get_coordination_number(cls, lattice_type: str) -> int:
        """Get coordination number for a lattice type"""
        numbers = {
            'FCC': 12,
            'HCP': 12,
            'BCC': 8,
            'BCT': 8,
            'Diamond': 4
        }
        return numbers.get(lattice_type, 8)

    @classmethod
    def _get_alloy_color(cls, primary_element: str) -> str:
        """Get display color for an alloy based on primary element"""
        colors = {
            'Fe': '#C0C0C0',  # Silver
            'Al': '#E8E8E8',  # Light grey
            'Cu': '#B87333',  # Copper
            'Ti': '#8E8E8E',  # Grey
            'Ni': '#A8A8A8',  # Light grey
            'Zn': '#B0B0B0',  # Grey
            'Sn': '#909090',  # Grey
            'Ag': '#C0C0C0',  # Silver
            'Au': '#FFD700',  # Gold
            'Pb': '#666666'   # Dark grey
        }
        return colors.get(primary_element, '#C0C0C0')

    @classmethod
    def _calculate_thermal_expansion(cls, elements: List[str], weight_fractions: List[float]) -> float:
        """
        Calculate coefficient of thermal expansion using rule of mixtures.

        CTE_alloy ≈ Σ(w_i × CTE_i)

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions

        Returns:
            Thermal expansion coefficient in per K (×10^-6)
        """
        # Thermal expansion coefficients for elements (×10^-6 /K)
        element_cte = {
            'Fe': 11.8, 'Al': 23.1, 'Cu': 16.5, 'Ni': 13.4, 'Cr': 4.9,
            'Ti': 8.6, 'Zn': 30.2, 'Sn': 22.0, 'Mn': 21.7, 'Mo': 4.8,
            'W': 4.5, 'V': 8.4, 'Co': 13.0, 'Nb': 7.3, 'Si': 2.6,
            'Ag': 18.9, 'Au': 14.2, 'Pb': 28.9, 'C': 1.0, 'Mg': 24.8
        }

        weighted_cte = 0
        for elem, wf in zip(elements, weight_fractions):
            cte = element_cte.get(elem, 12.0)  # Default 12 ppm/K
            weighted_cte += wf * cte

        return round(weighted_cte, 1)

    @classmethod
    def _calculate_specific_heat(cls, elements: List[str], weight_fractions: List[float]) -> float:
        """
        Calculate specific heat capacity using Kopp-Neumann rule.

        Cp_alloy ≈ Σ(w_i × Cp_i)

        For metals, Dulong-Petit law gives ~25 J/(mol·K) per atom
        Cp (J/kg·K) = 25 / M × 1000

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions

        Returns:
            Specific heat in J/(kg·K)
        """
        # Specific heat capacities (J/kg·K)
        element_cp = {
            'Fe': 449, 'Al': 897, 'Cu': 385, 'Ni': 444, 'Cr': 449,
            'Ti': 523, 'Zn': 388, 'Sn': 228, 'Mn': 479, 'Mo': 251,
            'W': 132, 'V': 489, 'Co': 421, 'Nb': 265, 'Si': 705,
            'Ag': 235, 'Au': 129, 'Pb': 129, 'C': 709, 'Mg': 1023, 'N': 1040
        }

        weighted_cp = 0
        for elem, wf in zip(elements, weight_fractions):
            cp = element_cp.get(elem, 450)  # Default 450 J/kg·K
            weighted_cp += wf * cp

        return round(weighted_cp, 0)

    @classmethod
    def _calculate_corrosion_resistance(cls, elements: List[str], weight_fractions: List[float]) -> Dict:
        """
        Calculate corrosion resistance metrics.

        PREN (Pitting Resistance Equivalent Number):
        PREN = %Cr + 3.3×%Mo + 16×%N

        Higher PREN = better pitting corrosion resistance
        PREN > 40 is considered highly corrosion resistant

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions

        Returns:
            Dict with PREN, passivation film, and corrosion rating
        """
        # Get element percentages
        elem_percent = {elem: wf * 100 for elem, wf in zip(elements, weight_fractions)}

        cr_pct = elem_percent.get('Cr', 0)
        mo_pct = elem_percent.get('Mo', 0)
        n_pct = elem_percent.get('N', 0)
        ni_pct = elem_percent.get('Ni', 0)

        # Calculate PREN
        pren = cr_pct + 3.3 * mo_pct + 16 * n_pct

        # Determine passivation film composition
        if cr_pct >= 10.5:
            passivation_film = "Cr2O3"
        elif elem_percent.get('Al', 0) > 1:
            passivation_film = "Al2O3"
        elif elem_percent.get('Ti', 0) > 1:
            passivation_film = "TiO2"
        else:
            passivation_film = "FeO/Fe2O3"

        # Pitting potential estimate (mV vs SCE)
        if pren > 40:
            pitting_potential = 400 + (pren - 40) * 5
            rating = "Excellent"
        elif pren > 25:
            pitting_potential = 200 + (pren - 25) * 13
            rating = "Good"
        elif pren > 15:
            pitting_potential = 50 + (pren - 15) * 15
            rating = "Moderate"
        else:
            pitting_potential = pren * 3
            rating = "Poor"

        # Critical pitting temperature (K)
        cpt = 253 + pren * 2  # Rough estimate

        return {
            'PREN': round(pren, 1),
            'PassivationFilmComposition': passivation_film,
            'PittingPotential_mV_SCE': round(pitting_potential, 0),
            'CriticalPittingTemperature_K': round(cpt, 0),
            'CorrosionRating': rating,
            'Details': {
                'Cr_percent': cr_pct,
                'Mo_percent': mo_pct,
                'N_percent': n_pct,
                'formula': 'PREN = %Cr + 3.3×%Mo + 16×%N'
            }
        }

    @classmethod
    def _estimate_phase_composition(cls, elements: List[str], weight_fractions: List[float],
                                     lattice_type: str) -> Dict:
        """
        Estimate phase composition from alloy composition.

        Uses empirical rules:
        - Ni equivalents determine austenite stability
        - Cr equivalents determine ferrite formation
        - Schaeffler diagram concepts for stainless steels

        Ni_eq = %Ni + 30×%C + 0.5×%Mn
        Cr_eq = %Cr + %Mo + 1.5×%Si + 0.5×%Nb

        Args:
            elements: List of element symbols
            weight_fractions: Weight fractions
            lattice_type: Primary lattice structure

        Returns:
            Dict with phases and their volume fractions
        """
        # Get element percentages
        elem_pct = {elem: wf * 100 for elem, wf in zip(elements, weight_fractions)}

        ni_pct = elem_pct.get('Ni', 0)
        cr_pct = elem_pct.get('Cr', 0)
        c_pct = elem_pct.get('C', 0)
        mn_pct = elem_pct.get('Mn', 0)
        mo_pct = elem_pct.get('Mo', 0)
        si_pct = elem_pct.get('Si', 0)
        nb_pct = elem_pct.get('Nb', 0)

        # Calculate Schaeffler equivalents
        ni_eq = ni_pct + 30 * c_pct + 0.5 * mn_pct
        cr_eq = cr_pct + mo_pct + 1.5 * si_pct + 0.5 * nb_pct

        phases = []

        # Determine phases using Schaeffler-like approach
        if 'Fe' in elements and cr_pct > 10:
            # Stainless steel - use Schaeffler diagram logic
            if ni_eq > 12 and cr_eq < 25:
                # Austenitic region
                phases.append({
                    'Name': 'Austenite',
                    'Symbol': 'gamma',
                    'Structure': 'FCC',
                    'VolumePercent': 95,
                    'Magnetic': False,
                    'Hardness_HV': 180
                })
                if cr_eq > 18:
                    phases.append({
                        'Name': 'Delta Ferrite',
                        'Symbol': 'delta',
                        'Structure': 'BCC',
                        'VolumePercent': 5,
                        'Magnetic': True,
                        'Hardness_HV': 200
                    })
            elif cr_eq > 18 and ni_eq < 8:
                # Ferritic region
                phases.append({
                    'Name': 'Ferrite',
                    'Symbol': 'alpha',
                    'Structure': 'BCC',
                    'VolumePercent': 100,
                    'Magnetic': True,
                    'Hardness_HV': 200
                })
            elif ni_eq > 8 and ni_eq < 12:
                # Duplex region
                austenite_pct = min(70, max(30, ni_eq * 5))
                phases.extend([
                    {
                        'Name': 'Austenite',
                        'Symbol': 'gamma',
                        'Structure': 'FCC',
                        'VolumePercent': int(austenite_pct),
                        'Magnetic': False,
                        'Hardness_HV': 180
                    },
                    {
                        'Name': 'Ferrite',
                        'Symbol': 'alpha',
                        'Structure': 'BCC',
                        'VolumePercent': int(100 - austenite_pct),
                        'Magnetic': True,
                        'Hardness_HV': 200
                    }
                ])
            else:
                # Martensitic or mixed
                phases.append({
                    'Name': 'Martensite',
                    'Symbol': 'alpha_prime',
                    'Structure': 'BCT',
                    'VolumePercent': 90,
                    'Magnetic': True,
                    'Hardness_HV': 300 + c_pct * 500
                })
        elif 'Al' in elements and elem_pct.get('Al', 0) > 80:
            # Aluminum alloys
            phases.append({
                'Name': 'Alpha Aluminum',
                'Symbol': 'alpha',
                'Structure': 'FCC',
                'VolumePercent': 95,
                'Magnetic': False,
                'Hardness_HV': 50
            })
            if elem_pct.get('Cu', 0) > 2:
                phases.append({
                    'Name': 'Theta (Al2Cu)',
                    'Symbol': 'theta',
                    'Structure': 'Tetragonal',
                    'VolumePercent': 5,
                    'Magnetic': False,
                    'Hardness_HV': 200
                })
        elif 'Cu' in elements and elem_pct.get('Cu', 0) > 50:
            # Copper alloys
            phases.append({
                'Name': 'Alpha Copper',
                'Symbol': 'alpha',
                'Structure': 'FCC',
                'VolumePercent': 100,
                'Magnetic': False,
                'Hardness_HV': 80
            })
        elif 'Ti' in elements and elem_pct.get('Ti', 0) > 70:
            # Titanium alloys
            al_pct = elem_pct.get('Al', 0)
            v_pct = elem_pct.get('V', 0)
            if v_pct > 3:
                # Alpha-beta alloy
                beta_pct = min(40, v_pct * 8)
                phases.extend([
                    {
                        'Name': 'Alpha Titanium',
                        'Symbol': 'alpha',
                        'Structure': 'HCP',
                        'VolumePercent': int(100 - beta_pct),
                        'Magnetic': False,
                        'Hardness_HV': 300
                    },
                    {
                        'Name': 'Beta Titanium',
                        'Symbol': 'beta',
                        'Structure': 'BCC',
                        'VolumePercent': int(beta_pct),
                        'Magnetic': False,
                        'Hardness_HV': 350
                    }
                ])
            else:
                phases.append({
                    'Name': 'Alpha Titanium',
                    'Symbol': 'alpha',
                    'Structure': 'HCP',
                    'VolumePercent': 100,
                    'Magnetic': False,
                    'Hardness_HV': 300
                })
        else:
            # Generic single phase
            phases.append({
                'Name': 'Matrix',
                'Symbol': 'matrix',
                'Structure': lattice_type,
                'VolumePercent': 100,
                'Magnetic': 'Fe' in elements or 'Ni' in elements or 'Co' in elements,
                'Hardness_HV': 150
            })

        return {
            'Phases': phases,
            'NickelEquivalent': round(ni_eq, 1),
            'ChromiumEquivalent': round(cr_eq, 1),
            'TransformationTemperatures': {
                'Ms_K': 273 + 500 - 350 * c_pct - 35 * mn_pct if c_pct > 0 else None,
                'Mf_K': 273 + 350 - 350 * c_pct - 35 * mn_pct if c_pct > 0 else None
            }
        }

    @classmethod
    def _calculate_atom_positions_in_lattice(cls, elements: List[str], weight_fractions: List[float],
                                              lattice_type: str, lattice_param: float) -> Dict:
        """
        Calculate atom positions in the crystal lattice.
        Returns representative unit cell positions with element assignments.
        """
        # Unit cell positions for different lattice types
        if lattice_type == 'FCC':
            base_positions = [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5]
            ]
        elif lattice_type == 'BCC':
            base_positions = [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5]
            ]
        elif lattice_type == 'HCP':
            base_positions = [
                [0.0, 0.0, 0.0],
                [1/3, 2/3, 0.5]
            ]
        else:
            base_positions = [[0.0, 0.0, 0.0]]

        # Assign elements to positions based on atomic fractions
        atomic_fracs = cls._weight_to_atomic_fractions(elements, weight_fractions)

        atom_positions = []
        for i, pos in enumerate(base_positions):
            # Randomly assign element based on atomic fraction (deterministic seed)
            cumulative = 0
            seed_val = sum(ord(c) for c in ''.join(elements)) + i
            rand_val = (seed_val * 1103515245 + 12345) % (2**31)
            normalized = rand_val / (2**31)

            assigned_element = elements[-1]
            for elem, af in zip(elements, atomic_fracs):
                cumulative += af
                if normalized < cumulative:
                    assigned_element = elem
                    break

            # Convert fractional to absolute coordinates
            abs_pos = [round(p * lattice_param, 2) for p in pos]

            atom_positions.append({
                'index': i,
                'element': assigned_element,
                'fractional_coordinates': [round(p, 4) for p in pos],
                'cartesian_coordinates_pm': abs_pos,
                'wyckoff_position': cls._get_wyckoff_position(lattice_type, i),
                'site_symmetry': cls._get_site_symmetry(lattice_type)
            })

        return {
            'unit_cell_atoms': atom_positions,
            'coordinate_system': 'fractional_and_cartesian',
            'units': 'pm',
            'lattice_vectors': {
                'a': [lattice_param, 0, 0],
                'b': [0, lattice_param, 0] if lattice_type != 'HCP' else [lattice_param * 0.5, lattice_param * 0.866, 0],
                'c': [0, 0, lattice_param] if lattice_type != 'HCP' else [0, 0, lattice_param * 1.633]
            },
            'space_group': cls._get_space_group(lattice_type),
            'method': 'ideal_lattice_with_random_substitution'
        }

    @classmethod
    def _get_wyckoff_position(cls, lattice_type: str, index: int) -> str:
        """Get Wyckoff position label for atom in unit cell."""
        wyckoff_map = {
            'FCC': ['4a', '4a', '4a', '4a'],
            'BCC': ['2a', '2a'],
            'HCP': ['2c', '2c']
        }
        positions = wyckoff_map.get(lattice_type, ['1a'])
        return positions[index % len(positions)]

    @classmethod
    def _get_site_symmetry(cls, lattice_type: str) -> str:
        """Get site symmetry for lattice type."""
        symmetry_map = {
            'FCC': 'm-3m',
            'BCC': 'm-3m',
            'HCP': '6/mmm'
        }
        return symmetry_map.get(lattice_type, 'unknown')

    @classmethod
    def _get_space_group(cls, lattice_type: str) -> Dict:
        """Get space group information for lattice type."""
        space_groups = {
            'FCC': {'number': 225, 'symbol': 'Fm-3m'},
            'BCC': {'number': 229, 'symbol': 'Im-3m'},
            'HCP': {'number': 194, 'symbol': 'P6_3/mmc'},
            'BCT': {'number': 139, 'symbol': 'I4/mmm'},
            'Diamond': {'number': 227, 'symbol': 'Fd-3m'}
        }
        return space_groups.get(lattice_type, {'number': 1, 'symbol': 'P1'})

    @classmethod
    def _get_atoms_per_unit_cell(cls, lattice_type: str) -> int:
        """Get number of atoms per unit cell."""
        atoms_map = {
            'FCC': 4,
            'BCC': 2,
            'HCP': 2,
            'BCT': 2,
            'Diamond': 8
        }
        return atoms_map.get(lattice_type, 1)

    @classmethod
    def _calculate_defect_concentrations(cls, elements: List[str], weight_fractions: List[float],
                                          melting_point: float) -> Dict:
        """
        Calculate equilibrium defect concentrations.
        Uses Arrhenius relationship: C = exp(-E_f / kT)
        """
        # Formation energies (eV) - typical values
        E_vacancy = 1.0  # Typical vacancy formation energy
        E_interstitial = 2.5  # Higher for interstitials

        # Room temperature (300 K)
        T = 300
        kT = 0.0259  # eV at 300K

        # Equilibrium vacancy concentration
        c_vacancy = math.exp(-E_vacancy / kT)
        c_interstitial = math.exp(-E_interstitial / kT)

        # Solute effects on vacancies
        num_solutes = len([wf for wf in weight_fractions if wf > 0.01 and wf < 0.5])
        vacancy_enhancement = 1 + 0.1 * num_solutes

        return {
            'vacancy_concentration': {
                'equilibrium_at_300K': c_vacancy * vacancy_enhancement,
                'formation_energy_eV': E_vacancy,
                'migration_energy_eV': 0.5 * E_vacancy
            },
            'interstitial_concentration': {
                'equilibrium_at_300K': c_interstitial,
                'formation_energy_eV': E_interstitial
            },
            'dislocation_density': {
                'annealed_m-2': 1e10,
                'cold_worked_m-2': 1e14,
                'typical_m-2': 1e12
            },
            'stacking_fault_energy_mJ_m2': cls._estimate_stacking_fault_energy(elements, weight_fractions),
            'method': 'Arrhenius_equilibrium_thermodynamics'
        }

    @classmethod
    def _estimate_stacking_fault_energy(cls, elements: List[str], weight_fractions: List[float]) -> float:
        """Estimate stacking fault energy from composition."""
        # Base SFE values for common elements (mJ/m^2)
        sfe_base = {
            'Al': 166, 'Cu': 78, 'Ni': 128, 'Fe': 180, 'Co': 15,
            'Ag': 22, 'Au': 45, 'Ti': 300, 'Cr': 40
        }

        weighted_sfe = 0
        for elem, wf in zip(elements, weight_fractions):
            sfe = sfe_base.get(elem, 100)
            weighted_sfe += wf * sfe

        # Alloying typically reduces SFE
        num_components = len([wf for wf in weight_fractions if wf > 0.01])
        reduction = 0.9 ** (num_components - 1)

        return round(weighted_sfe * reduction, 1)

    @classmethod
    def _calculate_grain_boundary_data(cls, elements: List[str], weight_fractions: List[float],
                                        density: float) -> Dict:
        """
        Calculate grain boundary properties.
        """
        # Grain boundary energy (J/m^2) - typical for metals
        gb_energy = 0.5  # Typical high-angle grain boundary

        # Segregation coefficients for common elements
        segregation_coeff = {}
        for elem, wf in zip(elements, weight_fractions):
            if wf < 0.5:  # Solute elements segregate more
                segregation_coeff[elem] = round(1.0 + (0.5 - wf) * 2, 2)
            else:
                segregation_coeff[elem] = 1.0

        return {
            'high_angle_gb': {
                'energy_J_m2': gb_energy,
                'mobility_m4_Js': 1e-13,
                'misorientation_threshold_deg': 15
            },
            'low_angle_gb': {
                'energy_J_m2': gb_energy * 0.3,
                'typical_misorientation_deg': 5
            },
            'segregation_coefficients': segregation_coeff,
            'triple_junction_density': {
                'typical_per_mm2': 1000,
                'formula': '2 * grain_density'
            },
            'grain_boundary_area_per_volume_mm-1': round(2 / 0.05, 1),  # Assuming 50um grain size
            'method': 'Read_Shockley_model_with_segregation'
        }

    @classmethod
    def _calculate_solid_solution_strengthening(cls, elements: List[str], weight_fractions: List[float]) -> Dict:
        """Calculate solid solution strengthening contribution."""
        # Strengthening coefficients (MPa per wt%)
        strengthening_coeff = {
            'C': 800, 'N': 700, 'Si': 60, 'Mn': 25, 'Mo': 35, 'W': 30,
            'V': 40, 'Cr': 15, 'Ni': 12, 'Cu': 30, 'Al': 20
        }

        total_strengthening = 0
        contributions = []
        for elem, wf in zip(elements, weight_fractions):
            coeff = strengthening_coeff.get(elem, 10)
            contrib = coeff * wf * 100
            if wf < 0.5:  # Only count solute strengthening
                total_strengthening += contrib
                contributions.append({
                    'element': elem,
                    'coefficient_MPa_per_wt%': coeff,
                    'weight_percent': round(wf * 100, 2),
                    'contribution_MPa': round(contrib, 1)
                })

        return {
            'total_strengthening_MPa': round(total_strengthening, 1),
            'contributions': contributions,
            'mechanism': 'Fleischer_model',
            'formula': 'Delta_sigma = sum(k_i * c_i^(1/2))'
        }

    @classmethod
    def _calculate_precipitation_strengthening(cls, elements: List[str], weight_fractions: List[float]) -> Dict:
        """Calculate precipitation hardening potential."""
        # Elements that form precipitates
        precipitate_formers = {
            'Cu': 'theta_Al2Cu',
            'Mg': 'beta_Mg2Si',
            'Ti': 'gamma_prime_Ni3Ti',
            'Al': 'gamma_prime_Ni3Al',
            'Nb': 'gamma_double_prime_Ni3Nb',
            'V': 'VC_carbide',
            'Mo': 'Mo2C_carbide',
            'W': 'WC_carbide'
        }

        precipitates = []
        total_potential = 0
        for elem, wf in zip(elements, weight_fractions):
            if elem in precipitate_formers and wf > 0.005:
                potential = wf * 1000  # Rough estimate
                precipitates.append({
                    'element': elem,
                    'precipitate_type': precipitate_formers[elem],
                    'potential_strengthening_MPa': round(potential, 1)
                })
                total_potential += potential

        return {
            'potential_precipitates': precipitates,
            'max_strengthening_potential_MPa': round(total_potential, 1),
            'mechanism': 'Orowan_bypass_and_cutting',
            'note': 'Requires appropriate heat treatment to realize'
        }

    @classmethod
    def _calculate_grain_boundary_strengthening(cls, grain_size_um: float) -> Dict:
        """Calculate Hall-Petch grain boundary strengthening."""
        # Hall-Petch coefficient (MPa * um^0.5) - typical for steels
        k_hp = 15.0

        # sigma_y = sigma_0 + k * d^(-0.5)
        strengthening = k_hp / math.sqrt(grain_size_um)

        return {
            'grain_size_um': grain_size_um,
            'hall_petch_coefficient_MPa_um05': k_hp,
            'strengthening_MPa': round(strengthening, 1),
            'formula': 'Delta_sigma = k / sqrt(d)'
        }

    @classmethod
    def to_simulation_format(cls, alloy_data: Dict) -> Dict:
        """
        Convert alloy data to complete simulation format.
        Ensures ALL properties are captured for accurate reproduction.

        Args:
            alloy_data: Output from create_alloy_from_components()

        Returns:
            Dict with all data formatted for simulation use
        """
        return {
            'metadata': {
                'format_version': '2.0',
                'calculator': 'AlloyCalculator',
                'timestamp': None,
                'reproducible': True
            },
            'alloy': {
                'identity': {
                    'name': alloy_data.get('Name'),
                    'formula': alloy_data.get('Formula'),
                    'category': alloy_data.get('Category'),
                    'subcategory': alloy_data.get('SubCategory')
                },
                'composition': alloy_data.get('Components', []),
                'physical_properties': alloy_data.get('PhysicalProperties', {}),
                'mechanical_properties': alloy_data.get('MechanicalProperties', {}),
                'lattice_properties': alloy_data.get('LatticeProperties', {}),
                'phase_composition': alloy_data.get('PhaseComposition', {}),
                'microstructure': alloy_data.get('Microstructure', {}),
                'corrosion_resistance': alloy_data.get('CorrosionResistance', {}),
                'simulation_data': alloy_data.get('SimulationData', {})
            },
            'input_preservation': {
                'preserved_elements': alloy_data.get('SimulationData', {}).get('PreservedElements', [])
            },
            'calculation_provenance': {
                'density_method': 'rule_of_mixtures',
                'strength_method': 'predictive_physics_with_empirical_corrections',
                'phase_method': 'Schaeffler_diagram_and_empirical',
                'lattice_method': 'Vegards_law'
            }
        }


# ==================== Convenience Functions ====================

def calculate_alloy_properties(elements: List[str], weight_percents: List[float],
                                lattice: str = 'FCC', name: str = None) -> Dict:
    """
    Convenience function to calculate alloy properties.

    Args:
        elements: List of element symbols (e.g., ['Fe', 'Cr', 'Ni'])
        weight_percents: Weight percentages for each element
        lattice: Crystal structure type
        name: Optional alloy name

    Returns:
        Dictionary containing alloy properties
    """
    # Convert percentages to fractions
    weight_fractions = [wp / 100 for wp in weight_percents]

    # Create simple component data
    component_data = [{'symbol': elem} for elem in elements]

    return AlloyCalculator.create_alloy_from_components(
        component_data, weight_fractions, lattice, name
    )
