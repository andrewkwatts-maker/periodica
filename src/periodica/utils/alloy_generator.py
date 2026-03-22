"""
Alloy Generator
================
Auto-generates plausible alloy compositions from metallic elements
using metallurgical rules, Hume-Rothery solubility predictions,
and standard alloy families.
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from periodica.utils.derivation_metadata import DerivationSource, DerivationTracker
from periodica.utils.logger import get_logger
from periodica.utils.phase_diagram import BinaryPhaseDiagram

logger = get_logger('alloy_generator')

# Elements that are commonly used as alloy bases or additions
_METALLIC_ELEMENTS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Cs',
    'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Pb', 'Bi',
}

# Common base metals for alloy systems
_BASE_METALS = ['Fe', 'Al', 'Cu', 'Ni', 'Ti', 'Zn', 'Mg', 'Co', 'W']

# Standard alloy families with typical alloying elements and ranges
_STEEL_ADDITIONS = {
    'C': (0.05, 2.0, 'Interstitial'),
    'Cr': (0.5, 26.0, 'Substitutional'),
    'Ni': (0.5, 36.0, 'Substitutional'),
    'Mo': (0.2, 5.0, 'Substitutional'),
    'Mn': (0.3, 15.0, 'Substitutional'),
    'V': (0.1, 3.0, 'Substitutional'),
    'W': (0.5, 18.0, 'Substitutional'),
    'Si': (0.2, 3.0, 'Substitutional'),
    'Nb': (0.02, 1.0, 'Substitutional'),
    'Ti': (0.01, 0.5, 'Substitutional'),
    'Co': (0.5, 10.0, 'Substitutional'),
}

_ALUMINUM_ADDITIONS = {
    'Cu': (0.5, 6.5, 'Substitutional'),
    'Mg': (0.3, 6.0, 'Substitutional'),
    'Si': (0.3, 13.0, 'Substitutional'),
    'Zn': (0.5, 8.0, 'Substitutional'),
    'Mn': (0.3, 2.0, 'Substitutional'),
    'Li': (0.5, 3.0, 'Substitutional'),
    'Fe': (0.1, 1.0, 'Substitutional'),
    'Cr': (0.05, 0.5, 'Substitutional'),
}

_COPPER_ADDITIONS = {
    'Zn': (5.0, 40.0, 'Substitutional'),
    'Sn': (2.0, 15.0, 'Substitutional'),
    'Ni': (5.0, 30.0, 'Substitutional'),
    'Al': (3.0, 12.0, 'Substitutional'),
    'Be': (0.5, 3.0, 'Substitutional'),
    'Mn': (1.0, 12.0, 'Substitutional'),
    'Si': (1.0, 4.0, 'Substitutional'),
    'Pb': (1.0, 5.0, 'Substitutional'),
}

_TITANIUM_ADDITIONS = {
    'Al': (2.0, 8.0, 'Substitutional'),
    'V': (2.0, 15.0, 'Substitutional'),
    'Mo': (1.0, 15.0, 'Substitutional'),
    'Sn': (1.0, 6.0, 'Substitutional'),
    'Zr': (1.0, 6.0, 'Substitutional'),
    'Nb': (1.0, 7.0, 'Substitutional'),
    'Fe': (0.5, 2.5, 'Substitutional'),
    'Cr': (1.0, 12.0, 'Substitutional'),
}

_NICKEL_ADDITIONS = {
    'Cr': (5.0, 25.0, 'Substitutional'),
    'Fe': (2.0, 40.0, 'Substitutional'),
    'Mo': (2.0, 16.0, 'Substitutional'),
    'Co': (2.0, 15.0, 'Substitutional'),
    'Al': (1.0, 6.0, 'Substitutional'),
    'Ti': (0.5, 4.0, 'Substitutional'),
    'W': (1.0, 10.0, 'Substitutional'),
    'Nb': (1.0, 6.0, 'Substitutional'),
    'Cu': (2.0, 35.0, 'Substitutional'),
}

# Lattice structures for common metals
_METAL_STRUCTURES = {
    'Fe': 'BCC', 'Al': 'FCC', 'Cu': 'FCC', 'Ni': 'FCC', 'Ti': 'HCP',
    'Zn': 'HCP', 'Mg': 'HCP', 'Co': 'HCP', 'W': 'BCC', 'Cr': 'BCC',
    'V': 'BCC', 'Nb': 'BCC', 'Mo': 'BCC', 'Ta': 'BCC', 'Ag': 'FCC',
    'Au': 'FCC', 'Pt': 'FCC', 'Pd': 'FCC', 'Pb': 'FCC', 'Sn': 'BCT',
}

# Category names from base element
_CATEGORY_MAP = {
    'Fe': 'Steel', 'Al': 'Aluminum Alloy', 'Cu': 'Copper Alloy',
    'Ti': 'Titanium Alloy', 'Ni': 'Nickel Alloy', 'Zn': 'Zinc Alloy',
    'Mg': 'Magnesium Alloy', 'Co': 'Cobalt Alloy', 'W': 'Tungsten Alloy',
    'Pb': 'Lead Alloy', 'Sn': 'Tin Alloy', 'Ag': 'Silver Alloy',
    'Au': 'Gold Alloy', 'Pt': 'Platinum Alloy',
}


class AlloyGenerator:
    """Generates plausible alloy compositions from metallic elements."""

    def __init__(self):
        self._phase_diagram = BinaryPhaseDiagram()
        self._generated_formulas = set()

    def get_metallic_elements(self, available_elements: Optional[List[str]] = None) -> List[str]:
        """Filter for metallic elements from available elements."""
        if available_elements:
            return [e for e in available_elements if e in _METALLIC_ELEMENTS]
        return sorted(_METALLIC_ELEMENTS)

    def generate_all(
        self,
        count_limit: int = 50,
        available_elements: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        Generate a collection of plausible alloys.

        Args:
            count_limit: Maximum number of alloys to generate
            available_elements: Elements to use (defaults to all metallic)
            progress_callback: fn(percent, message) for progress updates

        Returns:
            List of alloy dicts matching the standard JSON schema
        """
        self._generated_formulas.clear()
        alloys = []

        metals = self.get_metallic_elements(available_elements)
        total_steps = 5
        step = 0

        def _report(msg):
            nonlocal step
            step += 1
            if progress_callback:
                pct = min(int(step / total_steps * 100), 99)
                progress_callback(pct, msg)

        # 1. Steel variants
        if 'Fe' in metals:
            _report("Generating steel variants...")
            alloys.extend(self.generate_steel_variants(
                count=min(count_limit // 4, 15)
            ))

        # 2. Aluminum alloys
        if 'Al' in metals:
            _report("Generating aluminum alloys...")
            alloys.extend(self.generate_aluminum_alloys(
                count=min(count_limit // 5, 10)
            ))

        # 3. Copper alloys
        if 'Cu' in metals:
            _report("Generating copper alloys...")
            alloys.extend(self.generate_copper_alloys(
                count=min(count_limit // 5, 10)
            ))

        # 4. Binary alloys from available metals
        _report("Generating binary alloys...")
        remaining = max(0, count_limit - len(alloys))
        alloys.extend(self.generate_binary_alloys(
            metals, count=min(remaining // 2, 20)
        ))

        # 5. Ternary alloys
        _report("Generating ternary alloys...")
        remaining = max(0, count_limit - len(alloys))
        alloys.extend(self.generate_ternary_alloys(
            metals, count=min(remaining, 10)
        ))

        if progress_callback:
            progress_callback(100, f"Generated {len(alloys)} alloys")

        return alloys[:count_limit]

    def generate_steel_variants(self, count: int = 10) -> List[Dict]:
        """Generate steel alloy variants (Fe-based)."""
        alloys = []
        recipes = [
            ("Low Carbon Steel", {'C': 0.15, 'Mn': 0.8, 'Si': 0.3}),
            ("Medium Carbon Steel", {'C': 0.45, 'Mn': 0.7, 'Si': 0.25}),
            ("High Carbon Steel", {'C': 1.0, 'Mn': 0.4, 'Si': 0.2}),
            ("Chromium Steel", {'C': 0.4, 'Cr': 1.5, 'Mn': 0.5}),
            ("Nickel Steel", {'C': 0.35, 'Ni': 3.5, 'Mn': 0.5}),
            ("Cr-Mo Steel", {'C': 0.3, 'Cr': 1.0, 'Mo': 0.2, 'Mn': 0.5}),
            ("Cr-Ni Steel", {'C': 0.3, 'Cr': 1.5, 'Ni': 1.5, 'Mn': 0.5}),
            ("Spring Steel", {'C': 0.6, 'Si': 2.0, 'Mn': 0.8}),
            ("Austenitic Stainless", {'C': 0.08, 'Cr': 18.0, 'Ni': 8.0, 'Mn': 2.0}),
            ("Ferritic Stainless", {'C': 0.08, 'Cr': 17.0, 'Si': 0.5}),
            ("Martensitic Stainless", {'C': 0.3, 'Cr': 13.0, 'Mn': 0.5}),
            ("Duplex Stainless", {'C': 0.03, 'Cr': 22.0, 'Ni': 5.0, 'Mo': 3.0, 'Mn': 1.5}),
            ("Tool Steel A2", {'C': 1.0, 'Cr': 5.0, 'Mo': 1.0, 'V': 0.25}),
            ("Maraging Steel", {'C': 0.03, 'Ni': 18.0, 'Co': 9.0, 'Mo': 5.0, 'Ti': 0.7}),
            ("HSLA Steel", {'C': 0.08, 'Mn': 1.5, 'Nb': 0.05, 'V': 0.1, 'Ti': 0.02}),
        ]

        for name, additions in recipes[:count]:
            alloy = self._build_alloy_from_recipe('Fe', name, additions, 'Steel')
            if alloy and alloy['Formula'] not in self._generated_formulas:
                self._generated_formulas.add(alloy['Formula'])
                alloys.append(alloy)

        return alloys

    def generate_aluminum_alloys(self, count: int = 8) -> List[Dict]:
        """Generate aluminum alloy variants."""
        alloys = []
        recipes = [
            ("Al-Cu Alloy (2xxx)", {'Cu': 4.5, 'Mg': 1.5, 'Mn': 0.6}),
            ("Al-Mn Alloy (3xxx)", {'Mn': 1.2, 'Fe': 0.5, 'Si': 0.3}),
            ("Al-Si Alloy (4xxx)", {'Si': 12.0, 'Mg': 0.5}),
            ("Al-Mg Alloy (5xxx)", {'Mg': 4.5, 'Mn': 0.7, 'Cr': 0.15}),
            ("Al-Mg-Si Alloy (6xxx)", {'Mg': 1.0, 'Si': 0.6, 'Cu': 0.25}),
            ("Al-Zn Alloy (7xxx)", {'Zn': 5.6, 'Mg': 2.5, 'Cu': 1.6}),
            ("Al-Li Alloy (8xxx)", {'Li': 2.0, 'Cu': 1.5, 'Mg': 0.8}),
            ("Al-Si Casting", {'Si': 7.0, 'Mg': 0.35}),
            ("Al-Cu-Si Casting", {'Cu': 4.0, 'Si': 5.0, 'Mg': 0.5}),
            ("Al-Zn-Mg-Cu High Strength", {'Zn': 7.5, 'Mg': 2.5, 'Cu': 2.0, 'Cr': 0.2}),
        ]

        for name, additions in recipes[:count]:
            alloy = self._build_alloy_from_recipe('Al', name, additions, 'Aluminum Alloy')
            if alloy and alloy['Formula'] not in self._generated_formulas:
                self._generated_formulas.add(alloy['Formula'])
                alloys.append(alloy)

        return alloys

    def generate_copper_alloys(self, count: int = 8) -> List[Dict]:
        """Generate copper alloy variants."""
        alloys = []
        recipes = [
            ("Alpha Brass", {'Zn': 30.0}),
            ("Beta Brass", {'Zn': 40.0}),
            ("Phosphor Bronze", {'Sn': 8.0}),
            ("Aluminum Bronze", {'Al': 10.0}),
            ("Cupronickel 70/30", {'Ni': 30.0}),
            ("Cupronickel 90/10", {'Ni': 10.0}),
            ("Beryllium Copper", {'Be': 2.0}),
            ("Silicon Bronze", {'Si': 3.0, 'Mn': 1.0}),
            ("Manganese Bronze", {'Zn': 25.0, 'Mn': 5.0, 'Al': 3.0}),
            ("Naval Brass", {'Zn': 38.0, 'Sn': 1.0}),
        ]

        for name, additions in recipes[:count]:
            alloy = self._build_alloy_from_recipe('Cu', name, additions, 'Copper Alloy')
            if alloy and alloy['Formula'] not in self._generated_formulas:
                self._generated_formulas.add(alloy['Formula'])
                alloys.append(alloy)

        return alloys

    def generate_binary_alloys(
        self,
        metals: List[str],
        count: int = 15,
        ratios: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Dict]:
        """
        Generate binary alloys from pairs of metallic elements.

        Args:
            metals: Available metallic elements
            count: Maximum number to generate
            ratios: Composition ratios to try (base%, addition%)
        """
        if ratios is None:
            ratios = [(90, 10), (80, 20), (70, 30), (50, 50)]

        alloys = []
        # Prioritize common base metals
        bases = [m for m in _BASE_METALS if m in metals]
        others = [m for m in metals if m not in bases]

        for base in bases:
            for addition in metals:
                if addition == base or len(alloys) >= count:
                    continue

                # Check Hume-Rothery solubility
                max_sol = self._phase_diagram.predict_max_solubility(addition, base)
                if max_sol < 1.0:
                    continue

                # Pick a ratio within solubility limit
                for base_pct, add_pct in ratios:
                    if add_pct <= max_sol or max_sol >= 100:
                        additions = {addition: add_pct}
                        name = f"{base}-{addition} {int(base_pct)}/{int(add_pct)}"
                        alloy = self._build_alloy_from_recipe(
                            base, name, additions,
                            _CATEGORY_MAP.get(base, 'Binary Alloy')
                        )
                        if alloy and alloy['Formula'] not in self._generated_formulas:
                            self._generated_formulas.add(alloy['Formula'])
                            alloys.append(alloy)
                            break  # One ratio per pair

                if len(alloys) >= count:
                    break

        return alloys[:count]

    def generate_ternary_alloys(
        self,
        metals: List[str],
        count: int = 10,
    ) -> List[Dict]:
        """Generate ternary alloys from triplets of metallic elements."""
        alloys = []
        bases = [m for m in _BASE_METALS if m in metals]

        for base in bases:
            candidates = [m for m in metals if m != base]
            for i, add1 in enumerate(candidates):
                for add2 in candidates[i + 1:]:
                    if len(alloys) >= count:
                        return alloys

                    # Both additions must have some solubility
                    sol1 = self._phase_diagram.predict_max_solubility(add1, base)
                    sol2 = self._phase_diagram.predict_max_solubility(add2, base)
                    if sol1 < 2.0 or sol2 < 2.0:
                        continue

                    # Use conservative amounts
                    pct1 = min(10.0, sol1 * 0.5)
                    pct2 = min(10.0, sol2 * 0.5)
                    additions = {add1: round(pct1, 1), add2: round(pct2, 1)}
                    name = f"{base}-{add1}-{add2} Ternary"

                    alloy = self._build_alloy_from_recipe(
                        base, name, additions,
                        _CATEGORY_MAP.get(base, 'Ternary Alloy')
                    )
                    if alloy and alloy['Formula'] not in self._generated_formulas:
                        self._generated_formulas.add(alloy['Formula'])
                        alloys.append(alloy)

        return alloys[:count]

    def _build_alloy_from_recipe(
        self,
        base: str,
        name: str,
        additions: Dict[str, float],
        category: str,
    ) -> Optional[Dict]:
        """
        Build a complete alloy data dict from a recipe.

        Args:
            base: Base element symbol
            name: Alloy name
            additions: Dict of {element: weight_percent}
            category: Alloy category

        Returns:
            Alloy dict matching the standard JSON schema
        """
        try:
            # Calculate base percentage
            total_additions = sum(additions.values())
            if total_additions >= 100:
                logger.warning(f"Additions exceed 100% for {name}")
                return None
            base_pct = 100.0 - total_additions

            # Build components list
            components = [{
                'Element': base,
                'MinPercent': round(base_pct - 1.0, 2),
                'MaxPercent': round(base_pct + 1.0, 2),
                'Role': 'Base',
            }]
            for elem, pct in sorted(additions.items()):
                role = 'Alloying'
                if pct < 0.5:
                    role = 'Trace'
                components.append({
                    'Element': elem,
                    'MinPercent': round(max(0, pct - pct * 0.1), 2),
                    'MaxPercent': round(pct + pct * 0.1, 2),
                    'Role': role,
                })

            # Build formula string
            formula_parts = [base]
            for elem in sorted(additions.keys()):
                formula_parts.append(elem)
            formula = '-'.join(formula_parts)

            # Calculate properties using rule of mixtures
            all_elements = [base] + list(additions.keys())
            all_fractions = [base_pct / 100.0] + [p / 100.0 for p in additions.values()]

            physical = self._estimate_physical_properties(all_elements, all_fractions)
            mechanical = self._estimate_mechanical_properties(
                all_elements, all_fractions, physical
            )
            lattice = self._estimate_lattice_properties(base, all_elements, all_fractions)

            # Subcategory from composition
            subcategory = self._determine_subcategory(base, additions)

            alloy = {
                'Name': name,
                'Formula': formula,
                'Category': category,
                'SubCategory': subcategory,
                'Description': f"Auto-generated {category.lower()} with "
                               f"{', '.join(f'{e} {p:.1f}%' for e, p in additions.items())}",
                'Components': components,
                'PhysicalProperties': physical,
                'MechanicalProperties': mechanical,
                'LatticeProperties': lattice,
                'Applications': self._suggest_applications(category, mechanical),
                'ProcessingMethods': self._suggest_processing(category),
                'Color': self._estimate_color(base, additions),
            }

            # Stamp with derivation metadata
            derived_from = [f"element:{e}" for e in all_elements]
            DerivationTracker.stamp(
                alloy,
                source=DerivationSource.AUTO_GENERATED,
                derived_from=derived_from,
                derivation_chain=['elements', 'rule_of_mixtures', 'alloy'],
                confidence=0.7,
            )

            return alloy

        except Exception as e:
            logger.error(f"Failed to build alloy {name}: {e}")
            return None

    def _estimate_physical_properties(
        self, elements: List[str], fractions: List[float]
    ) -> Dict:
        """Estimate physical properties using rule of mixtures."""
        from periodica.utils.alloy_calculator import AlloyConstants

        # Density: inverse rule of mixtures
        density = 0.0
        density_sum = 0.0
        for elem, frac in zip(elements, fractions):
            d = AlloyConstants.ELEMENT_DENSITIES.get(elem, 7.0)
            if d > 0:
                density_sum += frac / d
        density = 1.0 / density_sum if density_sum > 0 else 7.0

        # Melting point: weighted average with depression
        mp = 0.0
        for elem, frac in zip(elements, fractions):
            mp += frac * AlloyConstants.ELEMENT_MELTING_POINTS.get(elem, 1500)
        depression = max(0.85, 1.0 - 0.03 * (len(elements) - 1))
        mp *= depression

        # Thermal conductivity: weighted with reduction
        tc = 0.0
        for elem, frac in zip(elements, fractions):
            tc += frac * AlloyConstants.ELEMENT_THERMAL_CONDUCTIVITY.get(elem, 50.0)
        tc *= max(0.3, 0.7 ** (len(elements) - 1))

        # Electrical resistivity: Matthiessen's rule
        res = 0.0
        for elem, frac in zip(elements, fractions):
            res += frac * AlloyConstants.ELEMENT_RESISTIVITY.get(elem, 10.0)
        res += 5.0 * (len(elements) - 1)  # alloying scattering

        return {
            'Density_g_cm3': round(density, 3),
            'MeltingPoint_K': round(mp, 1),
            'ThermalConductivity_W_mK': round(tc, 1),
            'ThermalExpansion_per_K': round(12e-6 + 2e-6 * (len(elements) - 1), 7),
            'ElectricalResistivity_Ohm_m': round(res * 1e-8, 10),
            'SpecificHeat_J_kgK': round(500 + 20 * (len(elements) - 1), 0),
        }

    def _estimate_mechanical_properties(
        self, elements: List[str], fractions: List[float], physical: Dict
    ) -> Dict:
        """Estimate mechanical properties from composition."""
        # Base tensile strength correlates with melting point
        mp = physical.get('MeltingPoint_K', 1500)
        base_ts = 200 + (mp - 500) * 0.3

        # Solid solution strengthening: more alloying = stronger
        n_additions = len(elements) - 1
        ss_factor = 1.0 + 0.15 * n_additions

        ts = base_ts * ss_factor
        ys = ts * 0.6
        elongation = max(5, 50 - n_additions * 8)
        hardness = ts * 0.3

        return {
            'TensileStrength_MPa': round(ts, 0),
            'YieldStrength_MPa': round(ys, 0),
            'Elongation_percent': round(elongation, 1),
            'Hardness_HV': round(hardness, 0),
        }

    def _estimate_lattice_properties(
        self, base: str, elements: List[str], fractions: List[float]
    ) -> Dict:
        """Estimate lattice properties from base element."""
        from periodica.utils.alloy_calculator import AlloyConstants

        structure = _METAL_STRUCTURES.get(base, 'FCC')
        lattice_data = AlloyConstants.LATTICE_CONSTANTS.get(base, {'a': 360})
        a = lattice_data.get('a', 360)

        packing = {'FCC': 0.74, 'BCC': 0.68, 'HCP': 0.74, 'BCT': 0.70}.get(structure, 0.70)
        coord = {'FCC': 12, 'BCC': 8, 'HCP': 12, 'BCT': 8}.get(structure, 12)
        atoms = {'FCC': 4, 'BCC': 2, 'HCP': 6, 'BCT': 4}.get(structure, 4)

        c = a
        if structure == 'HCP':
            c = round(a * 1.633, 1)

        return {
            'PrimaryStructure': structure,
            'LatticeParameters': {
                'a_pm': round(a, 1),
                'b_pm': round(a, 1),
                'c_pm': round(c, 1),
                'alpha_deg': 90,
                'beta_deg': 90,
                'gamma_deg': 120 if structure == 'HCP' else 90,
            },
            'AtomicPackingFactor': packing,
            'CoordinationNumber': coord,
            'AtomsPerUnitCell': atoms,
        }

    def _determine_subcategory(self, base: str, additions: Dict[str, float]) -> str:
        """Determine alloy subcategory from composition."""
        if base == 'Fe':
            cr = additions.get('Cr', 0)
            ni = additions.get('Ni', 0)
            c = additions.get('C', 0)
            if cr >= 10.5:
                if ni >= 6:
                    return 'Austenitic Stainless Steel'
                elif ni < 1 and c < 0.1:
                    return 'Ferritic Stainless Steel'
                else:
                    return 'Martensitic Stainless Steel'
            elif c > 0.6:
                return 'High Carbon Steel'
            elif c > 0.25:
                return 'Medium Carbon Steel'
            else:
                return 'Low Carbon Steel'
        elif base == 'Al':
            if 'Cu' in additions:
                return 'Al-Cu Wrought Alloy'
            elif 'Si' in additions and additions.get('Si', 0) > 5:
                return 'Al-Si Casting Alloy'
            elif 'Zn' in additions:
                return 'Al-Zn High Strength'
            elif 'Mg' in additions:
                return 'Al-Mg Alloy'
            return 'Aluminum Wrought Alloy'
        elif base == 'Cu':
            if 'Zn' in additions:
                return 'Brass'
            elif 'Sn' in additions:
                return 'Bronze'
            elif 'Ni' in additions:
                return 'Cupronickel'
            return 'Copper Alloy'
        elif base == 'Ti':
            al = additions.get('Al', 0)
            v = additions.get('V', 0)
            if al > 0 and v > 0:
                return 'Alpha-Beta Titanium'
            elif any(e in additions for e in ('Mo', 'V', 'Cr', 'Fe')):
                return 'Beta Titanium'
            return 'Alpha Titanium'
        return f'{base}-based Alloy'

    def _suggest_applications(self, category: str, mechanical: Dict) -> List[str]:
        """Suggest applications based on category and properties."""
        apps = {
            'Steel': ['Structural components', 'Machinery', 'Construction'],
            'Aluminum Alloy': ['Aerospace', 'Automotive', 'Lightweight structures'],
            'Copper Alloy': ['Electrical connectors', 'Heat exchangers', 'Marine hardware'],
            'Titanium Alloy': ['Aerospace', 'Biomedical implants', 'Chemical processing'],
            'Nickel Alloy': ['High-temperature service', 'Chemical processing', 'Gas turbines'],
        }
        base_apps = apps.get(category, ['General engineering'])

        ts = mechanical.get('TensileStrength_MPa', 0)
        if ts > 1000:
            base_apps.append('High-strength applications')
        if ts < 300:
            base_apps.append('Formable components')

        return base_apps[:4]

    def _suggest_processing(self, category: str) -> List[str]:
        """Suggest processing methods based on category."""
        methods = {
            'Steel': ['Hot rolling', 'Forging', 'Heat treatment'],
            'Aluminum Alloy': ['Extrusion', 'Rolling', 'Casting'],
            'Copper Alloy': ['Casting', 'Cold working', 'Annealing'],
            'Titanium Alloy': ['Forging', 'Machining', 'HIP'],
            'Nickel Alloy': ['Investment casting', 'Forging', 'Powder metallurgy'],
        }
        return methods.get(category, ['Casting', 'Machining'])

    def _estimate_color(self, base: str, additions: Dict[str, float]) -> str:
        """Estimate alloy color as hex string."""
        colors = {
            'Fe': '#808080', 'Al': '#C0C0C0', 'Cu': '#B87333', 'Ti': '#878681',
            'Ni': '#A0A0A0', 'Zn': '#B0B0B0', 'Mg': '#D0D0D0', 'Co': '#9090A0',
            'W': '#707070', 'Au': '#FFD700', 'Ag': '#C0C0C0', 'Pt': '#E5E4E2',
        }
        return colors.get(base, '#A0A0A0')

    def save_alloys(
        self,
        alloys: List[Dict],
        output_dir: Optional[str] = None,
    ) -> int:
        """
        Save generated alloys to JSON files.

        Returns:
            Number of files saved
        """
        if output_dir is None:
            output_dir = str(
                Path(__file__).parent.parent / 'data' / 'active' / 'alloys'
            )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved = 0

        for alloy in alloys:
            name = alloy.get('Name', f'alloy_{saved}')
            safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            filepath = out_path / f"{safe_name}.json"

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(alloy, f, indent=2, ensure_ascii=False)
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

        logger.info(f"Saved {saved} alloys to {output_dir}")
        return saved
