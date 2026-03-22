"""
Molecule Generator
===================
Generates valid molecular structures from available elements using
bonding rules, valence constraints, and VSEPR geometry prediction.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from periodica.utils.bonding_rules import BondingRulesEngine
from periodica.utils.derivation_metadata import DerivationSource, DerivationTracker
from periodica.utils.logger import get_logger

logger = get_logger('molecule_generator')

# VSEPR geometry lookup (bonding_pairs, lone_pairs) -> geometry
_VSEPR_GEOMETRY = {
    (2, 0): ("Linear", 180.0),
    (2, 1): ("Bent", 120.0),
    (2, 2): ("Bent", 104.5),
    (3, 0): ("Trigonal Planar", 120.0),
    (3, 1): ("Trigonal Pyramidal", 107.0),
    (4, 0): ("Tetrahedral", 109.5),
    (4, 1): ("Seesaw", 90.0),
    (5, 0): ("Trigonal Bipyramidal", 90.0),
    (6, 0): ("Octahedral", 90.0),
    (1, 0): ("Linear", 180.0),
    (1, 1): ("Linear", 180.0),
    (1, 2): ("Linear", 180.0),
    (1, 3): ("Linear", 180.0),
}

# Common molecule templates for organic chemistry
_ALKANE_TEMPLATES = [
    # (name, formula_dict, n_carbons)
    ("Methane", {"C": 1, "H": 4}, 1),
    ("Ethane", {"C": 2, "H": 6}, 2),
    ("Propane", {"C": 3, "H": 8}, 3),
    ("Butane", {"C": 4, "H": 10}, 4),
    ("Pentane", {"C": 5, "H": 12}, 5),
    ("Hexane", {"C": 6, "H": 14}, 6),
]

_ALCOHOL_TEMPLATES = [
    ("Methanol", {"C": 1, "H": 4, "O": 1}),
    ("Ethanol", {"C": 2, "H": 6, "O": 1}),
    ("Propanol", {"C": 3, "H": 8, "O": 1}),
]

_ACID_TEMPLATES = [
    ("Formic Acid", {"C": 1, "H": 2, "O": 2}),
    ("Acetic Acid", {"C": 2, "H": 4, "O": 2}),
]


class MoleculeGenerator:
    """
    Generates valid molecules from available elements.

    Uses bonding rules, valence constraints, and VSEPR predictions
    to create molecular structures with computed properties.
    """

    def __init__(self):
        self.rules = BondingRulesEngine()
        self._generated_formulas: Set[str] = set()

    def generate_all(
        self,
        available_elements: Optional[List[str]] = None,
        count_limit: int = 100,
        max_atoms: int = 8,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[Dict]:
        """
        Generate molecules from available elements.

        Args:
            available_elements: Element symbols to use. Defaults to common nonmetals.
            count_limit: Maximum number of molecules to generate.
            max_atoms: Maximum atoms per molecule.
            progress_callback: Called with (percent, status_message).

        Returns:
            List of molecule dicts matching existing JSON schema.
        """
        if available_elements is None:
            available_elements = [
                'H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'P', 'Si', 'B',
                'Na', 'K', 'Ca', 'Mg', 'Al', 'Fe', 'Cu', 'Zn', 'Br', 'I',
            ]

        self._generated_formulas.clear()
        molecules = []

        generators = [
            ("hydrides", self._generate_hydrides),
            ("oxides", self._generate_oxides),
            ("halides", self._generate_halides),
            ("binary compounds", self._generate_binary_compounds),
            ("organic compounds", self._generate_common_organics),
        ]

        total_steps = len(generators)
        for i, (name, gen_func) in enumerate(generators):
            if len(molecules) >= count_limit:
                break

            if progress_callback:
                pct = int((i / total_steps) * 100)
                progress_callback(pct, f"Generating {name}...")

            new_mols = gen_func(available_elements, max_atoms)
            for mol in new_mols:
                if len(molecules) >= count_limit:
                    break
                formula = mol.get('Formula', '')
                if formula not in self._generated_formulas:
                    self._generated_formulas.add(formula)
                    molecules.append(mol)

        if progress_callback:
            progress_callback(100, f"Generated {len(molecules)} molecules")

        logger.info("Generated %d molecules", len(molecules))
        return molecules

    def _generate_hydrides(self, elements: List[str], max_atoms: int) -> List[Dict]:
        """Generate hydrides: XHn for each element."""
        if 'H' not in elements:
            return []

        molecules = []
        for sym in elements:
            if sym == 'H':
                # H2
                molecules.append(self._build_molecule(
                    "Hydrogen", {"H": 2}, "Diatomic"
                ))
                continue

            if not self.rules.can_bond(sym, 'H'):
                continue

            valence = self.rules.get_valence(sym)
            if valence <= 0 or valence > max_atoms - 1:
                continue

            composition = {sym: 1, 'H': valence}
            name = f"{self._element_name(sym)} Hydride"

            # Special names for common hydrides
            special_names = {
                'O': ('Water', {"H": 2, "O": 1}),
                'N': ('Ammonia', {"N": 1, "H": 3}),
                'C': ('Methane', {"C": 1, "H": 4}),
                'S': ('Hydrogen Sulfide', {"H": 2, "S": 1}),
                'F': ('Hydrogen Fluoride', {"H": 1, "F": 1}),
                'Cl': ('Hydrogen Chloride', {"H": 1, "Cl": 1}),
                'Br': ('Hydrogen Bromide', {"H": 1, "Br": 1}),
                'I': ('Hydrogen Iodide', {"H": 1, "I": 1}),
            }

            if sym in special_names:
                name, composition = special_names[sym]

            molecules.append(self._build_molecule(name, composition))

        return molecules

    def _generate_oxides(self, elements: List[str], max_atoms: int) -> List[Dict]:
        """Generate oxides: XOn for each element."""
        if 'O' not in elements:
            return []

        molecules = []
        for sym in elements:
            if sym == 'O':
                # O2 and O3
                molecules.append(self._build_molecule("Oxygen", {"O": 2}, "Diatomic"))
                molecules.append(self._build_molecule("Ozone", {"O": 3}))
                continue

            if sym == 'H':
                continue  # Water handled in hydrides

            if not self.rules.can_bond(sym, 'O'):
                continue

            ratios = self.rules.get_valid_binary_ratios(sym, 'O')
            for n_sym, n_o in ratios:
                if n_sym + n_o > max_atoms:
                    continue

                composition = {sym: n_sym, 'O': n_o}
                formula = self.rules.get_formula(composition)

                special_names = {
                    'CO2': 'Carbon Dioxide', 'CO': 'Carbon Monoxide',
                    'SO2': 'Sulfur Dioxide', 'SO3': 'Sulfur Trioxide',
                    'NO2': 'Nitrogen Dioxide', 'NO': 'Nitric Oxide',
                    'N2O': 'Nitrous Oxide', 'P2O5': 'Phosphorus Pentoxide',
                    'Al2O3': 'Aluminum Oxide', 'Fe2O3': 'Iron(III) Oxide',
                    'CaO': 'Calcium Oxide', 'MgO': 'Magnesium Oxide',
                    'SiO2': 'Silicon Dioxide', 'Na2O': 'Sodium Oxide',
                }
                name = special_names.get(formula, f"{self._element_name(sym)} Oxide")
                molecules.append(self._build_molecule(name, composition))

        return molecules

    def _generate_halides(self, elements: List[str], max_atoms: int) -> List[Dict]:
        """Generate halides: XFn, XCln for each element."""
        halogens = [h for h in ['F', 'Cl', 'Br', 'I'] if h in elements]
        if not halogens:
            return []

        molecules = []
        for sym in elements:
            if sym in ('H', 'O') or sym in halogens:
                continue
            if not any(self.rules.can_bond(sym, h) for h in halogens):
                continue

            for halogen in halogens:
                ratios = self.rules.get_valid_binary_ratios(sym, halogen)
                for n_sym, n_hal in ratios:
                    if n_sym + n_hal > max_atoms:
                        continue

                    composition = {sym: n_sym, halogen: n_hal}
                    formula = self.rules.get_formula(composition)

                    special_names = {
                        'NaCl': 'Sodium Chloride', 'NaF': 'Sodium Fluoride',
                        'KCl': 'Potassium Chloride', 'CaCl2': 'Calcium Chloride',
                        'AlCl3': 'Aluminum Chloride', 'FeCl3': 'Iron(III) Chloride',
                        'MgCl2': 'Magnesium Chloride', 'CaF2': 'Calcium Fluoride',
                        'CCl4': 'Carbon Tetrachloride', 'SiCl4': 'Silicon Tetrachloride',
                    }
                    name = special_names.get(formula, f"{self._element_name(sym)} {self._halide_name(halogen)}")
                    molecules.append(self._build_molecule(name, composition))

        return molecules

    def _generate_binary_compounds(self, elements: List[str], max_atoms: int) -> List[Dict]:
        """Generate remaining binary compounds between element pairs."""
        skip_pairs = set()
        molecules = []

        nonmetals = {'H', 'C', 'N', 'O', 'F', 'S', 'P', 'Si', 'Cl', 'Se', 'Br', 'I', 'B'}

        for i, e1 in enumerate(elements):
            for e2 in elements[i+1:]:
                pair = tuple(sorted([e1, e2]))
                if pair in skip_pairs:
                    continue
                skip_pairs.add(pair)

                if not self.rules.can_bond(e1, e2):
                    continue

                ratios = self.rules.get_valid_binary_ratios(e1, e2)
                for n1, n2 in ratios:
                    if n1 + n2 > max_atoms:
                        continue

                    composition = {e1: n1, e2: n2}
                    formula = self.rules.get_formula(composition)
                    if formula in self._generated_formulas:
                        continue

                    name = f"{self._element_name(e1)}-{self._element_name(e2)} Compound"
                    molecules.append(self._build_molecule(name, composition))

        return molecules

    def _generate_common_organics(self, elements: List[str], max_atoms: int) -> List[Dict]:
        """Generate common organic molecules if C and H are available."""
        if 'C' not in elements or 'H' not in elements:
            return []

        molecules = []

        # Alkanes: CnH(2n+2)
        for name, comp, n_c in _ALKANE_TEMPLATES:
            if sum(comp.values()) <= max_atoms:
                molecules.append(self._build_molecule(name, comp, "Organic"))

        # Alcohols (if O available)
        if 'O' in elements:
            for name, comp in _ALCOHOL_TEMPLATES:
                if sum(comp.values()) <= max_atoms:
                    molecules.append(self._build_molecule(name, comp, "Organic"))

            for name, comp in _ACID_TEMPLATES:
                if sum(comp.values()) <= max_atoms:
                    molecules.append(self._build_molecule(name, comp, "Organic"))

        return molecules

    def _build_molecule(
        self, name: str, composition: Dict[str, int], category: str = ""
    ) -> Dict:
        """Build a molecule dict matching the existing JSON schema."""
        formula = self.rules.get_formula(composition)

        # Calculate molecular mass
        mass = self._calculate_mass(composition)

        # Determine central atom (heaviest non-H)
        central = self._get_central_atom(composition)

        # Determine geometry via VSEPR
        geometry, bond_angle = self._predict_geometry(central, composition)

        # Bond type
        elements_in_mol = list(composition.keys())
        if len(elements_in_mol) == 1:
            bond_type = "Covalent"
        elif len(elements_in_mol) == 2:
            bond_type = self.rules.get_bond_type(elements_in_mol[0], elements_in_mol[1])
        else:
            # Mixed — classify by average EN difference
            bond_types = set()
            for i, e1 in enumerate(elements_in_mol):
                for e2 in elements_in_mol[i+1:]:
                    bond_types.add(self.rules.get_bond_type(e1, e2))
            if "Ionic" in bond_types:
                bond_type = "Ionic"
            elif "Polar Covalent" in bond_types:
                bond_type = "Polar Covalent"
            else:
                bond_type = "Covalent"

        # Polarity
        polarity = self._determine_polarity(composition, geometry, bond_type)

        # Dipole moment estimate
        dipole = self._estimate_dipole(composition, geometry, bond_type)

        # Category
        if not category:
            if bond_type == "Ionic":
                category = "Ionic Compound"
            elif 'C' in composition:
                category = "Organic"
            else:
                category = "Inorganic"

        # Composition list format
        comp_list = [
            {"Element": sym, "Count": count}
            for sym, count in composition.items()
        ]

        molecule = {
            "Name": name,
            "Formula": formula,
            "MolecularMass_amu": round(mass, 3),
            "BondType": bond_type,
            "Geometry": geometry,
            "BondAngle_deg": bond_angle,
            "Polarity": polarity,
            "DipoleMoment_D": round(dipole, 2),
            "Category": category,
            "Composition": comp_list,
        }

        # Stamp derivation metadata
        DerivationTracker.stamp(
            molecule,
            source=DerivationSource.AUTO_GENERATED,
            derived_from=list(composition.keys()),
            derivation_chain=["elements", "bonding_rules", "vsepr", "molecule"],
            confidence=0.8,
        )

        return molecule

    def _calculate_mass(self, composition: Dict[str, int]) -> float:
        """Calculate molecular mass from atomic masses."""
        # Standard atomic masses
        masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.18,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
            'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.38, 'Br': 79.904, 'Ag': 107.868,
            'I': 126.904, 'Au': 196.967, 'Pt': 195.084, 'Pb': 207.2,
            'Ti': 47.867, 'Cr': 51.996, 'Mn': 54.938, 'Co': 58.933, 'Ni': 58.693,
            'Se': 78.96, 'Ge': 72.64, 'As': 74.922, 'Ga': 69.723,
        }
        total = 0.0
        for sym, count in composition.items():
            total += masses.get(sym, 100.0) * count
        return total

    def _get_central_atom(self, composition: Dict[str, int]) -> str:
        """Get the central atom (least electronegative non-H)."""
        candidates = [s for s in composition if s != 'H']
        if not candidates:
            return 'H'

        # Least electronegative is typically the central atom
        return min(candidates, key=lambda s: self.rules.get_electronegativity(s))

    def _predict_geometry(self, central: str, composition: Dict[str, int]) -> Tuple[str, float]:
        """Predict molecular geometry using VSEPR."""
        if sum(composition.values()) == 2:
            return "Linear", 180.0

        if sum(composition.values()) == 1:
            return "Atomic", 0.0

        # Count bonding pairs (atoms bonded to central)
        bonding_pairs = sum(composition.values()) - composition.get(central, 1)

        # Total valence electrons on central atom (not bonding valence!)
        _TOTAL_VALENCE = {
            'H': 1, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Cl': 7, 'Br': 7, 'I': 7,
            'S': 6, 'P': 5, 'Si': 4, 'B': 3, 'Li': 1, 'Na': 1, 'K': 1,
            'Ca': 2, 'Mg': 2, 'Al': 3, 'Be': 2, 'Se': 6, 'N': 5,
        }
        total_valence = _TOTAL_VALENCE.get(central, 4)

        # Lone pairs = (total_valence - bonding_pairs) / 2
        lone_pairs = max(0, (total_valence - bonding_pairs) // 2)

        # VSEPR lookup
        key = (bonding_pairs, lone_pairs)
        if key in _VSEPR_GEOMETRY:
            return _VSEPR_GEOMETRY[key]

        # Fallback
        if bonding_pairs <= 2:
            return "Linear", 180.0
        elif bonding_pairs <= 4:
            return "Tetrahedral", 109.5
        else:
            return "Complex", 90.0

    def _determine_polarity(self, composition: Dict[str, int], geometry: str, bond_type: str) -> str:
        """Determine molecular polarity."""
        if bond_type == "Ionic":
            return "Ionic"

        if len(composition) == 1:
            return "Nonpolar"

        # Symmetric geometries can be nonpolar even with polar bonds
        if geometry in ("Linear", "Trigonal Planar", "Tetrahedral", "Octahedral"):
            # Check if all terminal atoms are the same
            non_central = {s for s in composition if s != self._get_central_atom(composition)}
            if len(non_central) == 1:
                return "Nonpolar"

        if bond_type == "Covalent":
            return "Nonpolar"

        return "Polar"

    def _estimate_dipole(self, composition: Dict[str, int], geometry: str, bond_type: str) -> float:
        """Estimate dipole moment in Debye."""
        if bond_type == "Ionic":
            return 6.0 + sum(composition.values()) * 0.5

        elements = list(composition.keys())
        if len(elements) < 2:
            return 0.0

        # Get max EN difference
        ens = [self.rules.get_electronegativity(s) for s in elements]
        delta_en = max(ens) - min(ens) if ens else 0

        # Symmetric → cancel out
        if geometry in ("Linear", "Trigonal Planar", "Octahedral") and len(set(elements)) <= 2:
            central = self._get_central_atom(composition)
            non_central = {s for s in composition if s != central}
            if len(non_central) == 1:
                return 0.0

        return round(delta_en * 1.2, 2)

    def _element_name(self, symbol: str) -> str:
        """Get element name from symbol."""
        names = {
            'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium',
            'B': 'Boron', 'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen',
            'F': 'Fluorine', 'Ne': 'Neon', 'Na': 'Sodium', 'Mg': 'Magnesium',
            'Al': 'Aluminum', 'Si': 'Silicon', 'P': 'Phosphorus', 'S': 'Sulfur',
            'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium',
            'Fe': 'Iron', 'Cu': 'Copper', 'Zn': 'Zinc', 'Br': 'Bromine',
            'Ag': 'Silver', 'I': 'Iodine', 'Au': 'Gold', 'Pt': 'Platinum',
            'Ti': 'Titanium', 'Cr': 'Chromium', 'Mn': 'Manganese',
            'Co': 'Cobalt', 'Ni': 'Nickel', 'Se': 'Selenium',
        }
        return names.get(symbol, symbol)

    def _halide_name(self, halogen: str) -> str:
        """Get halide name."""
        names = {'F': 'Fluoride', 'Cl': 'Chloride', 'Br': 'Bromide', 'I': 'Iodide'}
        return names.get(halogen, f"{halogen}ide")

    def save_molecules(
        self,
        molecules: List[Dict],
        output_dir: Optional[Path] = None,
    ) -> int:
        """Save generated molecules to JSON files."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "data" / "active" / "molecules"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for mol in molecules:
            name = mol.get('Name', 'Unknown').replace(' ', '_').replace('/', '_')
            filepath = output_dir / f"{name}.json"
            if not filepath.exists():
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(mol, f, indent=2, ensure_ascii=False)
                saved += 1

        logger.info("Saved %d new molecules to %s", saved, output_dir)
        return saved
