#!/usr/bin/env python3
"""
Test script for validating molecular geometry calculations against
known experimental chemistry data.

Validates:
- H2O: bent, 104.5° angle, O-H bond ~0.96Å
- CH4: tetrahedral, 109.5° angles, C-H bond ~1.09Å
- CO2: linear, 180° angle, C=O bond ~1.16Å
- NH3: trigonal pyramidal, ~107° angle, N-H bond ~1.01Å
- C6H6 (benzene): planar hexagonal, 120° C-C-C angles, C-C bond ~1.40Å
"""

import sys
import math
sys.path.insert(0, '/home/user/Periodics')

from periodica.utils.molecular_geometry import (
    MolecularGeometryCalculator,
    get_molecule_structure,
    calculate_molecular_properties,
    COMMON_MOLECULES
)


# Experimental reference values from NIST and crystallographic data
EXPERIMENTAL_DATA = {
    'H2O': {
        'geometry': 'Bent',
        'bond_angle': 104.5,  # degrees
        'bond_lengths': {
            'O-H': 0.9584,  # Angstroms (experimental)
        },
        'description': 'Water molecule - bent geometry due to 2 lone pairs'
    },
    'CH4': {
        'geometry': 'Tetrahedral',
        'bond_angle': 109.47,  # degrees (tetrahedral angle)
        'bond_lengths': {
            'C-H': 1.0870,  # Angstroms
        },
        'description': 'Methane - perfect tetrahedral with sp3 carbon'
    },
    'CO2': {
        'geometry': 'Linear',
        'bond_angle': 180.0,  # degrees
        'bond_lengths': {
            'C=O': 1.1600,  # Angstroms (double bond)
        },
        'description': 'Carbon dioxide - linear with sp carbon'
    },
    'NH3': {
        'geometry': 'Trigonal Pyramidal',
        'bond_angle': 107.0,  # degrees (slightly less than tetrahedral due to lone pair)
        'bond_lengths': {
            'N-H': 1.0124,  # Angstroms
        },
        'description': 'Ammonia - trigonal pyramidal with 1 lone pair'
    },
    'C6H6': {
        'geometry': 'Planar Hexagonal',
        'bond_angle': 120.0,  # degrees (C-C-C angles)
        'bond_lengths': {
            'C-C': 1.3950,  # Angstroms (aromatic bond, between single/double)
            'C-H': 1.0870,  # Angstroms
        },
        'description': 'Benzene - planar hexagonal with delocalized pi electrons'
    }
}


class MolecularGeometryTester:
    """Tests molecular geometry calculations against experimental data."""

    def __init__(self):
        self.calculator = MolecularGeometryCalculator()
        self.results = []
        self.tolerance = 5.0  # 5% error tolerance

    def calculate_percent_error(self, expected, calculated):
        """Calculate percent error between expected and calculated values."""
        if expected == 0:
            return 0 if calculated == 0 else float('inf')
        return abs((calculated - expected) / expected) * 100

    def calculate_bond_length(self, atom1, atom2):
        """Calculate distance between two atoms."""
        dx = atom2['x'] - atom1['x']
        dy = atom2['y'] - atom1['y']
        dz = atom2['z'] - atom1['z']
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def calculate_angle(self, atom1, central, atom2):
        """Calculate angle between three atoms (atom1 - central - atom2) in degrees."""
        v1 = (atom1['x'] - central['x'], atom1['y'] - central['y'], atom1['z'] - central['z'])
        v2 = (atom2['x'] - central['x'], atom2['y'] - central['y'], atom2['z'] - central['z'])

        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    def get_atom_by_index(self, atoms, index):
        """Get atom by its original index."""
        for atom in atoms:
            if atom.get('index') == index:
                return atom
        return None

    def get_atoms_by_element(self, atoms, element):
        """Get all atoms of a specific element."""
        return [a for a in atoms if a['element'] == element]

    def test_h2o(self):
        """Test water molecule geometry."""
        print("\n" + "="*60)
        print("Testing H2O (Water)")
        print("="*60)

        exp = EXPERIMENTAL_DATA['H2O']
        structure = get_molecule_structure('H2O')

        result = {
            'molecule': 'H2O',
            'description': exp['description'],
            'tests': []
        }

        # Test geometry type
        calc_geometry = structure.get('geometry', 'Unknown')
        geo_match = calc_geometry == exp['geometry']
        result['tests'].append({
            'test': 'Geometry Type',
            'expected': exp['geometry'],
            'calculated': calc_geometry,
            'match': geo_match,
            'error': 0 if geo_match else 'MISMATCH'
        })
        print(f"  Geometry: Expected={exp['geometry']}, Calculated={calc_geometry} {'PASS' if geo_match else 'FAIL'}")

        # Test bond angle
        calc_angle = structure.get('bond_angle', 0)
        angle_error = self.calculate_percent_error(exp['bond_angle'], calc_angle)
        angle_pass = angle_error <= self.tolerance
        result['tests'].append({
            'test': 'H-O-H Bond Angle',
            'expected': exp['bond_angle'],
            'calculated': calc_angle,
            'unit': 'degrees',
            'error_percent': round(angle_error, 2),
            'pass': angle_pass
        })
        print(f"  Bond Angle: Expected={exp['bond_angle']}°, Calculated={calc_angle}° (Error: {angle_error:.2f}%) {'PASS' if angle_pass else 'FAIL'}")

        # Test O-H bond length
        atoms = structure.get('atoms', [])
        o_atom = self.get_atom_by_index(atoms, 0)  # Oxygen is central
        h_atoms = [a for a in atoms if a['element'] == 'H']

        if o_atom and h_atoms:
            oh_lengths = [self.calculate_bond_length(o_atom, h) for h in h_atoms]
            avg_oh = sum(oh_lengths) / len(oh_lengths)
            exp_oh = exp['bond_lengths']['O-H']
            oh_error = self.calculate_percent_error(exp_oh, avg_oh)
            oh_pass = oh_error <= self.tolerance
            result['tests'].append({
                'test': 'O-H Bond Length',
                'expected': exp_oh,
                'calculated': round(avg_oh, 4),
                'unit': 'Angstroms',
                'error_percent': round(oh_error, 2),
                'pass': oh_pass
            })
            print(f"  O-H Bond: Expected={exp_oh}Å, Calculated={avg_oh:.4f}Å (Error: {oh_error:.2f}%) {'PASS' if oh_pass else 'FAIL'}")

        # Verify actual calculated angle from atom positions
        if o_atom and len(h_atoms) >= 2:
            actual_angle = self.calculate_angle(h_atoms[0], o_atom, h_atoms[1])
            actual_angle_error = self.calculate_percent_error(exp['bond_angle'], actual_angle)
            actual_angle_pass = actual_angle_error <= self.tolerance
            result['tests'].append({
                'test': 'Actual H-O-H Angle (from positions)',
                'expected': exp['bond_angle'],
                'calculated': round(actual_angle, 2),
                'unit': 'degrees',
                'error_percent': round(actual_angle_error, 2),
                'pass': actual_angle_pass
            })
            print(f"  Actual Angle (from coords): Expected={exp['bond_angle']}°, Calculated={actual_angle:.2f}° (Error: {actual_angle_error:.2f}%) {'PASS' if actual_angle_pass else 'FAIL'}")

        self.results.append(result)
        return result

    def test_ch4(self):
        """Test methane molecule geometry."""
        print("\n" + "="*60)
        print("Testing CH4 (Methane)")
        print("="*60)

        exp = EXPERIMENTAL_DATA['CH4']
        structure = get_molecule_structure('CH4')

        result = {
            'molecule': 'CH4',
            'description': exp['description'],
            'tests': []
        }

        # Test geometry type
        calc_geometry = structure.get('geometry', 'Unknown')
        geo_match = calc_geometry == exp['geometry']
        result['tests'].append({
            'test': 'Geometry Type',
            'expected': exp['geometry'],
            'calculated': calc_geometry,
            'match': geo_match,
            'error': 0 if geo_match else 'MISMATCH'
        })
        print(f"  Geometry: Expected={exp['geometry']}, Calculated={calc_geometry} {'PASS' if geo_match else 'FAIL'}")

        # Test bond angle
        calc_angle = structure.get('bond_angle', 0)
        angle_error = self.calculate_percent_error(exp['bond_angle'], calc_angle)
        angle_pass = angle_error <= self.tolerance
        result['tests'].append({
            'test': 'H-C-H Bond Angle',
            'expected': exp['bond_angle'],
            'calculated': calc_angle,
            'unit': 'degrees',
            'error_percent': round(angle_error, 2),
            'pass': angle_pass
        })
        print(f"  Bond Angle: Expected={exp['bond_angle']}°, Calculated={calc_angle}° (Error: {angle_error:.2f}%) {'PASS' if angle_pass else 'FAIL'}")

        # Test C-H bond length
        atoms = structure.get('atoms', [])
        c_atom = self.get_atom_by_index(atoms, 0)  # Carbon is central
        h_atoms = [a for a in atoms if a['element'] == 'H']

        if c_atom and h_atoms:
            ch_lengths = [self.calculate_bond_length(c_atom, h) for h in h_atoms]
            avg_ch = sum(ch_lengths) / len(ch_lengths)
            exp_ch = exp['bond_lengths']['C-H']
            ch_error = self.calculate_percent_error(exp_ch, avg_ch)
            ch_pass = ch_error <= self.tolerance
            result['tests'].append({
                'test': 'C-H Bond Length',
                'expected': exp_ch,
                'calculated': round(avg_ch, 4),
                'unit': 'Angstroms',
                'error_percent': round(ch_error, 2),
                'pass': ch_pass
            })
            print(f"  C-H Bond: Expected={exp_ch}Å, Calculated={avg_ch:.4f}Å (Error: {ch_error:.2f}%) {'PASS' if ch_pass else 'FAIL'}")

        # Verify actual calculated angles from atom positions
        if c_atom and len(h_atoms) >= 2:
            angles = []
            for i in range(len(h_atoms)):
                for j in range(i+1, len(h_atoms)):
                    angle = self.calculate_angle(h_atoms[i], c_atom, h_atoms[j])
                    angles.append(angle)

            avg_angle = sum(angles) / len(angles)
            actual_angle_error = self.calculate_percent_error(exp['bond_angle'], avg_angle)
            actual_angle_pass = actual_angle_error <= self.tolerance
            result['tests'].append({
                'test': 'Actual H-C-H Angles (avg from positions)',
                'expected': exp['bond_angle'],
                'calculated': round(avg_angle, 2),
                'unit': 'degrees',
                'error_percent': round(actual_angle_error, 2),
                'pass': actual_angle_pass
            })
            print(f"  Actual Angles (from coords): Expected={exp['bond_angle']}°, Avg Calculated={avg_angle:.2f}° (Error: {actual_angle_error:.2f}%) {'PASS' if actual_angle_pass else 'FAIL'}")

        self.results.append(result)
        return result

    def test_co2(self):
        """Test carbon dioxide molecule geometry."""
        print("\n" + "="*60)
        print("Testing CO2 (Carbon Dioxide)")
        print("="*60)

        exp = EXPERIMENTAL_DATA['CO2']
        structure = get_molecule_structure('CO2')

        result = {
            'molecule': 'CO2',
            'description': exp['description'],
            'tests': []
        }

        # Test geometry type
        calc_geometry = structure.get('geometry', 'Unknown')
        geo_match = calc_geometry == exp['geometry']
        result['tests'].append({
            'test': 'Geometry Type',
            'expected': exp['geometry'],
            'calculated': calc_geometry,
            'match': geo_match,
            'error': 0 if geo_match else 'MISMATCH'
        })
        print(f"  Geometry: Expected={exp['geometry']}, Calculated={calc_geometry} {'PASS' if geo_match else 'FAIL'}")

        # Test bond angle
        calc_angle = structure.get('bond_angle', 0)
        angle_error = self.calculate_percent_error(exp['bond_angle'], calc_angle)
        angle_pass = angle_error <= self.tolerance
        result['tests'].append({
            'test': 'O-C-O Bond Angle',
            'expected': exp['bond_angle'],
            'calculated': calc_angle,
            'unit': 'degrees',
            'error_percent': round(angle_error, 2),
            'pass': angle_pass
        })
        print(f"  Bond Angle: Expected={exp['bond_angle']}°, Calculated={calc_angle}° (Error: {angle_error:.2f}%) {'PASS' if angle_pass else 'FAIL'}")

        # Test C=O bond length
        atoms = structure.get('atoms', [])
        c_atom = self.get_atom_by_index(atoms, 0)  # Carbon is central
        o_atoms = [a for a in atoms if a['element'] == 'O']

        if c_atom and o_atoms:
            co_lengths = [self.calculate_bond_length(c_atom, o) for o in o_atoms]
            avg_co = sum(co_lengths) / len(co_lengths)
            exp_co = exp['bond_lengths']['C=O']
            co_error = self.calculate_percent_error(exp_co, avg_co)
            co_pass = co_error <= self.tolerance
            result['tests'].append({
                'test': 'C=O Bond Length',
                'expected': exp_co,
                'calculated': round(avg_co, 4),
                'unit': 'Angstroms',
                'error_percent': round(co_error, 2),
                'pass': co_pass
            })
            print(f"  C=O Bond: Expected={exp_co}Å, Calculated={avg_co:.4f}Å (Error: {co_error:.2f}%) {'PASS' if co_pass else 'FAIL'}")

        # Verify actual calculated angle from atom positions
        if c_atom and len(o_atoms) >= 2:
            actual_angle = self.calculate_angle(o_atoms[0], c_atom, o_atoms[1])
            actual_angle_error = self.calculate_percent_error(exp['bond_angle'], actual_angle)
            actual_angle_pass = actual_angle_error <= self.tolerance
            result['tests'].append({
                'test': 'Actual O-C-O Angle (from positions)',
                'expected': exp['bond_angle'],
                'calculated': round(actual_angle, 2),
                'unit': 'degrees',
                'error_percent': round(actual_angle_error, 2),
                'pass': actual_angle_pass
            })
            print(f"  Actual Angle (from coords): Expected={exp['bond_angle']}°, Calculated={actual_angle:.2f}° (Error: {actual_angle_error:.2f}%) {'PASS' if actual_angle_pass else 'FAIL'}")

        self.results.append(result)
        return result

    def test_nh3(self):
        """Test ammonia molecule geometry."""
        print("\n" + "="*60)
        print("Testing NH3 (Ammonia)")
        print("="*60)

        exp = EXPERIMENTAL_DATA['NH3']
        structure = get_molecule_structure('NH3')

        result = {
            'molecule': 'NH3',
            'description': exp['description'],
            'tests': []
        }

        # Test geometry type
        calc_geometry = structure.get('geometry', 'Unknown')
        geo_match = calc_geometry == exp['geometry']
        result['tests'].append({
            'test': 'Geometry Type',
            'expected': exp['geometry'],
            'calculated': calc_geometry,
            'match': geo_match,
            'error': 0 if geo_match else 'MISMATCH'
        })
        print(f"  Geometry: Expected={exp['geometry']}, Calculated={calc_geometry} {'PASS' if geo_match else 'FAIL'}")

        # Test bond angle
        calc_angle = structure.get('bond_angle', 0)
        angle_error = self.calculate_percent_error(exp['bond_angle'], calc_angle)
        angle_pass = angle_error <= self.tolerance
        result['tests'].append({
            'test': 'H-N-H Bond Angle',
            'expected': exp['bond_angle'],
            'calculated': calc_angle,
            'unit': 'degrees',
            'error_percent': round(angle_error, 2),
            'pass': angle_pass
        })
        print(f"  Bond Angle: Expected={exp['bond_angle']}°, Calculated={calc_angle}° (Error: {angle_error:.2f}%) {'PASS' if angle_pass else 'FAIL'}")

        # Test N-H bond length
        atoms = structure.get('atoms', [])
        n_atom = self.get_atom_by_index(atoms, 0)  # Nitrogen is central
        h_atoms = [a for a in atoms if a['element'] == 'H']

        if n_atom and h_atoms:
            nh_lengths = [self.calculate_bond_length(n_atom, h) for h in h_atoms]
            avg_nh = sum(nh_lengths) / len(nh_lengths)
            exp_nh = exp['bond_lengths']['N-H']
            nh_error = self.calculate_percent_error(exp_nh, avg_nh)
            nh_pass = nh_error <= self.tolerance
            result['tests'].append({
                'test': 'N-H Bond Length',
                'expected': exp_nh,
                'calculated': round(avg_nh, 4),
                'unit': 'Angstroms',
                'error_percent': round(nh_error, 2),
                'pass': nh_pass
            })
            print(f"  N-H Bond: Expected={exp_nh}Å, Calculated={avg_nh:.4f}Å (Error: {nh_error:.2f}%) {'PASS' if nh_pass else 'FAIL'}")

        # Verify actual calculated angles from atom positions
        if n_atom and len(h_atoms) >= 2:
            angles = []
            for i in range(len(h_atoms)):
                for j in range(i+1, len(h_atoms)):
                    angle = self.calculate_angle(h_atoms[i], n_atom, h_atoms[j])
                    angles.append(angle)

            avg_angle = sum(angles) / len(angles)
            actual_angle_error = self.calculate_percent_error(exp['bond_angle'], avg_angle)
            actual_angle_pass = actual_angle_error <= self.tolerance
            result['tests'].append({
                'test': 'Actual H-N-H Angles (avg from positions)',
                'expected': exp['bond_angle'],
                'calculated': round(avg_angle, 2),
                'unit': 'degrees',
                'error_percent': round(actual_angle_error, 2),
                'pass': actual_angle_pass
            })
            print(f"  Actual Angles (from coords): Expected={exp['bond_angle']}°, Avg Calculated={avg_angle:.2f}° (Error: {actual_angle_error:.2f}%) {'PASS' if actual_angle_pass else 'FAIL'}")

        # Check lone pairs
        lone_pairs = structure.get('lone_pairs', 0)
        lp_pass = lone_pairs == 1
        result['tests'].append({
            'test': 'Lone Pairs',
            'expected': 1,
            'calculated': lone_pairs,
            'pass': lp_pass
        })
        print(f"  Lone Pairs: Expected=1, Calculated={lone_pairs} {'PASS' if lp_pass else 'FAIL'}")

        self.results.append(result)
        return result

    def test_benzene(self):
        """Test benzene molecule geometry."""
        print("\n" + "="*60)
        print("Testing C6H6 (Benzene)")
        print("="*60)

        exp = EXPERIMENTAL_DATA['C6H6']

        # Benzene is not in COMMON_MOLECULES, so we need to define it
        benzene_composition = [
            {'element': 'C'}, {'element': 'C'}, {'element': 'C'},
            {'element': 'C'}, {'element': 'C'}, {'element': 'C'},
            {'element': 'H'}, {'element': 'H'}, {'element': 'H'},
            {'element': 'H'}, {'element': 'H'}, {'element': 'H'}
        ]

        # Benzene bonds - aromatic ring with alternating double bonds (simplified model)
        # In reality, benzene has delocalized electrons making all C-C bonds equal
        benzene_bonds = [
            # Carbon ring
            {'from': 0, 'to': 1, 'type': 'aromatic'},
            {'from': 1, 'to': 2, 'type': 'aromatic'},
            {'from': 2, 'to': 3, 'type': 'aromatic'},
            {'from': 3, 'to': 4, 'type': 'aromatic'},
            {'from': 4, 'to': 5, 'type': 'aromatic'},
            {'from': 5, 'to': 0, 'type': 'aromatic'},
            # C-H bonds
            {'from': 0, 'to': 6, 'type': 'single'},
            {'from': 1, 'to': 7, 'type': 'single'},
            {'from': 2, 'to': 8, 'type': 'single'},
            {'from': 3, 'to': 9, 'type': 'single'},
            {'from': 4, 'to': 10, 'type': 'single'},
            {'from': 5, 'to': 11, 'type': 'single'}
        ]

        result = {
            'molecule': 'C6H6',
            'description': exp['description'],
            'tests': []
        }

        # For benzene, manually calculate the ideal geometry
        # This is a special case due to its planar hexagonal structure

        # Calculate ideal benzene positions manually
        cc_bond = exp['bond_lengths']['C-C']  # aromatic C-C bond
        ch_bond = exp['bond_lengths']['C-H']

        # Carbon atoms in a hexagon
        carbon_positions = []
        for i in range(6):
            angle = math.radians(i * 60)
            x = cc_bond * math.cos(angle)
            y = cc_bond * math.sin(angle)
            carbon_positions.append({'element': 'C', 'x': x, 'y': y, 'z': 0.0, 'index': i})

        # Test calculated vs expected bond length using the bond length from the calculator
        calculator = MolecularGeometryCalculator()

        # The calculator should use aromatic bond length - let's check what it gives
        # First check if aromatic is recognized
        try:
            calc_cc = calculator.get_bond_length('C', 'C', 'aromatic')
        except:
            # Fall back to estimate between single and double
            calc_cc = (calculator.get_bond_length('C', 'C', 'single') +
                      calculator.get_bond_length('C', 'C', 'double')) / 2

        exp_cc = exp['bond_lengths']['C-C']
        cc_error = self.calculate_percent_error(exp_cc, calc_cc)
        cc_pass = cc_error <= self.tolerance
        result['tests'].append({
            'test': 'C-C Aromatic Bond Length',
            'expected': exp_cc,
            'calculated': round(calc_cc, 4),
            'unit': 'Angstroms',
            'error_percent': round(cc_error, 2),
            'pass': cc_pass
        })
        print(f"  C-C Bond: Expected={exp_cc}Å, Calculated={calc_cc:.4f}Å (Error: {cc_error:.2f}%) {'PASS' if cc_pass else 'FAIL'}")

        calc_ch = calculator.get_bond_length('C', 'H', 'single')
        exp_ch = exp['bond_lengths']['C-H']
        ch_error = self.calculate_percent_error(exp_ch, calc_ch)
        ch_pass = ch_error <= self.tolerance
        result['tests'].append({
            'test': 'C-H Bond Length',
            'expected': exp_ch,
            'calculated': round(calc_ch, 4),
            'unit': 'Angstroms',
            'error_percent': round(ch_error, 2),
            'pass': ch_pass
        })
        print(f"  C-H Bond: Expected={exp_ch}Å, Calculated={calc_ch:.4f}Å (Error: {ch_error:.2f}%) {'PASS' if ch_pass else 'FAIL'}")

        # Test that the trigonal planar geometry produces 120 degree angles
        # This is what each carbon in benzene should have (sp2 hybridization)
        geo_name, geo_angle = calculator.determine_geometry(3, 0)  # 3 bonds, 0 lone pairs

        geo_match = 'Trigonal Planar' in geo_name
        result['tests'].append({
            'test': 'Geometry Type (sp2 carbon)',
            'expected': 'Trigonal Planar',
            'calculated': geo_name,
            'match': geo_match,
            'error': 0 if geo_match else 'MISMATCH'
        })
        print(f"  Geometry (sp2): Expected=Trigonal Planar, Calculated={geo_name} {'PASS' if geo_match else 'FAIL'}")

        angle_error = self.calculate_percent_error(exp['bond_angle'], geo_angle)
        angle_pass = angle_error <= self.tolerance
        result['tests'].append({
            'test': 'C-C-C Bond Angle',
            'expected': exp['bond_angle'],
            'calculated': geo_angle,
            'unit': 'degrees',
            'error_percent': round(angle_error, 2),
            'pass': angle_pass
        })
        print(f"  Bond Angle: Expected={exp['bond_angle']}°, Calculated={geo_angle}° (Error: {angle_error:.2f}%) {'PASS' if angle_pass else 'FAIL'}")

        self.results.append(result)
        return result

    def generate_report(self):
        """Generate a comprehensive accuracy report."""
        print("\n")
        print("="*70)
        print("MOLECULAR GEOMETRY ACCURACY REPORT")
        print("="*70)

        total_tests = 0
        passed_tests = 0
        failed_tests = []
        errors_above_threshold = []

        for result in self.results:
            print(f"\n{result['molecule']}: {result['description']}")
            print("-" * 50)

            for test in result['tests']:
                total_tests += 1
                is_pass = test.get('pass', test.get('match', False))

                if is_pass:
                    passed_tests += 1
                    status = "PASS"
                else:
                    status = "FAIL"
                    failed_tests.append({
                        'molecule': result['molecule'],
                        'test': test['test'],
                        'expected': test.get('expected'),
                        'calculated': test.get('calculated'),
                        'error': test.get('error_percent', test.get('error'))
                    })

                # Track significant errors
                if 'error_percent' in test and test['error_percent'] > self.tolerance:
                    errors_above_threshold.append({
                        'molecule': result['molecule'],
                        'test': test['test'],
                        'expected': test.get('expected'),
                        'calculated': test.get('calculated'),
                        'error_percent': test['error_percent']
                    })

                if 'error_percent' in test:
                    print(f"  {test['test']}: {test.get('expected')} vs {test.get('calculated')} "
                          f"(Error: {test['error_percent']:.2f}%) [{status}]")
                else:
                    print(f"  {test['test']}: {test.get('expected')} vs {test.get('calculated')} [{status}]")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"\nOverall Accuracy: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        print(f"Error Tolerance: {self.tolerance}%")

        if failed_tests:
            print(f"\n*** FAILED TESTS ({len(failed_tests)}): ***")
            for ft in failed_tests:
                print(f"  - {ft['molecule']}: {ft['test']}")
                print(f"    Expected: {ft['expected']}, Calculated: {ft['calculated']}, Error: {ft['error']}")

        if errors_above_threshold:
            print(f"\n*** ERRORS ABOVE {self.tolerance}% THRESHOLD ({len(errors_above_threshold)}): ***")
            for err in errors_above_threshold:
                print(f"  - {err['molecule']}: {err['test']}")
                print(f"    Expected: {err['expected']}, Calculated: {err['calculated']}, Error: {err['error_percent']:.2f}%")

        # Recommendations
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)

        if pass_rate >= 95:
            print("\nExcellent! The molecular geometry calculations are highly accurate.")
        elif pass_rate >= 80:
            print("\nGood accuracy overall, but some improvements could be made.")
        else:
            print("\nSignificant improvements are needed for accurate molecular geometry calculations.")

        if errors_above_threshold:
            print("\nSpecific issues to address:")

            # Analyze patterns in errors
            bond_length_errors = [e for e in errors_above_threshold if 'Bond Length' in e['test']]
            angle_errors = [e for e in errors_above_threshold if 'Angle' in e['test']]

            if bond_length_errors:
                print(f"\n  Bond Length Errors ({len(bond_length_errors)}):")
                for e in bond_length_errors:
                    print(f"    - {e['molecule']} {e['test']}: {e['error_percent']:.2f}% error")
                    print(f"      Recommendation: Update BOND_LENGTHS constant for this bond type")

            if angle_errors:
                print(f"\n  Bond Angle Errors ({len(angle_errors)}):")
                for e in angle_errors:
                    print(f"    - {e['molecule']} {e['test']}: {e['error_percent']:.2f}% error")
                    print(f"      Recommendation: Check GEOMETRY_CONFIG or position calculation logic")

        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': pass_rate,
            'failed_tests': failed_tests,
            'errors_above_threshold': errors_above_threshold
        }


def main():
    """Run all molecular geometry accuracy tests."""
    print("="*70)
    print("MOLECULAR GEOMETRY ACCURACY VALIDATION")
    print("Testing against experimental chemistry reference data")
    print("="*70)

    tester = MolecularGeometryTester()

    # Run all tests
    tester.test_h2o()
    tester.test_ch4()
    tester.test_co2()
    tester.test_nh3()
    tester.test_benzene()

    # Generate report
    report = tester.generate_report()

    return report


if __name__ == '__main__':
    main()
