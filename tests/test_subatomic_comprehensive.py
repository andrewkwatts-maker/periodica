#!/usr/bin/env python3
"""
Comprehensive Test Script for SubatomicCalculator
==================================================
Tests the SubatomicCalculator against ALL default subatomic particles,
validates predictions, tests exotic hadron combinations, and provides
detailed analysis with improvement recommendations.

Author: Claude Code Analysis
Date: 2024
"""

import sys
import os
import json
import statistics
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from periodica.utils.physics_calculator import SubatomicCalculator, PhysicsConstants


# ============================================================================
# Data Classes for Test Results
# ============================================================================

@dataclass
class ParticleTestResult:
    """Result of testing a single particle."""
    name: str
    quarks: List[str]
    spin_aligned: bool
    actual_mass: float
    predicted_mass: float
    mass_error_pct: float
    actual_charge: float
    predicted_charge: float
    charge_correct: bool
    actual_spin: float
    predicted_spin: float
    spin_correct: bool
    actual_baryon_num: int
    predicted_baryon_num: int
    baryon_correct: bool
    particle_type: str

    @property
    def overall_pass(self) -> bool:
        return self.charge_correct and self.spin_correct and self.baryon_correct and self.mass_error_pct < 15


@dataclass
class RunStatistics:
    """Statistics from a test run."""
    total_particles: int
    charge_accuracy: float
    spin_accuracy: float
    baryon_accuracy: float
    mass_errors: List[float]
    mean_mass_error: float
    median_mass_error: float
    max_mass_error: float
    min_mass_error: float
    std_mass_error: float
    worst_particles: List[ParticleTestResult]
    best_particles: List[ParticleTestResult]


# ============================================================================
# Quark Symbol Mapping
# ============================================================================

QUARK_NAME_TO_SYMBOL = {
    'Up Quark': 'u',
    'Down Quark': 'd',
    'Strange Quark': 's',
    'Charm Quark': 'c',
    'Bottom Quark': 'b',
    'Top Quark': 't',
    'Anti-Up Quark': 'u\u0305',
    'Anti-Down Quark': 'd\u0305',
    'Anti-Strange Quark': 's\u0305',
    'Anti-Charm Quark': 'c\u0305',
    'Anti-Bottom Quark': 'b\u0305',
    'Anti-Top Quark': 't\u0305',
    'Antiup Quark': 'u\u0305',
    'Antidown Quark': 'd\u0305',
    'Antistrange Quark': 's\u0305',
    'Anticharm Quark': 'c\u0305',
    'Antibottom Quark': 'b\u0305',
    'Antitop Quark': 't\u0305',
}

# Additional mappings for symbol fields
SYMBOL_TO_QUARK = {
    'u': 'u', 'd': 'd', 's': 's', 'c': 'c', 'b': 'b', 't': 't',
    'u-bar': 'u\u0305', 'd-bar': 'd\u0305', 's-bar': 's\u0305',
    'c-bar': 'c\u0305', 'b-bar': 'b\u0305', 't-bar': 't\u0305',
    'anti-u': 'u\u0305', 'anti-d': 'd\u0305', 'anti-s': 's\u0305',
    'anti-c': 'c\u0305', 'anti-b': 'b\u0305', 'anti-t': 't\u0305',
}


def parse_quark_composition(composition: List[Dict]) -> Tuple[List[str], bool]:
    """
    Parse quark composition from particle JSON.
    Returns (quarks_list, is_spin_aligned).
    """
    quarks = []

    for item in composition:
        constituent = item.get('Constituent', '')
        symbol = item.get('Symbol', '')
        count = item.get('Count', 1)
        is_anti = item.get('IsAnti', False)

        # Try to get quark symbol from constituent name
        quark_symbol = None

        # First try the symbol field
        if symbol:
            if symbol in SYMBOL_TO_QUARK:
                quark_symbol = SYMBOL_TO_QUARK[symbol]
            elif len(symbol) == 1 and symbol in 'udscbt':
                quark_symbol = symbol
                if is_anti:
                    quark_symbol = symbol + '\u0305'

        # Fall back to constituent name
        if not quark_symbol:
            for name, sym in QUARK_NAME_TO_SYMBOL.items():
                if name.lower() in constituent.lower():
                    quark_symbol = sym
                    break

            # Handle anti-quarks from constituent name
            if not quark_symbol and 'anti' in constituent.lower():
                base = constituent.lower().replace('anti-', '').replace('anti', '').strip()
                for name, sym in QUARK_NAME_TO_SYMBOL.items():
                    if base in name.lower() and '\u0305' in sym:
                        quark_symbol = sym
                        break

        if quark_symbol:
            quarks.extend([quark_symbol] * count)

    return quarks, False  # Default to not spin-aligned (ground state)


def load_particle_data(json_path: str) -> Optional[Dict]:
    """Load particle data from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Remove JavaScript-style comments
            lines = content.split('\n')
            clean_lines = []
            for line in lines:
                # Remove // comments
                if '//' in line:
                    # Find the // that's not in a string
                    in_string = False
                    result = []
                    i = 0
                    while i < len(line):
                        if line[i] == '"' and (i == 0 or line[i-1] != '\\'):
                            in_string = not in_string
                        if not in_string and line[i:i+2] == '//':
                            break
                        result.append(line[i])
                        i += 1
                    line = ''.join(result)
                clean_lines.append(line)
            content = '\n'.join(clean_lines)
            return json.loads(content)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def load_all_particles(data_dir: str) -> Dict[str, Dict]:
    """Load all particle JSON files from a directory."""
    particles = {}
    subatomic_dir = os.path.join(data_dir, 'subatomic')

    if not os.path.exists(subatomic_dir):
        print(f"Warning: Directory not found: {subatomic_dir}")
        return particles

    for filename in os.listdir(subatomic_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(subatomic_dir, filename)
            data = load_particle_data(filepath)
            if data:
                name = data.get('Name', filename.replace('.json', ''))
                particles[name] = data

    return particles


# ============================================================================
# Main Testing Functions
# ============================================================================

def _check_particle(particle_data: Dict, is_resonance: bool = False) -> Optional[ParticleTestResult]:
    """Test a single particle against the calculator."""
    name = particle_data.get('Name', 'Unknown')
    composition = particle_data.get('Composition', [])

    if not composition:
        return None

    quarks, spin_aligned_default = parse_quark_composition(composition)

    if not quarks:
        return None

    # Get actual values
    actual_mass = particle_data.get('Mass_MeVc2', 0)
    actual_charge = particle_data.get('Charge_e', 0)
    actual_spin = particle_data.get('Spin_hbar', 0)
    actual_baryon = particle_data.get('BaryonNumber_B', 0)

    # Determine if spin-aligned (resonances have spin 3/2 for baryons, 1 for mesons)
    spin_aligned = actual_spin == 1.5 or (len(quarks) == 2 and actual_spin == 1)

    # Calculate predictions
    predicted_charge = SubatomicCalculator.calculate_charge(quarks)
    predicted_mass = SubatomicCalculator.calculate_mass(quarks, spin_aligned=spin_aligned)
    predicted_spin = SubatomicCalculator.calculate_spin(quarks, aligned=spin_aligned)
    predicted_baryon = SubatomicCalculator.calculate_baryon_number(quarks)
    particle_type = SubatomicCalculator.determine_particle_type(quarks)

    # Calculate errors
    mass_error_pct = abs(predicted_mass - actual_mass) / actual_mass * 100 if actual_mass > 0 else 0
    charge_correct = abs(predicted_charge - actual_charge) < 0.01
    spin_correct = abs(predicted_spin - actual_spin) < 0.01
    baryon_correct = predicted_baryon == actual_baryon

    return ParticleTestResult(
        name=name,
        quarks=quarks,
        spin_aligned=spin_aligned,
        actual_mass=actual_mass,
        predicted_mass=predicted_mass,
        mass_error_pct=mass_error_pct,
        actual_charge=actual_charge,
        predicted_charge=predicted_charge,
        charge_correct=charge_correct,
        actual_spin=actual_spin,
        predicted_spin=predicted_spin,
        spin_correct=spin_correct,
        actual_baryon_num=actual_baryon,
        predicted_baryon_num=predicted_baryon,
        baryon_correct=baryon_correct,
        particle_type=particle_type
    )


def run_default_particle_tests(particles: Dict[str, Dict]) -> Tuple[List[ParticleTestResult], RunStatistics]:
    """Run tests on all default particles and compute statistics."""
    results = []

    for name, data in particles.items():
        result = _check_particle(data)
        if result:
            results.append(result)

    if not results:
        return [], None

    # Compute statistics
    mass_errors = [r.mass_error_pct for r in results]
    sorted_by_error = sorted(results, key=lambda r: r.mass_error_pct, reverse=True)

    stats = RunStatistics(
        total_particles=len(results),
        charge_accuracy=sum(1 for r in results if r.charge_correct) / len(results) * 100,
        spin_accuracy=sum(1 for r in results if r.spin_correct) / len(results) * 100,
        baryon_accuracy=sum(1 for r in results if r.baryon_correct) / len(results) * 100,
        mass_errors=mass_errors,
        mean_mass_error=statistics.mean(mass_errors),
        median_mass_error=statistics.median(mass_errors),
        max_mass_error=max(mass_errors),
        min_mass_error=min(mass_errors),
        std_mass_error=statistics.stdev(mass_errors) if len(mass_errors) > 1 else 0,
        worst_particles=sorted_by_error[:5],
        best_particles=sorted_by_error[-5:],
    )

    return results, stats


# ============================================================================
# Novel/Exotic Quark Combination Tests
# ============================================================================

@dataclass
class ExoticParticleTest:
    """Test result for an exotic quark combination."""
    name: str
    quarks: List[str]
    spin_aligned: bool
    predicted_mass: float
    predicted_charge: float
    predicted_spin: float
    predicted_baryon: int
    particle_type: str
    expected_mass_range: Tuple[float, float]  # (min, max) expected from theory
    is_reasonable: bool
    notes: str


def check_exotic_hadrons() -> List[ExoticParticleTest]:
    """Test novel quark combinations that may not exist in defaults."""
    exotic_tests = []

    # Define exotic particles with expected mass ranges from theory
    exotic_definitions = [
        # Triple-strange already exists (Omega-), but let's verify
        {
            'name': 'Omega- (sss baryon)',
            'quarks': ['s', 's', 's'],
            'spin_aligned': True,
            'expected_range': (1600, 1750),
            'notes': 'Triple strange baryon, should be ~1672 MeV'
        },
        # Triple-charm baryon (Omega_ccc)
        {
            'name': 'Omega_ccc (ccc baryon)',
            'quarks': ['c', 'c', 'c'],
            'spin_aligned': True,
            'expected_range': (4700, 5100),
            'notes': 'Triple charm baryon, predicted ~4800 MeV, not yet observed'
        },
        # Triple-bottom baryon (Omega_bbb)
        {
            'name': 'Omega_bbb (bbb baryon)',
            'quarks': ['b', 'b', 'b'],
            'spin_aligned': True,
            'expected_range': (14000, 15000),
            'notes': 'Triple bottom baryon, predicted ~14.4 GeV, not yet observed'
        },
        # Doubly-charmed baryon (Xi_cc)
        {
            'name': 'Xi_cc++ (ccu baryon)',
            'quarks': ['c', 'c', 'u'],
            'spin_aligned': False,
            'expected_range': (3500, 3700),
            'notes': 'Doubly charmed baryon, observed at ~3621 MeV'
        },
        # Charmed-strange baryon
        {
            'name': 'Omega_c (css baryon)',
            'quarks': ['c', 's', 's'],
            'spin_aligned': False,
            'expected_range': (2650, 2750),
            'notes': 'Charmed strange baryon, observed at ~2695 MeV'
        },
        # Bottom-strange meson (Bs)
        {
            'name': 'Bs meson (b s-bar)',
            'quarks': ['b', 's\u0305'],
            'spin_aligned': False,
            'expected_range': (5300, 5400),
            'notes': 'Bottom-strange meson, observed at ~5367 MeV'
        },
        # Bc meson (bottom-charm)
        {
            'name': 'Bc meson (b c-bar)',
            'quarks': ['b', 'c\u0305'],
            'spin_aligned': False,
            'expected_range': (6200, 6350),
            'notes': 'Bottom-charm meson, observed at ~6275 MeV'
        },
        # Charmonium (J/psi) - vector meson
        {
            'name': 'J/psi (c c-bar vector)',
            'quarks': ['c', 'c\u0305'],
            'spin_aligned': True,
            'expected_range': (3000, 3200),
            'notes': 'Charmonium vector meson, observed at 3097 MeV'
        },
        # Bottomonium (Upsilon) - vector meson
        {
            'name': 'Upsilon (b b-bar vector)',
            'quarks': ['b', 'b\u0305'],
            'spin_aligned': True,
            'expected_range': (9300, 9600),
            'notes': 'Bottomonium vector meson, observed at 9460 MeV'
        },
        # Tetraquark (X(3872) like)
        {
            'name': 'X(3872) tetraquark',
            'quarks': ['c', 'c\u0305', 'u', 'u\u0305'],
            'spin_aligned': False,
            'expected_range': (3700, 4000),
            'notes': 'Tetraquark candidate, observed at 3872 MeV'
        },
        # Pentaquark (LHCb Pc)
        {
            'name': 'Pc pentaquark (uudc c-bar)',
            'quarks': ['u', 'u', 'd', 'c', 'c\u0305'],
            'spin_aligned': False,
            'expected_range': (4300, 4500),
            'notes': 'Pentaquark, observed by LHCb at ~4380-4450 MeV'
        },
        # All-bottom tetraquark
        {
            'name': 'All-bottom tetraquark',
            'quarks': ['b', 'b', 'b\u0305', 'b\u0305'],
            'spin_aligned': False,
            'expected_range': (18000, 20000),
            'notes': 'Fully bottom tetraquark, predicted ~18.8 GeV'
        },
        # Double-charm tetraquark (Tcc)
        {
            'name': 'Tcc+ tetraquark (ccu d-bar)',
            'quarks': ['c', 'c', 'u\u0305', 'd\u0305'],
            'spin_aligned': False,
            'expected_range': (3800, 3900),
            'notes': 'Double-charm tetraquark, discovered 2021 at ~3875 MeV'
        },
        # Hybrid charmonium-like
        {
            'name': 'Y(4260) exotic',
            'quarks': ['c', 'c\u0305', 'u', 'u\u0305'],
            'spin_aligned': True,
            'expected_range': (4100, 4400),
            'notes': 'Exotic charmonium candidate at 4260 MeV'
        },
    ]

    for definition in exotic_definitions:
        quarks = definition['quarks']
        spin_aligned = definition['spin_aligned']

        # Calculate predictions
        predicted_mass = SubatomicCalculator.calculate_mass(quarks, spin_aligned=spin_aligned)
        predicted_charge = SubatomicCalculator.calculate_charge(quarks)
        predicted_spin = SubatomicCalculator.calculate_spin(quarks, aligned=spin_aligned)
        predicted_baryon = SubatomicCalculator.calculate_baryon_number(quarks)
        particle_type = SubatomicCalculator.determine_particle_type(quarks)

        min_expected, max_expected = definition['expected_range']
        is_reasonable = min_expected <= predicted_mass <= max_expected

        exotic_tests.append(ExoticParticleTest(
            name=definition['name'],
            quarks=quarks,
            spin_aligned=spin_aligned,
            predicted_mass=predicted_mass,
            predicted_charge=predicted_charge,
            predicted_spin=predicted_spin,
            predicted_baryon=predicted_baryon,
            particle_type=particle_type,
            expected_mass_range=(min_expected, max_expected),
            is_reasonable=is_reasonable,
            notes=definition['notes']
        ))

    return exotic_tests


# ============================================================================
# Analysis and Recommendations
# ============================================================================

def analyze_mass_formula_issues(results: List[ParticleTestResult]) -> Dict[str, Any]:
    """Analyze patterns in mass prediction errors."""
    analysis = {
        'by_particle_type': defaultdict(list),
        'by_quark_flavor': defaultdict(list),
        'overestimates': [],
        'underestimates': [],
        'patterns': []
    }

    for r in results:
        # Group by type
        analysis['by_particle_type'][r.particle_type].append(r)

        # Group by quark flavor
        has_strange = 's' in ''.join(r.quarks) or 's\u0305' in ''.join(r.quarks)
        has_charm = 'c' in ''.join(r.quarks) or 'c\u0305' in ''.join(r.quarks)
        has_bottom = 'b' in ''.join(r.quarks) or 'b\u0305' in ''.join(r.quarks)

        if has_bottom:
            analysis['by_quark_flavor']['bottom'].append(r)
        elif has_charm:
            analysis['by_quark_flavor']['charm'].append(r)
        elif has_strange:
            analysis['by_quark_flavor']['strange'].append(r)
        else:
            analysis['by_quark_flavor']['light'].append(r)

        # Track over/underestimates
        if r.predicted_mass > r.actual_mass:
            analysis['overestimates'].append(r)
        else:
            analysis['underestimates'].append(r)

    # Identify patterns
    for ptype, particles in analysis['by_particle_type'].items():
        if particles:
            avg_error = statistics.mean([p.mass_error_pct for p in particles])
            analysis['patterns'].append(f"{ptype}: avg error {avg_error:.1f}%")

    return analysis


def generate_recommendations(results: List[ParticleTestResult],
                             exotic_results: List[ExoticParticleTest],
                             analysis: Dict[str, Any]) -> List[str]:
    """Generate specific code improvement recommendations."""
    recommendations = []

    # Check baryon mass formula
    baryons = [r for r in results if r.particle_type == 'Baryon']
    if baryons:
        baryon_errors = [r.mass_error_pct for r in baryons]
        avg_baryon_error = statistics.mean(baryon_errors)

        if avg_baryon_error > 5:
            recommendations.append(
                f"BARYON MASS FORMULA: Average error is {avg_baryon_error:.1f}%. "
                "Consider adjusting:\n"
                "  - BASE_NUCLEON_MASS (currently 939.0 MeV)\n"
                "  - STRANGE_CONTRIBUTION (currently 180.0 MeV, consider ~175 MeV)\n"
                "  - Add hyperfine splitting term for spin-spin interactions\n"
                "  - Add isospin-dependent corrections"
            )

    # Check meson mass formula
    mesons = [r for r in results if r.particle_type == 'Meson']
    if mesons:
        meson_errors = [r.mass_error_pct for r in mesons]
        avg_meson_error = statistics.mean(meson_errors)

        if avg_meson_error > 5:
            pions = [r for r in mesons if 'Pion' in r.name]
            kaons = [r for r in mesons if 'Kaon' in r.name]

            if pions:
                pion_avg = statistics.mean([r.mass_error_pct for r in pions])
                recommendations.append(
                    f"PION MASS: Average pion error is {pion_avg:.1f}%. "
                    f"PION_MASS constant should be ~137 MeV (currently {SubatomicCalculator._calculate_meson_mass(0,0,0,0,False):.0f} MeV)"
                )

            if kaons:
                kaon_avg = statistics.mean([r.mass_error_pct for r in kaons])
                recommendations.append(
                    f"KAON MASS: Average kaon error is {kaon_avg:.1f}%. "
                    "KAON_MASS constant should be ~494 MeV"
                )

    # Check exotic hadron predictions
    exotic_failures = [e for e in exotic_results if not e.is_reasonable]
    if exotic_failures:
        recommendations.append(
            f"EXOTIC HADRONS: {len(exotic_failures)}/{len(exotic_results)} exotic predictions outside expected range.\n"
            "  - Consider using constituent quark masses for tetraquarks/pentaquarks\n"
            "  - Add QCD string tension term for multi-quark states\n"
            "  - Heavy quark symmetry corrections needed for charm/bottom"
        )

    # Check specific quark flavors
    strange_particles = analysis['by_quark_flavor'].get('strange', [])
    if strange_particles:
        strange_avg = statistics.mean([r.mass_error_pct for r in strange_particles])
        if strange_avg > 8:
            recommendations.append(
                f"STRANGE QUARK: Average error for strange hadrons is {strange_avg:.1f}%.\n"
                "  - STRANGE_CONTRIBUTION may need adjustment (currently 180 MeV)\n"
                "  - Consider different values for baryons vs mesons\n"
                "  - Add SU(3) breaking corrections"
            )

    charm_particles = analysis['by_quark_flavor'].get('charm', [])
    if charm_particles:
        charm_avg = statistics.mean([r.mass_error_pct for r in charm_particles])
        if charm_avg > 5:
            recommendations.append(
                f"CHARM QUARK: Average error for charm hadrons is {charm_avg:.1f}%.\n"
                "  - Review CHARM_CONTRIBUTION (currently 1350 MeV for baryons)\n"
                "  - D_MESON_MASS may need tuning (currently 1870 MeV, should be ~1870)\n"
                "  - Consider spin-dependent terms for charmonium states"
            )

    bottom_particles = analysis['by_quark_flavor'].get('bottom', [])
    if bottom_particles:
        bottom_avg = statistics.mean([r.mass_error_pct for r in bottom_particles])
        if bottom_avg > 3:
            recommendations.append(
                f"BOTTOM QUARK: Average error for bottom hadrons is {bottom_avg:.1f}%.\n"
                "  - Review BOTTOM_CONTRIBUTION (currently 4680 MeV for baryons)\n"
                "  - B_MESON_MASS needs verification\n"
                "  - Upsilon states may need special handling"
            )

    # General QCD recommendations
    recommendations.append(
        "MISSING QCD EFFECTS that could improve accuracy:\n"
        "  1. Color hyperfine interaction (spin-spin splitting)\n"
        "  2. Spin-orbit coupling for excited states\n"
        "  3. Tensor force contributions\n"
        "  4. Running of strong coupling constant with mass scale\n"
        "  5. Chiral symmetry breaking effects for light quarks"
    )

    # Code improvement suggestions
    recommendations.append(
        "SUGGESTED CODE CHANGES:\n"
        "  1. Add hyperfine_correction(quarks, spin_aligned) function\n"
        "  2. Separate constituent quark masses from current quark masses\n"
        "  3. Add isospin multiplet corrections\n"
        "  4. Implement Gell-Mann-Okubo mass formula for baryon octets\n"
        "  5. Add color factor for exotic multiquark states"
    )

    return recommendations


# ============================================================================
# Report Generation
# ============================================================================

def print_detailed_report(results: List[ParticleTestResult],
                          stats: RunStatistics,
                          exotic_results: List[ExoticParticleTest],
                          analysis: Dict[str, Any],
                          recommendations: List[str]):
    """Print a comprehensive test report."""

    print("\n" + "="*100)
    print("COMPREHENSIVE SUBATOMIC CALCULATOR TEST REPORT")
    print("="*100)

    # Section 1: Summary Statistics
    print("\n" + "-"*100)
    print("1. OVERALL ACCURACY STATISTICS")
    print("-"*100)
    print(f"  Total Particles Tested: {stats.total_particles}")
    print(f"  Charge Prediction Accuracy: {stats.charge_accuracy:.1f}%")
    print(f"  Spin Prediction Accuracy: {stats.spin_accuracy:.1f}%")
    print(f"  Baryon Number Accuracy: {stats.baryon_accuracy:.1f}%")
    print(f"\n  Mass Prediction Errors:")
    print(f"    Mean Error: {stats.mean_mass_error:.2f}%")
    print(f"    Median Error: {stats.median_mass_error:.2f}%")
    print(f"    Std Deviation: {stats.std_mass_error:.2f}%")
    print(f"    Min Error: {stats.min_mass_error:.2f}%")
    print(f"    Max Error: {stats.max_mass_error:.2f}%")

    # Section 2: Detailed Results for All Particles
    print("\n" + "-"*100)
    print("2. DETAILED RESULTS FOR ALL DEFAULT PARTICLES")
    print("-"*100)
    print(f"{'Particle':<15} {'Quarks':<12} {'Actual':>10} {'Predicted':>10} {'Error%':>8} {'Chg':>5} {'Spin':>5} {'B#':>4} {'Pass':>6}")
    print("-"*100)

    for r in sorted(results, key=lambda x: x.mass_error_pct, reverse=True):
        quark_str = ''.join(r.quarks)[:10]
        chg_status = 'OK' if r.charge_correct else 'FAIL'
        spin_status = 'OK' if r.spin_correct else 'FAIL'
        baryon_status = 'OK' if r.baryon_correct else 'FAIL'
        overall = 'PASS' if r.overall_pass else 'FAIL'

        print(f"{r.name:<15} {quark_str:<12} {r.actual_mass:>10.2f} {r.predicted_mass:>10.2f} "
              f"{r.mass_error_pct:>7.1f}% {chg_status:>5} {spin_status:>5} {baryon_status:>4} {overall:>6}")

    # Section 3: Worst Performing Particles
    print("\n" + "-"*100)
    print("3. WORST PERFORMING PARTICLES (HIGHEST MASS ERRORS)")
    print("-"*100)

    for r in stats.worst_particles:
        print(f"\n  {r.name} ({r.particle_type})")
        print(f"    Quarks: {r.quarks}")
        print(f"    Actual Mass: {r.actual_mass:.2f} MeV")
        print(f"    Predicted Mass: {r.predicted_mass:.2f} MeV")
        print(f"    Error: {r.mass_error_pct:.2f}%")
        print(f"    Spin Aligned: {r.spin_aligned}")

        # Analyze why
        if r.predicted_mass > r.actual_mass:
            print(f"    Issue: OVERESTIMATE by {r.predicted_mass - r.actual_mass:.2f} MeV")
        else:
            print(f"    Issue: UNDERESTIMATE by {r.actual_mass - r.predicted_mass:.2f} MeV")

    # Section 4: Exotic Hadron Tests
    print("\n" + "-"*100)
    print("4. EXOTIC/NOVEL QUARK COMBINATION TESTS")
    print("-"*100)
    print(f"{'Name':<30} {'Quarks':<15} {'Predicted':>10} {'Expected':>15} {'Type':>12} {'Reasonable':>12}")
    print("-"*100)

    for e in exotic_results:
        quark_str = ''.join(e.quarks)[:13]
        expected_str = f"{e.expected_mass_range[0]:.0f}-{e.expected_mass_range[1]:.0f}"
        reasonable = 'YES' if e.is_reasonable else 'NO'
        print(f"{e.name:<30} {quark_str:<15} {e.predicted_mass:>10.1f} {expected_str:>15} {e.particle_type:>12} {reasonable:>12}")

    print("\n  Detailed Exotic Particle Analysis:")
    for e in exotic_results:
        status = "REASONABLE" if e.is_reasonable else "OUTSIDE EXPECTED RANGE"
        print(f"\n  {e.name}:")
        print(f"    {e.notes}")
        print(f"    Predicted: {e.predicted_mass:.1f} MeV, Charge: {e.predicted_charge:.2f}, Spin: {e.predicted_spin}")
        print(f"    Expected Range: {e.expected_mass_range[0]:.0f}-{e.expected_mass_range[1]:.0f} MeV")
        print(f"    Status: {status}")

    # Section 5: Analysis by Category
    print("\n" + "-"*100)
    print("5. ANALYSIS BY PARTICLE CATEGORY")
    print("-"*100)

    for ptype, particles in analysis['by_particle_type'].items():
        if particles:
            errors = [p.mass_error_pct for p in particles]
            print(f"\n  {ptype}s ({len(particles)} particles):")
            print(f"    Mean Error: {statistics.mean(errors):.2f}%")
            print(f"    Range: {min(errors):.2f}% - {max(errors):.2f}%")

    print("\n  By Quark Flavor:")
    for flavor, particles in analysis['by_quark_flavor'].items():
        if particles:
            errors = [p.mass_error_pct for p in particles]
            print(f"    {flavor.capitalize()}: {len(particles)} particles, mean error {statistics.mean(errors):.2f}%")

    # Section 6: Recommendations
    print("\n" + "-"*100)
    print("6. SPECIFIC RECOMMENDATIONS FOR IMPROVING THE CALCULATOR")
    print("-"*100)

    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec}")

    # Section 7: Summary
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)

    total_pass = sum(1 for r in results if r.overall_pass)
    exotic_pass = sum(1 for e in exotic_results if e.is_reasonable)

    print(f"\n  Default Particles: {total_pass}/{len(results)} passed all tests ({total_pass/len(results)*100:.1f}%)")
    print(f"  Exotic Particles: {exotic_pass}/{len(exotic_results)} within expected range ({exotic_pass/len(exotic_results)*100:.1f}%)")
    print(f"  Mean Mass Error: {stats.mean_mass_error:.2f}%")
    print(f"  Charge Accuracy: {stats.charge_accuracy:.1f}%")
    print(f"  Spin Accuracy: {stats.spin_accuracy:.1f}%")

    if stats.mean_mass_error < 5:
        print("\n  Overall Assessment: GOOD - Mass predictions are reasonably accurate")
    elif stats.mean_mass_error < 10:
        print("\n  Overall Assessment: FAIR - Some mass formula refinements needed")
    else:
        print("\n  Overall Assessment: NEEDS IMPROVEMENT - Significant formula adjustments required")

    print("\n" + "="*100)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main test runner."""
    # Determine data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data', 'defaults')

    print(f"Loading particles from: {data_dir}")

    # Load all particles
    particles = load_all_particles(data_dir)
    print(f"Loaded {len(particles)} particles from JSON files")

    if not particles:
        print("ERROR: No particle data loaded. Check data directory.")
        return False

    # Run tests on default particles
    print("\nRunning tests on default particles...")
    results, stats = run_default_particle_tests(particles)

    if not results:
        print("ERROR: No test results generated.")
        return False

    # Run exotic hadron tests
    print("Testing exotic quark combinations...")
    exotic_results = check_exotic_hadrons()

    # Analyze results
    print("Analyzing results...")
    analysis = analyze_mass_formula_issues(results)

    # Generate recommendations
    recommendations = generate_recommendations(results, exotic_results, analysis)

    # Print detailed report
    print_detailed_report(results, stats, exotic_results, analysis, recommendations)

    # Return success/failure
    return stats.mean_mass_error < 15 and stats.charge_accuracy == 100


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
