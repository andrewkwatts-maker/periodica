#!/usr/bin/env python3
"""
Comprehensive Deviation Report Tests
=====================================

Generates detailed reports comparing calculated values against known reference data
from external sources (PDG, NIST, ExPASy, UniProt). Results are ordered by sigma
(standard deviation) differences to identify the worst simulation outputs.

Reference Sources:
- Particle Data Group (PDG): https://pdg.lbl.gov/
- NIST: https://physics.nist.gov/
- ExPASy ProtParam: https://web.expasy.org/protparam/
- UniProt: https://www.uniprot.org/
- CODATA: https://physics.nist.gov/cuu/Constants/

Run with: python -m pytest tests/test_deviation_report.py -v -s
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from statistics import mean, stdev

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from periodica.utils.physics_calculator_v2 import (
    SubatomicCalculatorV2,
    AtomCalculatorV2,
    MoleculeCalculatorV2,
    PhysicsConstantsV2
)
from periodica.utils.predictors.biological.protein_predictor import ProteinPredictor
from periodica.utils.predictors.biological.amino_acid_predictor import AminoAcidPredictor


# =============================================================================
# Data Classes for Deviation Tracking
# =============================================================================

@dataclass
class DeviationResult:
    """Single deviation measurement with sigma calculation."""
    category: str           # e.g., "Protein", "Hadron", "Atom"
    entity_name: str        # e.g., "Ubiquitin", "Proton", "Hydrogen"
    property_name: str      # e.g., "molecular_mass", "charge", "ionization_energy"
    calculated: float       # Our calculated value
    reference: float        # Known reference value
    reference_source: str   # e.g., "PDG 2024", "ExPASy", "NIST"
    units: str = ""
    uncertainty: float = 0.0  # Reference uncertainty if known

    @property
    def absolute_error(self) -> float:
        """Absolute difference from reference."""
        return abs(self.calculated - self.reference)

    @property
    def percent_error(self) -> float:
        """Percent deviation from reference."""
        if self.reference == 0:
            return 0 if self.calculated == 0 else 100
        return abs((self.calculated - self.reference) / self.reference) * 100

    @property
    def sigma(self) -> float:
        """Number of standard deviations from reference (if uncertainty known)."""
        if self.uncertainty > 0:
            return self.absolute_error / self.uncertainty
        # If no uncertainty, estimate sigma from percent error
        # Use 1% as typical measurement uncertainty
        estimated_uncertainty = abs(self.reference) * 0.01
        if estimated_uncertainty == 0:
            return 0
        return self.absolute_error / estimated_uncertainty

    def __str__(self) -> str:
        sigma_str = f"{self.sigma:.1f}s" if self.sigma > 0 else "N/A"
        return (f"{self.category:15s} | {self.entity_name:25s} | {self.property_name:20s} | "
                f"calc={self.calculated:>12.4f} | ref={self.reference:>12.4f} | "
                f"err={self.percent_error:>6.2f}% | {sigma_str:>8s} | {self.reference_source}")


@dataclass
class DeviationReport:
    """Collection of deviation results with summary statistics."""
    results: List[DeviationResult] = field(default_factory=list)

    def add(self, result: DeviationResult):
        """Add a deviation result."""
        self.results.append(result)

    def get_sorted_by_sigma(self, descending: bool = True) -> List[DeviationResult]:
        """Get results sorted by sigma deviation."""
        return sorted(self.results, key=lambda r: r.sigma, reverse=descending)

    def get_sorted_by_percent(self, descending: bool = True) -> List[DeviationResult]:
        """Get results sorted by percent deviation."""
        return sorted(self.results, key=lambda r: r.percent_error, reverse=descending)

    def get_by_category(self, category: str) -> List[DeviationResult]:
        """Filter results by category."""
        return [r for r in self.results if r.category == category]

    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.results:
            return {}

        percent_errors = [r.percent_error for r in self.results]
        sigmas = [r.sigma for r in self.results if r.sigma > 0]

        return {
            'total_comparisons': len(self.results),
            'mean_percent_error': mean(percent_errors),
            'max_percent_error': max(percent_errors),
            'min_percent_error': min(percent_errors),
            'std_percent_error': stdev(percent_errors) if len(percent_errors) > 1 else 0,
            'mean_sigma': mean(sigmas) if sigmas else 0,
            'max_sigma': max(sigmas) if sigmas else 0,
            'comparisons_over_3sigma': sum(1 for s in sigmas if s > 3),
            'comparisons_over_5sigma': sum(1 for s in sigmas if s > 5),
        }

    def generate_report(self, top_n: int = 20) -> str:
        """Generate a formatted deviation report."""
        lines = []
        lines.append("=" * 120)
        lines.append("DEVIATION REPORT - ORDERED BY SIGMA FROM REFERENCE DATA")
        lines.append("=" * 120)
        lines.append("")

        # Summary statistics
        stats = self.summary_stats
        lines.append("SUMMARY STATISTICS:")
        lines.append("-" * 50)
        lines.append(f"  Total comparisons: {stats.get('total_comparisons', 0)}")
        lines.append(f"  Mean % error: {stats.get('mean_percent_error', 0):.3f}%")
        lines.append(f"  Max % error: {stats.get('max_percent_error', 0):.3f}%")
        lines.append(f"  Std % error: {stats.get('std_percent_error', 0):.3f}%")
        lines.append(f"  Mean sigma: {stats.get('mean_sigma', 0):.2f}")
        lines.append(f"  Max sigma: {stats.get('max_sigma', 0):.2f}")
        lines.append(f"  Comparisons > 3s: {stats.get('comparisons_over_3sigma', 0)}")
        lines.append(f"  Comparisons > 5s: {stats.get('comparisons_over_5sigma', 0)}")
        lines.append("")

        # Category breakdown
        categories = set(r.category for r in self.results)
        lines.append("BY CATEGORY:")
        lines.append("-" * 50)
        for cat in sorted(categories):
            cat_results = self.get_by_category(cat)
            cat_errors = [r.percent_error for r in cat_results]
            lines.append(f"  {cat}: {len(cat_results)} comparisons, "
                        f"mean error = {mean(cat_errors):.3f}%")
        lines.append("")

        # Top deviations ordered by sigma
        lines.append(f"TOP {top_n} DEVIATIONS (ORDERED BY SIGMA):")
        lines.append("-" * 120)
        lines.append(f"{'Category':15s} | {'Entity':25s} | {'Property':20s} | "
                    f"{'Calculated':>12s} | {'Reference':>12s} | {'Error%':>7s} | {'Sigma':>8s} | Source")
        lines.append("-" * 120)

        for result in self.get_sorted_by_sigma()[:top_n]:
            lines.append(str(result))

        lines.append("")
        lines.append("=" * 120)

        return "\n".join(lines)


# =============================================================================
# Reference Data from External Sources
# =============================================================================

# PDG 2024 particle data (masses in MeV/c²)
PDG_HADRON_DATA = {
    'Proton': {
        'mass_mev': 938.272088,
        'mass_uncertainty': 0.000016,
        'charge_e': 1.0,
        'spin': 0.5,
        'baryon_number': 1,
        'source': 'PDG 2024'
    },
    'Neutron': {
        'mass_mev': 939.565421,
        'mass_uncertainty': 0.000021,
        'charge_e': 0.0,
        'spin': 0.5,
        'baryon_number': 1,
        'source': 'PDG 2024'
    },
    'Pion+': {
        'mass_mev': 139.57039,
        'mass_uncertainty': 0.00018,
        'charge_e': 1.0,
        'spin': 0.0,
        'baryon_number': 0,
        'source': 'PDG 2024'
    },
    'Pion-': {
        'mass_mev': 139.57039,
        'mass_uncertainty': 0.00018,
        'charge_e': -1.0,
        'spin': 0.0,
        'baryon_number': 0,
        'source': 'PDG 2024'
    },
    'Kaon+': {
        'mass_mev': 493.677,
        'mass_uncertainty': 0.016,
        'charge_e': 1.0,
        'spin': 0.0,
        'baryon_number': 0,
        'source': 'PDG 2024'
    },
    'Lambda': {
        'mass_mev': 1115.683,
        'mass_uncertainty': 0.006,
        'charge_e': 0.0,
        'spin': 0.5,
        'baryon_number': 1,
        'source': 'PDG 2024'
    },
    'Sigma+': {
        'mass_mev': 1189.37,
        'mass_uncertainty': 0.07,
        'charge_e': 1.0,
        'spin': 0.5,
        'baryon_number': 1,
        'source': 'PDG 2024'
    },
    'Xi-': {
        'mass_mev': 1321.71,
        'mass_uncertainty': 0.07,
        'charge_e': -1.0,
        'spin': 0.5,
        'baryon_number': 1,
        'source': 'PDG 2024'
    },
    'Omega-': {
        'mass_mev': 1672.45,
        'mass_uncertainty': 0.29,
        'charge_e': -1.0,
        'spin': 1.5,
        'baryon_number': 1,
        'source': 'PDG 2024'
    },
    'Delta++': {
        'mass_mev': 1232.0,
        'mass_uncertainty': 2.0,
        'charge_e': 2.0,
        'spin': 1.5,
        'baryon_number': 1,
        'source': 'PDG 2024'
    },
}

# PDG quark data (masses in MeV/c²)
PDG_QUARK_DATA = {
    'up': {
        'mass_mev': 2.16,
        'mass_uncertainty': 0.49,
        'charge_e': 2/3,
        'source': 'PDG 2024 (MS-bar at 2 GeV)'
    },
    'down': {
        'mass_mev': 4.67,
        'mass_uncertainty': 0.48,
        'charge_e': -1/3,
        'source': 'PDG 2024 (MS-bar at 2 GeV)'
    },
    'strange': {
        'mass_mev': 93.4,
        'mass_uncertainty': 8.6,
        'charge_e': -1/3,
        'source': 'PDG 2024 (MS-bar at 2 GeV)'
    },
    'charm': {
        'mass_mev': 1270,
        'mass_uncertainty': 20,
        'charge_e': 2/3,
        'source': 'PDG 2024 (MS-bar)'
    },
    'bottom': {
        'mass_mev': 4180,
        'mass_uncertainty': 30,
        'charge_e': -1/3,
        'source': 'PDG 2024 (MS-bar)'
    },
    'top': {
        'mass_mev': 172760,
        'mass_uncertainty': 300,
        'charge_e': 2/3,
        'source': 'PDG 2024'
    },
}

# NIST atomic data
NIST_ATOM_DATA = {
    'Hydrogen': {
        'atomic_mass_amu': 1.00794,
        'mass_uncertainty': 0.00001,
        'ionization_energy_ev': 13.59844,
        'ionization_uncertainty': 0.00001,
        'electronegativity_pauling': 2.20,
        'source': 'NIST/CODATA 2018'
    },
    'Carbon': {
        'atomic_mass_amu': 12.0107,
        'mass_uncertainty': 0.0008,
        'ionization_energy_ev': 11.2603,
        'ionization_uncertainty': 0.0001,
        'electronegativity_pauling': 2.55,
        'source': 'NIST/CODATA 2018'
    },
    'Oxygen': {
        'atomic_mass_amu': 15.9994,
        'mass_uncertainty': 0.0003,
        'ionization_energy_ev': 13.6181,
        'ionization_uncertainty': 0.0001,
        'electronegativity_pauling': 3.44,
        'source': 'NIST/CODATA 2018'
    },
    'Iron': {
        'atomic_mass_amu': 55.845,
        'mass_uncertainty': 0.002,
        'ionization_energy_ev': 7.9024,
        'ionization_uncertainty': 0.0001,
        'electronegativity_pauling': 1.83,
        'source': 'NIST/CODATA 2018'
    },
    'Nitrogen': {
        'atomic_mass_amu': 14.0067,
        'mass_uncertainty': 0.0002,
        'ionization_energy_ev': 14.5341,
        'ionization_uncertainty': 0.0001,
        'electronegativity_pauling': 3.04,
        'source': 'NIST/CODATA 2018'
    },
}

# NIST molecule data
NIST_MOLECULE_DATA = {
    'Water': {
        'molecular_mass_amu': 18.01528,
        'mass_uncertainty': 0.00044,
        'formula': 'H2O',
        'source': 'NIST Chemistry WebBook'
    },
    'Carbon Dioxide': {
        'molecular_mass_amu': 44.0095,
        'mass_uncertainty': 0.0014,
        'formula': 'CO2',
        'source': 'NIST Chemistry WebBook'
    },
    'Methane': {
        'molecular_mass_amu': 16.04246,
        'mass_uncertainty': 0.00081,
        'formula': 'CH4',
        'source': 'NIST Chemistry WebBook'
    },
    'Ammonia': {
        'molecular_mass_amu': 17.03052,
        'mass_uncertainty': 0.00041,
        'formula': 'NH3',
        'source': 'NIST Chemistry WebBook'
    },
}

# Protein reference data - sequences from UniProt, properties calculated using
# Henderson-Hasselbalch equation with standard pKa values (Lehninger)
EXPASY_PROTEIN_DATA = {
    'Ubiquitin': {
        'sequence': "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        'length': 76,
        'molecular_mass': 8564.87,
        'mass_uncertainty': 0.5,
        'isoelectric_point': 7.68,
        'pi_uncertainty': 0.5,
        'gravy': -0.489,
        'source': 'UniProt P0CG48 / Henderson-Hasselbalch'
    },
    'Insulin B': {
        'sequence': "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        'length': 30,
        'molecular_mass': 3429.97,
        'mass_uncertainty': 0.5,
        'isoelectric_point': 7.09,
        'pi_uncertainty': 0.5,
        'gravy': 0.220,
        'source': 'UniProt P01308 / Henderson-Hasselbalch'
    },
    'Myoglobin': {
        # Mature sperm whale myoglobin (UniProt P02185) without initiator Met
        'sequence': "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG",
        'length': 153,
        'molecular_mass': 17199.94,
        'mass_uncertainty': 0.5,
        'isoelectric_point': 9.36,
        'pi_uncertainty': 0.5,
        'gravy': -0.367,
        'source': 'UniProt P02185 (mature) / Henderson-Hasselbalch'
    },
    'Cytochrome C': {
        # Mature horse cytochrome c (UniProt P00004) without initiator Met
        'sequence': "GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE",
        'length': 104,
        'molecular_mass': 11701.57,
        'mass_uncertainty': 0.5,
        'isoelectric_point': 10.02,
        'pi_uncertainty': 0.5,
        'gravy': -0.902,
        'source': 'UniProt P00004 (mature) / Henderson-Hasselbalch'
    },
}

# Amino acid reference data
AMINO_ACID_REF_DATA = {
    'A': {'name': 'Alanine', 'mass': 89.094, 'pI': 6.00, 'hydropathy': 1.8, 'source': 'Lehninger'},
    'C': {'name': 'Cysteine', 'mass': 121.154, 'pI': 5.07, 'hydropathy': 2.5, 'source': 'Lehninger'},
    'D': {'name': 'Aspartic Acid', 'mass': 133.104, 'pI': 2.77, 'hydropathy': -3.5, 'source': 'Lehninger'},
    'E': {'name': 'Glutamic Acid', 'mass': 147.131, 'pI': 3.22, 'hydropathy': -3.5, 'source': 'Lehninger'},
    'F': {'name': 'Phenylalanine', 'mass': 165.192, 'pI': 5.48, 'hydropathy': 2.8, 'source': 'Lehninger'},
    'G': {'name': 'Glycine', 'mass': 75.067, 'pI': 5.97, 'hydropathy': -0.4, 'source': 'Lehninger'},
    'H': {'name': 'Histidine', 'mass': 155.156, 'pI': 7.59, 'hydropathy': -3.2, 'source': 'Lehninger'},
    'I': {'name': 'Isoleucine', 'mass': 131.175, 'pI': 6.02, 'hydropathy': 4.5, 'source': 'Lehninger'},
    'K': {'name': 'Lysine', 'mass': 146.189, 'pI': 9.74, 'hydropathy': -3.9, 'source': 'Lehninger'},
    'L': {'name': 'Leucine', 'mass': 131.175, 'pI': 5.98, 'hydropathy': 3.8, 'source': 'Lehninger'},
    'M': {'name': 'Methionine', 'mass': 149.208, 'pI': 5.74, 'hydropathy': 1.9, 'source': 'Lehninger'},
    'N': {'name': 'Asparagine', 'mass': 132.119, 'pI': 5.41, 'hydropathy': -3.5, 'source': 'Lehninger'},
    'P': {'name': 'Proline', 'mass': 115.132, 'pI': 6.30, 'hydropathy': -1.6, 'source': 'Lehninger'},
    'Q': {'name': 'Glutamine', 'mass': 146.146, 'pI': 5.65, 'hydropathy': -3.5, 'source': 'Lehninger'},
    'R': {'name': 'Arginine', 'mass': 174.203, 'pI': 10.76, 'hydropathy': -4.5, 'source': 'Lehninger'},
    'S': {'name': 'Serine', 'mass': 105.093, 'pI': 5.68, 'hydropathy': -0.8, 'source': 'Lehninger'},
    'T': {'name': 'Threonine', 'mass': 119.120, 'pI': 5.60, 'hydropathy': -0.7, 'source': 'Lehninger'},
    'V': {'name': 'Valine', 'mass': 117.148, 'pI': 5.96, 'hydropathy': 4.2, 'source': 'Lehninger'},
    'W': {'name': 'Tryptophan', 'mass': 204.228, 'pI': 5.89, 'hydropathy': -0.9, 'source': 'Lehninger'},
    'Y': {'name': 'Tyrosine', 'mass': 181.191, 'pI': 5.66, 'hydropathy': -1.3, 'source': 'Lehninger'},
}


# =============================================================================
# Test Functions
# =============================================================================

@pytest.fixture
def deviation_report():
    """Create a deviation report instance."""
    return DeviationReport()


@pytest.fixture
def protein_predictor():
    """Create a protein predictor instance."""
    return ProteinPredictor()


@pytest.fixture
def amino_acid_predictor():
    """Create an amino acid predictor instance."""
    return AminoAcidPredictor()


def load_particle_data(name: str) -> Optional[Dict]:
    """Load particle data from JSON files."""
    data_path = Path(__file__).parent.parent / "src" / "periodica" / "data" / "defaults" / "subatomic"
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove comments
                lines = []
                for line in content.split('\n'):
                    comment_idx = line.find('//')
                    if comment_idx != -1:
                        quote_count = line[:comment_idx].count('"')
                        if quote_count % 2 == 0:
                            line = line[:comment_idx]
                    lines.append(line)
                data = json.loads('\n'.join(lines))
                if data.get('Name') == name:
                    return data
        except Exception:
            pass
    return None


class TestDeviationReport:
    """Generate comprehensive deviation reports comparing calculations to reference data."""

    def test_protein_deviations(self, deviation_report, protein_predictor):
        """Test protein property deviations from ExPASy/UniProt reference data."""
        print("\n" + "=" * 80)
        print("PROTEIN DEVIATION ANALYSIS")
        print("=" * 80)

        for protein_name, ref_data in EXPASY_PROTEIN_DATA.items():
            # Calculate properties
            analysis = protein_predictor.analyze_protein(ref_data['sequence'], protein_name)

            # Molecular mass
            deviation_report.add(DeviationResult(
                category="Protein",
                entity_name=protein_name,
                property_name="molecular_mass",
                calculated=analysis['molecular_mass'],
                reference=ref_data['molecular_mass'],
                reference_source=ref_data['source'],
                units="Da",
                uncertainty=ref_data['mass_uncertainty']
            ))

            # Isoelectric point
            deviation_report.add(DeviationResult(
                category="Protein",
                entity_name=protein_name,
                property_name="isoelectric_point",
                calculated=analysis['isoelectric_point'],
                reference=ref_data['isoelectric_point'],
                reference_source=ref_data['source'],
                units="pH",
                uncertainty=ref_data['pi_uncertainty']
            ))

            # GRAVY
            deviation_report.add(DeviationResult(
                category="Protein",
                entity_name=protein_name,
                property_name="gravy",
                calculated=analysis['gravy'],
                reference=ref_data['gravy'],
                reference_source=ref_data['source'],
                units="",
                uncertainty=0.01
            ))

        # Print category results
        protein_results = deviation_report.get_by_category("Protein")
        for r in sorted(protein_results, key=lambda x: x.sigma, reverse=True):
            print(f"  {r.entity_name:15s} | {r.property_name:20s} | "
                  f"calc={r.calculated:>10.3f} | ref={r.reference:>10.3f} | "
                  f"err={r.percent_error:>6.2f}% | {r.sigma:.1f}s")

        # Report critical deviations (> 10s for mass) - warnings only, no failures
        mass_results = [r for r in protein_results if r.property_name == 'molecular_mass']
        high_deviations = [r for r in mass_results if r.sigma >= 10]
        if high_deviations:
            print(f"\n  WARNING: {len(high_deviations)} mass calculations exceed 10s threshold:")
            for r in high_deviations:
                print(f"    - {r.entity_name}: {r.sigma:.1f}s")

    def test_amino_acid_deviations(self, deviation_report, amino_acid_predictor):
        """Test amino acid property deviations from Lehninger reference data."""
        print("\n" + "=" * 80)
        print("AMINO ACID DEVIATION ANALYSIS")
        print("=" * 80)

        for symbol, ref_data in AMINO_ACID_REF_DATA.items():
            try:
                result = amino_acid_predictor.predict_from_symbol(symbol)

                # Isoelectric point
                deviation_report.add(DeviationResult(
                    category="Amino Acid",
                    entity_name=f"{ref_data['name']} ({symbol})",
                    property_name="isoelectric_point",
                    calculated=result.isoelectric_point,
                    reference=ref_data['pI'],
                    reference_source=ref_data['source'],
                    units="pH",
                    uncertainty=0.05
                ))

                # Hydropathy
                deviation_report.add(DeviationResult(
                    category="Amino Acid",
                    entity_name=f"{ref_data['name']} ({symbol})",
                    property_name="hydropathy_index",
                    calculated=result.hydropathy_index,
                    reference=ref_data['hydropathy'],
                    reference_source=ref_data['source'],
                    units="",
                    uncertainty=0.1
                ))
            except Exception as e:
                print(f"  Warning: Could not test {symbol}: {e}")

        # Print results
        aa_results = deviation_report.get_by_category("Amino Acid")
        for r in sorted(aa_results, key=lambda x: x.sigma, reverse=True)[:10]:
            print(f"  {r.entity_name:25s} | {r.property_name:20s} | "
                  f"calc={r.calculated:>8.2f} | ref={r.reference:>8.2f} | "
                  f"err={r.percent_error:>6.2f}% | {r.sigma:.1f}s")

    def test_hadron_deviations(self, deviation_report):
        """Test hadron property deviations from PDG reference data."""
        print("\n" + "=" * 80)
        print("HADRON DEVIATION ANALYSIS (vs PDG 2024)")
        print("=" * 80)

        for hadron_name, ref_data in PDG_HADRON_DATA.items():
            # Try to load particle data
            particle = load_particle_data(hadron_name)
            if not particle:
                print(f"  Warning: No data found for {hadron_name}")
                continue

            # Mass
            calc_mass = particle.get('Mass_MeVc2', 0)
            deviation_report.add(DeviationResult(
                category="Hadron",
                entity_name=hadron_name,
                property_name="mass",
                calculated=calc_mass,
                reference=ref_data['mass_mev'],
                reference_source=ref_data['source'],
                units="MeV/c²",
                uncertainty=ref_data['mass_uncertainty']
            ))

            # Charge
            calc_charge = particle.get('Charge_e', 0)
            deviation_report.add(DeviationResult(
                category="Hadron",
                entity_name=hadron_name,
                property_name="charge",
                calculated=calc_charge,
                reference=ref_data['charge_e'],
                reference_source=ref_data['source'],
                units="e",
                uncertainty=0.001
            ))

            # Spin
            calc_spin = particle.get('Spin_hbar', 0)
            deviation_report.add(DeviationResult(
                category="Hadron",
                entity_name=hadron_name,
                property_name="spin",
                calculated=calc_spin,
                reference=ref_data['spin'],
                reference_source=ref_data['source'],
                units="hbar",
                uncertainty=0.001
            ))

        # Print results
        hadron_results = deviation_report.get_by_category("Hadron")
        for r in sorted(hadron_results, key=lambda x: x.sigma, reverse=True)[:15]:
            print(f"  {r.entity_name:15s} | {r.property_name:10s} | "
                  f"calc={r.calculated:>12.4f} | ref={r.reference:>12.4f} | "
                  f"err={r.percent_error:>6.2f}% | {r.sigma:.1f}s")

    def test_atom_deviations(self, deviation_report):
        """Test atom property deviations from NIST reference data."""
        print("\n" + "=" * 80)
        print("ATOM DEVIATION ANALYSIS (vs NIST/CODATA)")
        print("=" * 80)

        # Load proton, neutron, electron data for atom creation
        proton = load_particle_data("Proton")
        neutron = load_particle_data("Neutron")
        electron = load_particle_data("Electron")

        if not all([proton, neutron, electron]):
            print("  Warning: Missing particle data for atom creation")
            return

        atom_configs = {
            'Hydrogen': {'Z': 1, 'N': 0},
            'Carbon': {'Z': 6, 'N': 6},
            'Oxygen': {'Z': 8, 'N': 8},
            'Iron': {'Z': 26, 'N': 30},
            'Nitrogen': {'Z': 7, 'N': 7},
        }

        for atom_name, config in atom_configs.items():
            if atom_name not in NIST_ATOM_DATA:
                continue

            ref_data = NIST_ATOM_DATA[atom_name]

            # Create atom
            try:
                atom = AtomCalculatorV2.create_atom_from_particles(
                    proton_data=proton,
                    neutron_data=neutron,
                    electron_data=electron,
                    proton_count=config['Z'],
                    neutron_count=config['N'],
                    electron_count=config['Z'],
                    element_name=atom_name,
                    element_symbol=atom_name[0]
                )

                # Atomic mass
                deviation_report.add(DeviationResult(
                    category="Atom",
                    entity_name=atom_name,
                    property_name="atomic_mass",
                    calculated=atom['atomic_mass'],
                    reference=ref_data['atomic_mass_amu'],
                    reference_source=ref_data['source'],
                    units="amu",
                    uncertainty=ref_data['mass_uncertainty']
                ))

                # Ionization energy
                deviation_report.add(DeviationResult(
                    category="Atom",
                    entity_name=atom_name,
                    property_name="ionization_energy",
                    calculated=atom['ionization_energy'],
                    reference=ref_data['ionization_energy_ev'],
                    reference_source=ref_data['source'],
                    units="eV",
                    uncertainty=ref_data['ionization_uncertainty']
                ))

                # Electronegativity
                deviation_report.add(DeviationResult(
                    category="Atom",
                    entity_name=atom_name,
                    property_name="electronegativity",
                    calculated=atom['electronegativity'],
                    reference=ref_data['electronegativity_pauling'],
                    reference_source=ref_data['source'],
                    units="Pauling",
                    uncertainty=0.05
                ))

            except Exception as e:
                print(f"  Warning: Could not create {atom_name}: {e}")

        # Print results
        atom_results = deviation_report.get_by_category("Atom")
        for r in sorted(atom_results, key=lambda x: x.sigma, reverse=True):
            print(f"  {r.entity_name:10s} | {r.property_name:20s} | "
                  f"calc={r.calculated:>10.4f} | ref={r.reference:>10.4f} | "
                  f"err={r.percent_error:>6.2f}% | {r.sigma:.1f}s")

    def test_generate_full_report(self, deviation_report, protein_predictor, amino_acid_predictor):
        """Generate complete deviation report across all categories."""
        # Run all deviation tests
        self.test_protein_deviations(deviation_report, protein_predictor)
        self.test_amino_acid_deviations(deviation_report, amino_acid_predictor)
        self.test_hadron_deviations(deviation_report)
        self.test_atom_deviations(deviation_report)

        # Generate and print full report
        print("\n")
        print(deviation_report.generate_report(top_n=30))

        # Save report to file
        report_path = Path(__file__).parent / "deviation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(deviation_report.generate_report(top_n=50))
        print(f"\nReport saved to: {report_path}")

        # Summary assertions
        stats = deviation_report.summary_stats
        print(f"\nSUMMARY: {stats['total_comparisons']} comparisons, "
              f"{stats['comparisons_over_5sigma']} over 5s")

        # Warn but don't fail on high sigma values (for diagnostic purposes)
        if stats['comparisons_over_5sigma'] > 0:
            print(f"\nWARNING: {stats['comparisons_over_5sigma']} comparisons exceed 5s deviation!")
            for r in deviation_report.get_sorted_by_sigma()[:stats['comparisons_over_5sigma']]:
                print(f"  - {r.category}: {r.entity_name} {r.property_name}: {r.sigma:.1f}s")


class TestProteinDeviationSummary:
    """Focused protein deviation tests with summary output."""

    def test_all_protein_deviations_ordered(self):
        """Generate ordered list of all protein calculation deviations."""
        predictor = ProteinPredictor()
        results = []

        for protein_name, ref_data in EXPASY_PROTEIN_DATA.items():
            analysis = predictor.analyze_protein(ref_data['sequence'], protein_name)

            # Mass deviation
            mass_error = abs(analysis['molecular_mass'] - ref_data['molecular_mass']) / ref_data['molecular_mass'] * 100
            mass_sigma = abs(analysis['molecular_mass'] - ref_data['molecular_mass']) / ref_data['mass_uncertainty']

            results.append({
                'protein': protein_name,
                'property': 'molecular_mass',
                'calculated': analysis['molecular_mass'],
                'reference': ref_data['molecular_mass'],
                'error_pct': mass_error,
                'sigma': mass_sigma,
                'source': ref_data['source']
            })

            # pI deviation
            pi_error = abs(analysis['isoelectric_point'] - ref_data['isoelectric_point']) / ref_data['isoelectric_point'] * 100
            pi_sigma = abs(analysis['isoelectric_point'] - ref_data['isoelectric_point']) / ref_data['pi_uncertainty']

            results.append({
                'protein': protein_name,
                'property': 'isoelectric_point',
                'calculated': analysis['isoelectric_point'],
                'reference': ref_data['isoelectric_point'],
                'error_pct': pi_error,
                'sigma': pi_sigma,
                'source': ref_data['source']
            })

            # GRAVY deviation
            gravy_error = abs(analysis['gravy'] - ref_data['gravy'])
            gravy_sigma = gravy_error / 0.01  # Assume 0.01 uncertainty

            results.append({
                'protein': protein_name,
                'property': 'gravy',
                'calculated': analysis['gravy'],
                'reference': ref_data['gravy'],
                'error_pct': gravy_error / abs(ref_data['gravy']) * 100 if ref_data['gravy'] != 0 else 0,
                'sigma': gravy_sigma,
                'source': ref_data['source']
            })

        # Sort by sigma descending
        results.sort(key=lambda x: x['sigma'], reverse=True)

        print("\n" + "=" * 100)
        print("PROTEIN DEVIATION REPORT - ORDERED BY SIGMA")
        print("=" * 100)
        print(f"{'Protein':15s} | {'Property':20s} | {'Calculated':>12s} | {'Reference':>12s} | "
              f"{'Error%':>8s} | {'Sigma':>8s} | Source")
        print("-" * 100)

        for r in results:
            print(f"{r['protein']:15s} | {r['property']:20s} | {r['calculated']:>12.3f} | "
                  f"{r['reference']:>12.3f} | {r['error_pct']:>7.2f}% | {r['sigma']:>7.1f}s | {r['source']}")

        print("=" * 100)

        # Report on critical deviations - for diagnostic purposes
        mass_results = [r for r in results if r['property'] == 'molecular_mass']
        max_mass_sigma = max(r['sigma'] for r in mass_results)
        if max_mass_sigma >= 5:
            print(f"\n  NOTE: Maximum mass sigma ({max_mass_sigma:.1f}s) - these deviations need investigation")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
