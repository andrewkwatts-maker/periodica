"""
Gemini Asset Generator Tests

IMPORTANT: These tests call the Gemini API and are NON-DETERMINISTIC.
They should only be run when updating/validating the generation process.
They are NOT included in the regular test suite.

Run with: pytest tests/test_gemini_generator.py -v -s
Skip in CI: pytest tests/ --ignore=tests/test_gemini_generator.py
"""

import json
import pytest
import sys
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Skip all tests if API key not configured or in CI
def gemini_available():
    """Check if Gemini API is available."""
    try:
        from config import get_gemini_api_key
        key = get_gemini_api_key()
        return key is not None and key != "YOUR_GEMINI_API_KEY_HERE"
    except Exception:
        return False


# Decorator to skip tests if Gemini not available
requires_gemini = pytest.mark.skipif(
    not gemini_available(),
    reason="Gemini API key not configured"
)

# Mark all tests as slow (they call external API)
pytestmark = [pytest.mark.slow, pytest.mark.gemini]


@dataclass
class QualityMetrics:
    """Metrics for assessing generation quality."""
    has_valid_config: bool
    has_required_fields: bool
    final_score: float
    iterations_improved: bool  # Did scores improve over iterations?
    planning_succeeded: bool
    output_saved: bool
    error_message: str = ""


class TestGeminiGeneratorProtein:
    """Tests for protein generation with Gemini."""

    @requires_gemini
    def test_generate_hydrophobic_peptide(self):
        """Test generating a hydrophobic peptide with positive GRAVY."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("protein", verbose=True)

        session = generator.generate(
            user_intent="Create a short hydrophobic peptide with positive GRAVY score, suitable for membrane insertion",
            selected_components=["I", "V", "L", "A", "F"],
            target_properties={
                "gravy": 1.5,
                "length": 10,
                "isoelectric_point": 6.0
            }
        )

        metrics = self._assess_quality(session, ["name", "sequence"])
        self._print_report("Hydrophobic Peptide", session, metrics)

        # Assertions
        assert metrics.has_valid_config, f"Invalid config: {metrics.error_message}"
        assert session.best_result is not None, "No best result"
        assert session.best_result.score >= 0.3, f"Score too low: {session.best_result.score}"

        # Check the sequence contains hydrophobic residues
        if "sequence" in session.best_result.config:
            seq = session.best_result.config["sequence"]
            hydrophobic_count = sum(1 for aa in seq if aa in "IVLAFWM")
            assert hydrophobic_count > len(seq) * 0.5, "Not enough hydrophobic residues"

    @requires_gemini
    def test_generate_basic_protein(self):
        """Test generating a basic (high pI) protein."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("protein", verbose=True)

        session = generator.generate(
            user_intent="Create a basic peptide with high isoelectric point (pI > 9)",
            selected_components=["K", "R", "H", "A", "L"],
            target_properties={
                "isoelectric_point": 10.0,
                "length": 8
            }
        )

        metrics = self._assess_quality(session, ["name", "sequence"])
        self._print_report("Basic Protein", session, metrics)

        assert metrics.has_valid_config
        assert session.best_result.score >= 0.3

    @requires_gemini
    def test_generate_alpha_helix_former(self):
        """Test generating a peptide with high helix propensity."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("protein", verbose=True)

        session = generator.generate(
            user_intent="Design a peptide that forms an alpha helix, avoiding helix breakers like proline",
            selected_components=["A", "E", "L", "M", "K"],
            target_properties={
                "helix_fraction": 0.8,
                "length": 15
            }
        )

        metrics = self._assess_quality(session, ["name", "sequence"])
        self._print_report("Alpha Helix Former", session, metrics)

        assert metrics.has_valid_config
        # Check no proline in middle of sequence
        if "sequence" in session.best_result.config:
            seq = session.best_result.config["sequence"]
            # Proline should not be in middle (positions 2 to n-2)
            if len(seq) > 4:
                middle = seq[2:-2]
                assert "P" not in middle, "Proline found in middle of helix"

    def _assess_quality(self, session, required_fields: list) -> QualityMetrics:
        """Assess the quality of a generation session."""
        metrics = QualityMetrics(
            has_valid_config=False,
            has_required_fields=False,
            final_score=0.0,
            iterations_improved=False,
            planning_succeeded=False,
            output_saved=False
        )

        # Check if we have a best result
        if not session.best_result:
            metrics.error_message = "No best result generated"
            return metrics

        metrics.has_valid_config = isinstance(session.best_result.config, dict)

        # Check required fields
        if metrics.has_valid_config:
            config = session.best_result.config
            metrics.has_required_fields = all(
                field in config for field in required_fields
            )
            if not metrics.has_required_fields:
                missing = [f for f in required_fields if f not in config]
                metrics.error_message = f"Missing fields: {missing}"

        # Get final score
        metrics.final_score = session.best_result.score

        # Check if iterations improved
        if len(session.iterations) >= 2:
            first_score = session.iterations[0].score
            last_score = session.iterations[-1].score
            metrics.iterations_improved = last_score >= first_score

        # Check planning
        metrics.planning_succeeded = (
            "feasibility_assessment" in session.planning_analysis or
            "proposed_config" in session.planning_analysis
        )

        return metrics

    def _print_report(self, test_name: str, session, metrics: QualityMetrics):
        """Print a detailed report of the generation."""
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        print(f"Generation time: {session.generation_time_seconds:.1f}s")
        print(f"API calls: {session.total_api_calls}")
        print(f"Final score: {metrics.final_score:.2f}")
        print(f"Valid config: {metrics.has_valid_config}")
        print(f"Required fields: {metrics.has_required_fields}")
        print(f"Planning succeeded: {metrics.planning_succeeded}")
        print(f"Iterations improved: {metrics.iterations_improved}")

        if session.best_result:
            print(f"\nBest configuration:")
            print(json.dumps(session.best_result.config, indent=2))

        if session.iterations:
            print(f"\nScore progression:")
            for it in session.iterations:
                print(f"  Iteration {it.iteration}: {it.score:.2f} - {it.feedback[:50]}...")


class TestGeminiGeneratorNucleicAcid:
    """Tests for nucleic acid generation with Gemini."""

    @requires_gemini
    def test_generate_high_tm_primer(self):
        """Test generating a DNA primer with high melting temperature."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("nucleic_acid", verbose=True)

        session = generator.generate(
            user_intent="Design a PCR primer with high GC content for stable binding",
            selected_components=["G", "C", "A", "T"],
            target_properties={
                "gc_content": 0.6,
                "melting_temperature": 62,
                "length": 20,
                "type": "DNA"
            }
        )

        assert session.best_result is not None
        assert session.best_result.score >= 0.3

        print(f"\nGenerated primer config:")
        print(json.dumps(session.best_result.config, indent=2))

    @requires_gemini
    def test_generate_rna_sequence(self):
        """Test generating an RNA sequence."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("nucleic_acid", verbose=True)

        session = generator.generate(
            user_intent="Create a short RNA sequence with moderate stability",
            selected_components=["A", "U", "G", "C"],
            target_properties={
                "gc_content": 0.5,
                "length": 15,
                "type": "RNA"
            }
        )

        assert session.best_result is not None

        # Verify it's RNA (has U, not T)
        if "sequence" in session.best_result.config:
            seq = session.best_result.config["sequence"]
            assert "T" not in seq, "RNA should not contain T"


class TestGeminiGeneratorCell:
    """Tests for cell generation with Gemini."""

    @requires_gemini
    def test_generate_high_metabolism_cell(self):
        """Test generating a cell with high metabolic activity."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("cell", verbose=True)

        session = generator.generate(
            user_intent="Design a highly metabolically active cell, like a hepatocyte",
            selected_components=["Nucleus", "Mitochondrion", "Ribosome", "Endoplasmic_Reticulum"],
            target_properties={
                "metabolic_rate": 2.0,  # Relative to baseline
                "volume": 5000  # cubic micrometers
            }
        )

        assert session.best_result is not None
        print(f"\nGenerated cell config:")
        print(json.dumps(session.best_result.config, indent=2))


class TestGeminiGeneratorBiomaterial:
    """Tests for biomaterial generation with Gemini."""

    @requires_gemini
    def test_generate_cartilage_like_material(self):
        """Test generating a cartilage-like biomaterial."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("biomaterial", verbose=True)

        session = generator.generate(
            user_intent="Design a cartilage-like biomaterial for joint repair",
            selected_components=["Chondrocyte", "Fibroblast"],
            target_properties={
                "elastic_modulus": 1000,  # kPa
                "porosity": 0.3,
                "stiffness_category": "medium"
            }
        )

        assert session.best_result is not None
        print(f"\nGenerated biomaterial config:")
        print(json.dumps(session.best_result.config, indent=2))


class TestQualityAssessment:
    """Tests for overall quality assessment of the generator."""

    @requires_gemini
    def test_score_improvement_over_iterations(self):
        """Test that scores generally improve over refinement iterations."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("protein", verbose=True)

        session = generator.generate(
            user_intent="Create a balanced peptide",
            selected_components=["A", "V", "L", "E", "K"],
            target_properties={"length": 10}
        )

        if len(session.iterations) >= 3:
            # Check if average of last 3 is higher than first 3
            first_avg = sum(r.score for r in session.iterations[:3]) / 3
            last_avg = sum(r.score for r in session.iterations[-3:]) / 3

            print(f"\nScore progression analysis:")
            print(f"First 3 iterations average: {first_avg:.2f}")
            print(f"Last 3 iterations average: {last_avg:.2f}")
            print(f"Improvement: {last_avg - first_avg:+.2f}")

            # Allow some variance - we just want to see the system is working
            assert last_avg >= first_avg * 0.8, "Scores should not degrade significantly"

    @requires_gemini
    def test_planning_produces_valid_config(self):
        """Test that the planning step produces a usable initial config."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("protein", verbose=True)

        session = generator.generate(
            user_intent="Create a simple test peptide",
            selected_components=["A", "G"],
            target_properties={}
        )

        # Check planning analysis
        assert "error" not in session.planning_analysis or session.iterations, \
            "Planning should succeed or iterations should still run"

        if "proposed_config" in session.planning_analysis:
            config = session.planning_analysis["proposed_config"]
            assert isinstance(config, dict), "Proposed config should be a dict"
            print(f"\nInitial proposed config from planning:")
            print(json.dumps(config, indent=2))


class TestOutputSaving:
    """Tests for session output saving."""

    @requires_gemini
    def test_save_session_creates_valid_json(self):
        """Test that saved session files are valid JSON."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator
        import tempfile
        import os

        generator = GeminiAssetGenerator("protein", verbose=False)

        session = generator.generate(
            user_intent="Quick test",
            selected_components=["A"],
            target_properties={}
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)

        try:
            generator.save_session(session, output_path)

            # Read and validate
            with open(output_path, 'r') as f:
                saved_data = json.load(f)

            assert "metadata" in saved_data
            assert "input" in saved_data
            assert "iterations" in saved_data
            assert "best_result" in saved_data

            print(f"\nSaved session structure:")
            print(f"  Metadata keys: {list(saved_data['metadata'].keys())}")
            print(f"  Iterations: {len(saved_data['iterations'])}")
            print(f"  Has best result: {saved_data['best_result'] is not None}")

        finally:
            os.unlink(output_path)


# Benchmark test - measures performance across multiple runs
class TestBenchmark:
    """Benchmark tests for measuring generator performance."""

    @requires_gemini
    @pytest.mark.benchmark
    def test_benchmark_protein_generation(self):
        """Benchmark protein generation time and quality."""
        from scripts.gemini_asset_generator import GeminiAssetGenerator
        import statistics

        generator = GeminiAssetGenerator("protein", verbose=False)

        times = []
        scores = []

        # Run 3 times
        for i in range(3):
            print(f"\nBenchmark run {i+1}/3...")
            session = generator.generate(
                user_intent="Create a hydrophobic peptide",
                selected_components=["I", "V", "L"],
                target_properties={"gravy": 1.0, "length": 8}
            )

            times.append(session.generation_time_seconds)
            if session.best_result:
                scores.append(session.best_result.score)

        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Time - Mean: {statistics.mean(times):.1f}s, "
              f"Std: {statistics.stdev(times) if len(times) > 1 else 0:.1f}s")
        print(f"Score - Mean: {statistics.mean(scores):.2f}, "
              f"Std: {statistics.stdev(scores) if len(scores) > 1 else 0:.2f}")


if __name__ == "__main__":
    # Run a quick validation
    print("Running Gemini Generator validation...")
    print(f"Gemini available: {gemini_available()}")

    if gemini_available():
        # Quick test
        from scripts.gemini_asset_generator import GeminiAssetGenerator

        generator = GeminiAssetGenerator("protein", verbose=True)
        session = generator.generate(
            user_intent="Create a short test peptide",
            selected_components=["A", "V", "L"],
            target_properties={"length": 5}
        )

        print(f"\nValidation complete!")
        print(f"Final score: {session.best_result.score if session.best_result else 'N/A'}")
    else:
        print("Gemini API not available - skipping validation")
