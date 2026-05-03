"""
Tests for the derivation metadata system.
Verifies stamping, serialization, source tracking, and metadata queries.
"""

import pytest
from periodica.utils.derivation_metadata import (
    DerivationSource,
    DerivationMetadata,
    DerivationTracker,
)


class TestDerivationSource:
    """Test DerivationSource enum."""

    def test_all_sources_defined(self):
        sources = list(DerivationSource)
        assert len(sources) == 6

    def test_quark_derived_value(self):
        assert DerivationSource.QUARK_DERIVED.value == "quark_derived"

    def test_physics_derived_value(self):
        assert DerivationSource.PHYSICS_DERIVED.value == "physics_derived"

    def test_manual_value(self):
        assert DerivationSource.MANUAL.value == "manual"

    def test_loaded_default_value(self):
        assert DerivationSource.LOADED_DEFAULT.value == "loaded_default"

    def test_ai_generated_value(self):
        assert DerivationSource.AI_GENERATED.value == "ai_generated"

    def test_auto_generated_value(self):
        assert DerivationSource.AUTO_GENERATED.value == "auto_generated"


class TestDerivationMetadata:
    """Test DerivationMetadata dataclass."""

    def test_create_with_defaults(self):
        meta = DerivationMetadata(source=DerivationSource.MANUAL.value)
        assert meta.source == "manual"
        assert meta.derived_from == []
        assert meta.derivation_chain == []
        assert meta.confidence == 1.0
        assert meta.timestamp != ""
        assert meta.model_version == "1.0.0"

    def test_create_with_all_fields(self):
        meta = DerivationMetadata(
            source=DerivationSource.QUARK_DERIVED.value,
            derived_from=["up_quark", "down_quark"],
            derivation_chain=["quarks", "hadrons", "nucleus"],
            confidence=0.95,
            timestamp="2026-01-01T00:00:00+00:00",
            model_version="2.0.0",
        )
        assert meta.source == "quark_derived"
        assert len(meta.derived_from) == 2
        assert meta.confidence == 0.95

    def test_to_dict(self):
        meta = DerivationMetadata(
            source=DerivationSource.PHYSICS_DERIVED.value,
            confidence=0.8,
        )
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "physics_derived"
        assert d["confidence"] == 0.8
        assert "timestamp" in d
        assert "derived_from" in d

    def test_from_dict(self):
        data = {
            "source": "quark_derived",
            "derived_from": ["u", "d"],
            "derivation_chain": ["quarks", "hadrons"],
            "confidence": 0.9,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "model_version": "1.5.0",
        }
        meta = DerivationMetadata.from_dict(data)
        assert meta.source == "quark_derived"
        assert meta.derived_from == ["u", "d"]
        assert meta.confidence == 0.9
        assert meta.model_version == "1.5.0"

    def test_from_dict_with_missing_fields(self):
        data = {"source": "manual"}
        meta = DerivationMetadata.from_dict(data)
        assert meta.source == "manual"
        assert meta.derived_from == []
        assert meta.confidence == 1.0

    def test_roundtrip(self):
        original = DerivationMetadata(
            source=DerivationSource.AUTO_GENERATED.value,
            derived_from=["elem_H", "elem_O"],
            confidence=0.85,
        )
        d = original.to_dict()
        restored = DerivationMetadata.from_dict(d)
        assert restored.source == original.source
        assert restored.derived_from == original.derived_from
        assert restored.confidence == original.confidence


class TestDerivationTracker:
    """Test DerivationTracker class."""

    def test_stamp_adds_metadata(self):
        data = {"Name": "Hydrogen", "atomic_number": 1}
        DerivationTracker.stamp(data, DerivationSource.QUARK_DERIVED)
        assert "_derivation" in data
        assert data["_derivation"]["source"] == "quark_derived"

    def test_stamp_with_all_params(self):
        data = {"Name": "Water"}
        DerivationTracker.stamp(
            data,
            source=DerivationSource.PHYSICS_DERIVED,
            derived_from=["H", "O"],
            derivation_chain=["elements", "molecules"],
            confidence=0.9,
            model_version="2.0.0",
        )
        meta = data["_derivation"]
        assert meta["source"] == "physics_derived"
        assert meta["derived_from"] == ["H", "O"]
        assert meta["confidence"] == 0.9
        assert meta["model_version"] == "2.0.0"

    def test_stamp_returns_same_dict(self):
        data = {"Name": "Iron"}
        result = DerivationTracker.stamp(data, DerivationSource.MANUAL)
        assert result is data

    def test_get_metadata(self):
        data = {"Name": "Test"}
        DerivationTracker.stamp(data, DerivationSource.QUARK_DERIVED, confidence=0.95)
        meta = DerivationTracker.get_metadata(data)
        assert meta is not None
        assert meta.source == "quark_derived"
        assert meta.confidence == 0.95

    def test_get_metadata_returns_none_when_absent(self):
        data = {"Name": "Test"}
        assert DerivationTracker.get_metadata(data) is None

    def test_get_source(self):
        data = {"Name": "Test"}
        DerivationTracker.stamp(data, DerivationSource.AI_GENERATED)
        assert DerivationTracker.get_source(data) == DerivationSource.AI_GENERATED

    def test_get_source_returns_none_when_absent(self):
        assert DerivationTracker.get_source({}) is None

    def test_is_derived_quark(self):
        data = {}
        DerivationTracker.stamp(data, DerivationSource.QUARK_DERIVED)
        assert DerivationTracker.is_derived(data) is True

    def test_is_derived_physics(self):
        data = {}
        DerivationTracker.stamp(data, DerivationSource.PHYSICS_DERIVED)
        assert DerivationTracker.is_derived(data) is True

    def test_is_derived_auto(self):
        data = {}
        DerivationTracker.stamp(data, DerivationSource.AUTO_GENERATED)
        assert DerivationTracker.is_derived(data) is True

    def test_is_derived_false_for_manual(self):
        data = {}
        DerivationTracker.stamp(data, DerivationSource.MANUAL)
        assert DerivationTracker.is_derived(data) is False

    def test_is_derived_false_for_no_metadata(self):
        assert DerivationTracker.is_derived({}) is False

    def test_is_manual(self):
        data = {}
        DerivationTracker.stamp(data, DerivationSource.MANUAL)
        assert DerivationTracker.is_manual(data) is True

    def test_is_manual_false_for_derived(self):
        data = {}
        DerivationTracker.stamp(data, DerivationSource.QUARK_DERIVED)
        assert DerivationTracker.is_manual(data) is False

    def test_strip_metadata(self):
        data = {"Name": "Test", "value": 42}
        DerivationTracker.stamp(data, DerivationSource.QUARK_DERIVED)
        stripped = DerivationTracker.strip_metadata(data)
        assert "_derivation" not in stripped
        assert stripped["Name"] == "Test"
        assert stripped["value"] == 42
        # Original not modified
        assert "_derivation" in data

    def test_get_confidence(self):
        data = {}
        DerivationTracker.stamp(data, DerivationSource.PHYSICS_DERIVED, confidence=0.75)
        assert DerivationTracker.get_confidence(data) == 0.75

    def test_get_confidence_returns_zero_when_absent(self):
        assert DerivationTracker.get_confidence({}) == 0.0

    def test_overwrite_existing_metadata(self):
        data = {"Name": "Test"}
        DerivationTracker.stamp(data, DerivationSource.LOADED_DEFAULT, confidence=0.5)
        DerivationTracker.stamp(data, DerivationSource.QUARK_DERIVED, confidence=0.95)
        assert DerivationTracker.get_source(data) == DerivationSource.QUARK_DERIVED
        assert DerivationTracker.get_confidence(data) == 0.95
