"""
Tests for the ValidationReporter.
Verifies report generation, error calculation, and comparison logic.
"""

import pytest
from periodica.utils.validation_report import ValidationReporter, ValidationResult


@pytest.fixture
def reporter():
    return ValidationReporter()


class TestValidationResult:
    def test_zero_error(self):
        r = ValidationResult("test", "mass", "amu", 10.0, 10.0)
        assert r.percent_error == 0.0
        assert r.absolute_error == 0.0

    def test_positive_error(self):
        r = ValidationResult("test", "mass", "amu", 100.0, 110.0)
        assert r.percent_error == 10.0
        assert r.absolute_error == 10.0

    def test_negative_error(self):
        r = ValidationResult("test", "mass", "amu", 100.0, 90.0)
        assert r.percent_error == 10.0

    def test_zero_reference(self):
        r = ValidationResult("test", "mass", "amu", 0.0, 5.0)
        assert r.percent_error == float('inf')

    def test_to_dict(self):
        r = ValidationResult("H", "mass", "amu", 1.008, 1.01)
        d = r.to_dict()
        assert d['item'] == 'H'
        assert d['property'] == 'mass'
        assert 'percent_error' in d


class TestErrorStatistics:
    def test_empty_results(self, reporter):
        stats = reporter.get_error_statistics([])
        assert stats['count'] == 0
        assert stats['mean_error'] == 0

    def test_perfect_results(self, reporter):
        results = [
            ValidationResult("A", "x", "", 10, 10),
            ValidationResult("B", "x", "", 20, 20),
        ]
        stats = reporter.get_error_statistics(results)
        assert stats['mean_error'] == 0
        assert stats['within_1pct'] == 100

    def test_mixed_results(self, reporter):
        results = [
            ValidationResult("A", "x", "", 100, 100),    # 0% error
            ValidationResult("B", "x", "", 100, 95),     # 5% error
            ValidationResult("C", "x", "", 100, 80),     # 20% error
        ]
        stats = reporter.get_error_statistics(results)
        assert stats['count'] == 3
        assert stats['within_5pct'] > 30  # at least 1/3

    def test_all_within_10pct(self, reporter):
        results = [
            ValidationResult("A", "x", "", 100, 95),
            ValidationResult("B", "x", "", 100, 92),
        ]
        stats = reporter.get_error_statistics(results)
        assert stats['within_10pct'] == 100


class TestCompareCategory:
    def test_elements_returns_list(self, reporter):
        results = reporter.compare_category('elements')
        assert isinstance(results, list)

    def test_unknown_category_empty(self, reporter):
        results = reporter.compare_category('nonexistent')
        assert results == []

    def test_element_results_have_properties(self, reporter):
        results = reporter.compare_category('elements')
        if results:
            r = results[0]
            assert hasattr(r, 'item_name')
            assert hasattr(r, 'property_name')
            assert hasattr(r, 'percent_error')


class TestCompareAll:
    def test_returns_dict(self, reporter):
        results = reporter.compare_all()
        assert isinstance(results, dict)


class TestGenerateReport:
    def test_text_report_not_empty(self, reporter):
        report = reporter.generate_report()
        assert len(report) > 0
        assert 'VALIDATION REPORT' in report

    def test_html_report_not_empty(self, reporter):
        report = reporter.generate_html_report()
        assert len(report) > 0
        assert '<html>' in report

    def test_html_report_has_structure(self, reporter):
        report = reporter.generate_html_report()
        assert '<h1>' in report
        assert '</html>' in report


class TestNestedPropertyAccess:
    def test_simple_property(self, reporter):
        data = {'mass': 10.0}
        assert reporter._get_nested(data, 'mass') == 10.0

    def test_nested_property(self, reporter):
        data = {'PhysicalProperties': {'Density_g_cm3': 7.93}}
        assert reporter._get_nested(data, 'PhysicalProperties.Density_g_cm3') == 7.93

    def test_missing_property(self, reporter):
        data = {'mass': 10.0}
        assert reporter._get_nested(data, 'nonexistent') is None

    def test_missing_nested(self, reporter):
        data = {'PhysicalProperties': {'Density_g_cm3': 7.93}}
        assert reporter._get_nested(data, 'PhysicalProperties.MeltingPoint_K') is None
