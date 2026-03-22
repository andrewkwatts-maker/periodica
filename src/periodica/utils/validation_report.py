"""
Validation Report Generator
=============================
Compares derived property values against reference (experimental) data
from data/defaults/ to produce margin-of-error estimates.
"""

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from periodica.utils.logger import get_logger

logger = get_logger('validation_report')

# Properties to compare for each category
_COMPARISON_PROPERTIES = {
    'elements': [
        ('atomic_mass', 'Atomic Mass', 'amu'),
        ('melting_point', 'Melting Point', 'K'),
        ('boiling_point', 'Boiling Point', 'K'),
        ('density', 'Density', 'g/cm³'),
        ('ionization_energy', 'Ionization Energy', 'eV'),
        ('electronegativity', 'Electronegativity', ''),
        ('atomic_radius', 'Atomic Radius', 'pm'),
    ],
    'molecules': [
        ('MolecularMass_amu', 'Molecular Mass', 'amu'),
    ],
    'alloys': [
        ('PhysicalProperties.Density_g_cm3', 'Density', 'g/cm³'),
        ('PhysicalProperties.MeltingPoint_K', 'Melting Point', 'K'),
        ('MechanicalProperties.TensileStrength_MPa', 'Tensile Strength', 'MPa'),
        ('MechanicalProperties.YieldStrength_MPa', 'Yield Strength', 'MPa'),
    ],
}


class ValidationResult:
    """Result of comparing a single property for one item."""

    def __init__(
        self,
        item_name: str,
        property_name: str,
        unit: str,
        reference_value: float,
        derived_value: float,
    ):
        self.item_name = item_name
        self.property_name = property_name
        self.unit = unit
        self.reference_value = reference_value
        self.derived_value = derived_value

        if reference_value != 0:
            self.percent_error = abs(derived_value - reference_value) / abs(reference_value) * 100
        else:
            self.percent_error = 0.0 if derived_value == 0 else float('inf')

        self.absolute_error = abs(derived_value - reference_value)

    def to_dict(self) -> Dict:
        return {
            'item': self.item_name,
            'property': self.property_name,
            'unit': self.unit,
            'reference': self.reference_value,
            'derived': self.derived_value,
            'percent_error': round(self.percent_error, 2),
            'absolute_error': round(self.absolute_error, 4),
        }


class ValidationReporter:
    """Compares derived data against reference (experimental) data."""

    def __init__(
        self,
        active_dir: Optional[str] = None,
        defaults_dir: Optional[str] = None,
    ):
        base = Path(__file__).parent.parent / 'data'
        self._active_dir = Path(active_dir) if active_dir else base / 'active'
        self._defaults_dir = Path(defaults_dir) if defaults_dir else base / 'defaults'

    def compare_category(self, category: str) -> List[ValidationResult]:
        """
        Compare all items in a category against reference data.

        Args:
            category: 'elements', 'molecules', 'alloys', etc.

        Returns:
            List of ValidationResult objects
        """
        props = _COMPARISON_PROPERTIES.get(category, [])
        if not props:
            return []

        # Load reference data
        ref_dir = self._defaults_dir / category
        active_dir = self._active_dir / category

        if not ref_dir.exists() or not active_dir.exists():
            return []

        ref_data = self._load_dir(ref_dir)
        active_data = self._load_dir(active_dir)

        results = []

        for name, ref_item in ref_data.items():
            derived_item = active_data.get(name)
            if derived_item is None:
                continue

            for prop_path, prop_label, unit in props:
                ref_val = self._get_nested(ref_item, prop_path)
                der_val = self._get_nested(derived_item, prop_path)

                if ref_val is not None and der_val is not None:
                    try:
                        ref_float = float(ref_val)
                        der_float = float(der_val)
                        results.append(ValidationResult(
                            item_name=name,
                            property_name=prop_label,
                            unit=unit,
                            reference_value=ref_float,
                            derived_value=der_float,
                        ))
                    except (TypeError, ValueError):
                        pass

        return results

    def compare_all(self) -> Dict[str, List[ValidationResult]]:
        """Compare all categories that have comparison properties."""
        results = {}
        for category in _COMPARISON_PROPERTIES:
            cat_results = self.compare_category(category)
            if cat_results:
                results[category] = cat_results
        return results

    def get_error_statistics(
        self,
        results: List[ValidationResult],
    ) -> Dict:
        """
        Compute error statistics for a list of validation results.

        Returns:
            Dict with mean_error, median_error, max_error, within_5pct, etc.
        """
        if not results:
            return {
                'count': 0, 'mean_error': 0, 'median_error': 0,
                'max_error': 0, 'within_1pct': 0, 'within_5pct': 0,
                'within_10pct': 0, 'within_20pct': 0,
            }

        errors = [r.percent_error for r in results if not math.isinf(r.percent_error)]
        if not errors:
            return {
                'count': len(results), 'mean_error': float('inf'),
                'median_error': float('inf'), 'max_error': float('inf'),
                'within_1pct': 0, 'within_5pct': 0,
                'within_10pct': 0, 'within_20pct': 0,
            }

        errors.sort()
        n = len(errors)
        mean_err = sum(errors) / n
        median_err = errors[n // 2]
        max_err = max(errors)

        within_1 = sum(1 for e in errors if e <= 1) / n * 100
        within_5 = sum(1 for e in errors if e <= 5) / n * 100
        within_10 = sum(1 for e in errors if e <= 10) / n * 100
        within_20 = sum(1 for e in errors if e <= 20) / n * 100

        return {
            'count': n,
            'mean_error': round(mean_err, 2),
            'median_error': round(median_err, 2),
            'max_error': round(max_err, 2),
            'within_1pct': round(within_1, 1),
            'within_5pct': round(within_5, 1),
            'within_10pct': round(within_10, 1),
            'within_20pct': round(within_20, 1),
        }

    def generate_report(self) -> str:
        """
        Generate a complete validation report as formatted text.

        Returns:
            Multi-line report string
        """
        all_results = self.compare_all()
        lines = [
            "=" * 70,
            "DERIVATION VALIDATION REPORT",
            "Comparing derived values against reference (experimental) data",
            "=" * 70,
            "",
        ]

        for category, results in all_results.items():
            stats = self.get_error_statistics(results)
            lines.append(f"--- {category.upper()} ---")
            lines.append(f"  Properties compared: {stats['count']}")
            lines.append(f"  Mean error:   {stats['mean_error']:.2f}%")
            lines.append(f"  Median error: {stats['median_error']:.2f}%")
            lines.append(f"  Max error:    {stats['max_error']:.2f}%")
            lines.append(f"  Within  1%:   {stats['within_1pct']:.1f}%")
            lines.append(f"  Within  5%:   {stats['within_5pct']:.1f}%")
            lines.append(f"  Within 10%:   {stats['within_10pct']:.1f}%")
            lines.append(f"  Within 20%:   {stats['within_20pct']:.1f}%")
            lines.append("")

            # Show worst offenders
            worst = sorted(results, key=lambda r: r.percent_error, reverse=True)[:5]
            if worst:
                lines.append("  Largest deviations:")
                for w in worst:
                    lines.append(
                        f"    {w.item_name} / {w.property_name}: "
                        f"{w.reference_value} -> {w.derived_value} "
                        f"({w.percent_error:.1f}% error)"
                    )
                lines.append("")

        if not all_results:
            lines.append("No comparison data available.")
            lines.append("Run cascade regeneration first, then compare against defaults.")

        lines.append("=" * 70)
        return '\n'.join(lines)

    def generate_html_report(self) -> str:
        """Generate an HTML-formatted validation report."""
        all_results = self.compare_all()

        html = ['<html><head><style>']
        html.append('body { font-family: sans-serif; margin: 20px; }')
        html.append('table { border-collapse: collapse; width: 100%; margin: 10px 0; }')
        html.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
        html.append('th { background: #4a5568; color: white; }')
        html.append('.good { background: #c6f6d5; }')
        html.append('.ok { background: #fefcbf; }')
        html.append('.bad { background: #fed7d7; }')
        html.append('</style></head><body>')
        html.append('<h1>Derivation Validation Report</h1>')

        for category, results in all_results.items():
            stats = self.get_error_statistics(results)
            html.append(f'<h2>{category.title()}</h2>')
            html.append(f'<p>Mean error: {stats["mean_error"]:.2f}% | ')
            html.append(f'Median: {stats["median_error"]:.2f}% | ')
            html.append(f'Within 5%: {stats["within_5pct"]:.1f}%</p>')

            html.append('<table><tr>')
            html.append('<th>Item</th><th>Property</th><th>Reference</th>')
            html.append('<th>Derived</th><th>Error %</th></tr>')

            for r in sorted(results, key=lambda x: x.percent_error, reverse=True):
                css = 'good' if r.percent_error < 5 else ('ok' if r.percent_error < 20 else 'bad')
                html.append(f'<tr class="{css}">')
                html.append(f'<td>{r.item_name}</td>')
                html.append(f'<td>{r.property_name}</td>')
                html.append(f'<td>{r.reference_value}</td>')
                html.append(f'<td>{r.derived_value}</td>')
                html.append(f'<td>{r.percent_error:.2f}%</td>')
                html.append('</tr>')

            html.append('</table>')

        if not all_results:
            html.append('<p>No comparison data available.</p>')

        html.append('</body></html>')
        return '\n'.join(html)

    def _load_dir(self, dir_path: Path) -> Dict[str, Dict]:
        """Load all JSON files from a directory, keyed by filename stem."""
        data = {}
        for f in dir_path.glob('*.json'):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data[f.stem] = json.load(fp)
            except Exception:
                pass
        return data

    def _get_nested(self, data: Dict, path: str):
        """Get a nested value using dot notation."""
        parts = path.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
