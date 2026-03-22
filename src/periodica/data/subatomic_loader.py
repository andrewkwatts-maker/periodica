"""
Subatomic Particle Data Loader
Loads particle data from JSON files in the SubAtomic folder.
"""

import json
import os
import math
from pathlib import Path
from typing import Dict, List, Optional


class SubatomicDataLoader:
    """Loads subatomic particle data from JSON files"""

    def __init__(self, subatomic_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            subatomic_dir: Path to directory containing particle JSON files.
                          If None, uses default 'SubAtomic' directory.
        """
        if subatomic_dir is None:
            # Default to data/active/subatomic relative to this file
            base_dir = Path(__file__).parent
            subatomic_dir = base_dir / "active" / "subatomic"

        self.subatomic_dir = Path(subatomic_dir)
        self.particles: List[Dict] = []
        self.particles_by_name: Dict[str, Dict] = {}
        self.particles_by_symbol: Dict[str, Dict] = {}
        self.baryons: List[Dict] = []
        self.mesons: List[Dict] = []

    def load_all_particles(self) -> List[Dict]:
        """
        Load all particle data from JSON files.

        Returns:
            List of particle dictionaries sorted by mass.
        """
        if not self.subatomic_dir.exists():
            print(f"Warning: SubAtomic directory not found: {self.subatomic_dir}")
            return []

        # Find all JSON files
        json_files = sorted(self.subatomic_dir.glob("*.json"))

        if not json_files:
            print(f"Warning: No particle JSON files found in {self.subatomic_dir}")
            return []

        loaded_particles = []

        for json_file in json_files:
            try:
                particle_data = self._load_particle_file(json_file)
                if particle_data:
                    loaded_particles.append(particle_data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")
                continue

        # Sort by mass
        loaded_particles.sort(key=lambda p: p.get('Mass_MeVc2', 0))

        # Store in instance variables
        self.particles = loaded_particles
        self.particles_by_name = {p['Name']: p for p in loaded_particles}
        self.particles_by_symbol = {p.get('Symbol', p['Name']): p for p in loaded_particles}

        # Categorize particles
        self._categorize_particles()

        return loaded_particles

    def _load_particle_file(self, filepath: Path) -> Optional[Dict]:
        """
        Load a single particle JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dictionary containing particle data or None if invalid
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Handle JSON with comments (remove them)
                content = f.read()
                # Simple comment removal (line comments)
                lines = content.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Remove // comments
                    comment_idx = line.find('//')
                    if comment_idx != -1:
                        line = line[:comment_idx]
                    cleaned_lines.append(line)
                content = '\n'.join(cleaned_lines)

                data = json.loads(content)

            # Validate required fields
            required_fields = ['Name', 'Type']
            for field in required_fields:
                if field not in data:
                    print(f"Warning: Missing required field '{field}' in {filepath.name}")
                    return None

            # Add computed fields
            data = self._add_computed_fields(data)

            return data
        except json.JSONDecodeError as e:
            print(f"Warning: JSON parse error in {filepath.name}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Failed to load {filepath.name}: {e}")
            return None

    def _add_computed_fields(self, data: Dict) -> Dict:
        """Add computed fields for visualization"""

        # Determine category
        classification = data.get('Classification', [])
        data['_category'] = self._determine_category(classification)

        # Determine if it's a baryon or meson
        data['_is_baryon'] = 'Baryon' in classification
        data['_is_meson'] = 'Meson' in classification

        # Calculate log mass for visualization (mass spans many orders of magnitude)
        mass = data.get('Mass_MeVc2', 0)
        if mass > 0:
            data['_log_mass'] = math.log10(mass)
        else:
            data['_log_mass'] = 0

        # Calculate log half-life for visualization
        half_life = data.get('HalfLife_s')
        if half_life and half_life > 0:
            data['_log_half_life'] = math.log10(half_life)
        else:
            data['_log_half_life'] = None

        # Count quarks
        composition = data.get('Composition', [])
        quark_count = sum(c.get('Count', 1) for c in composition)
        data['_quark_count'] = quark_count

        # Parse quark content string
        quark_content = data.get('QuarkContent', '')
        data['_parsed_quarks'] = self._parse_quark_content(quark_content, composition)

        # Stability factor (for visualization: 0 = extremely unstable, 1 = stable)
        stability = data.get('Stability', 'Unstable')
        if stability == 'Stable':
            data['_stability_factor'] = 1.0
        elif half_life:
            # Map log half-life to 0-1 range
            # Most unstable: ~10^-24 s, Most stable unstable: ~10^3 s (neutron)
            log_hl = data['_log_half_life']
            if log_hl is not None:
                # Normalize: -24 -> 0.0, 3 -> 0.9
                data['_stability_factor'] = max(0, min(0.9, (log_hl + 24) / 30))
            else:
                data['_stability_factor'] = 0.5
        else:
            data['_stability_factor'] = 0.5

        return data

    def _determine_category(self, classification: List[str]) -> str:
        """Determine particle category from classification list"""
        classification_lower = [c.lower() for c in classification]

        if 'baryon' in classification_lower:
            # Further categorize baryons
            if any('delta' in c.lower() for c in classification):
                return 'delta'
            elif any('sigma' in c.lower() for c in classification):
                return 'sigma'
            elif any('xi' in c.lower() or 'cascade' in c.lower() for c in classification):
                return 'xi'
            elif any('lambda' in c.lower() for c in classification):
                return 'lambda'
            elif any('omega' in c.lower() for c in classification):
                return 'omega'
            return 'baryon'
        elif 'meson' in classification_lower:
            if any('pion' in c.lower() for c in classification):
                return 'pion'
            elif any('kaon' in c.lower() for c in classification):
                return 'kaon'
            elif any('eta' in c.lower() for c in classification):
                return 'eta'
            elif any('charmonium' in c.lower() or 'jpsi' in c.lower() for c in classification):
                return 'jpsi'
            elif any('bottomonium' in c.lower() or 'upsilon' in c.lower() for c in classification):
                return 'upsilon'
            return 'meson'
        elif 'lepton' in classification_lower:
            return 'lepton'
        elif 'boson' in classification_lower:
            return 'boson'

        return 'other'

    def _parse_quark_content(self, quark_content: str, composition: List[Dict]) -> List[Dict]:
        """Parse quark content into structured format"""
        quarks = []

        for comp in composition:
            constituent = comp.get('Constituent', '')
            count = comp.get('Count', 1)
            symbol = comp.get('Symbol', '')
            is_anti = comp.get('IsAnti', False) or 'anti' in constituent.lower()

            # Determine quark type
            quark_type = symbol.replace('-bar', '').lower()
            if not quark_type:
                if 'up' in constituent.lower():
                    quark_type = 'u'
                elif 'down' in constituent.lower():
                    quark_type = 'd'
                elif 'strange' in constituent.lower():
                    quark_type = 's'
                elif 'charm' in constituent.lower():
                    quark_type = 'c'
                elif 'bottom' in constituent.lower():
                    quark_type = 'b'
                elif 'top' in constituent.lower():
                    quark_type = 't'

            for _ in range(count):
                quarks.append({
                    'type': quark_type,
                    'is_anti': is_anti,
                    'charge': comp.get('Charge_e', 0)
                })

        return quarks

    def _categorize_particles(self):
        """Separate particles into baryons and mesons"""
        self.baryons = [p for p in self.particles if p.get('_is_baryon', False)]
        self.mesons = [p for p in self.particles if p.get('_is_meson', False)]

    def get_particle_by_name(self, name: str) -> Optional[Dict]:
        """Get particle data by name"""
        return self.particles_by_name.get(name)

    def get_particle_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get particle data by symbol"""
        return self.particles_by_symbol.get(symbol)

    def get_all_particles(self) -> List[Dict]:
        """Get all loaded particles"""
        return self.particles

    def get_baryons(self) -> List[Dict]:
        """Get all baryon particles"""
        return self.baryons

    def get_mesons(self) -> List[Dict]:
        """Get all meson particles"""
        return self.mesons

    def get_particles_by_charge(self, charge: int) -> List[Dict]:
        """Get particles with specific charge"""
        return [p for p in self.particles if p.get('Charge_e', 0) == charge]

    def get_particles_by_category(self, category: str) -> List[Dict]:
        """Get particles by internal category"""
        return [p for p in self.particles if p.get('_category', '') == category]

    def get_particle_count(self) -> int:
        """Get total number of loaded particles"""
        return len(self.particles)

    def get_mass_range(self) -> tuple:
        """Get min and max mass of all particles"""
        masses = [p.get('Mass_MeVc2', 0) for p in self.particles]
        if masses:
            return min(masses), max(masses)
        return 0, 0

    def get_decay_chain(self, particle_name: str, max_depth: int = 5) -> List[List[str]]:
        """
        Get decay chain for a particle.

        Args:
            particle_name: Name of the particle
            max_depth: Maximum decay depth to trace

        Returns:
            List of decay chains (list of particle names)
        """
        particle = self.get_particle_by_name(particle_name)
        if not particle:
            return []

        chains = []
        decay_products = particle.get('DecayProducts', [])

        if not decay_products:
            return [[particle_name]]

        for product in decay_products:
            if max_depth > 0:
                sub_chains = self.get_decay_chain(product, max_depth - 1)
                for chain in sub_chains:
                    chains.append([particle_name] + chain)
            else:
                chains.append([particle_name, product])

        return chains if chains else [[particle_name]]


# Singleton instance for easy access
_loader_instance = None


def get_subatomic_loader() -> SubatomicDataLoader:
    """Get or create the singleton loader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = SubatomicDataLoader()
        _loader_instance.load_all_particles()
    return _loader_instance


def load_subatomic_data() -> List[Dict]:
    """Convenience function to load all particle data"""
    loader = get_subatomic_loader()
    return loader.get_all_particles()
