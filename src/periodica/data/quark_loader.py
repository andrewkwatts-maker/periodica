#!/usr/bin/env python3
"""
Quark/Particle Data Loader
Loads particle data from JSON files in the Quarks, AntiQuarks, and SubAtomic directories.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from periodica.core.quark_enums import ParticleType, QuarkGeneration


class QuarkDataLoader:
    """Loads particle data from JSON files"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the loader.

        Args:
            base_dir: Path to base directory containing Quarks/AntiQuarks folders.
                     If None, uses parent of this file's directory.
        """
        if base_dir is None:
            # Default to data directory (where this file is located)
            base_dir = Path(__file__).parent

        self.base_dir = Path(base_dir)
        self.particles: List[Dict] = []
        self.particles_by_name: Dict[str, Dict] = {}
        self.particles_by_symbol: Dict[str, Dict] = {}

        # Source directories - use active data directories
        self.quarks_dir = self.base_dir / "active" / "quarks"
        self.antiquarks_dir = self.base_dir / "active" / "antiquarks"
        self.subatomic_dir = self.base_dir / "active" / "subatomic"

    def load_all_particles(self, include_antiparticles: bool = True,
                          include_composite: bool = True) -> List[Dict]:
        """
        Load all particle data from JSON files.

        Args:
            include_antiparticles: Whether to include antiparticles
            include_composite: Whether to include composite particles

        Returns:
            List of particle dictionaries
        """
        loaded_particles = []

        # Load from Quarks directory (Standard Model particles)
        if self.quarks_dir.exists():
            for json_file in self.quarks_dir.glob("*.json"):
                particle = self._load_particle_file(json_file, is_antiparticle=False)
                if particle:
                    loaded_particles.append(particle)

        # Load from AntiQuarks directory
        if include_antiparticles and self.antiquarks_dir.exists():
            for json_file in self.antiquarks_dir.glob("*.json"):
                particle = self._load_particle_file(json_file, is_antiparticle=True)
                if particle:
                    loaded_particles.append(particle)

        # Load from SubAtomic directory (composite particles)
        if include_composite and self.subatomic_dir.exists():
            for json_file in self.subatomic_dir.glob("*.json"):
                particle = self._load_particle_file(json_file, is_antiparticle=False)
                if particle:
                    particle['is_composite'] = True
                    loaded_particles.append(particle)

        # Process particles to add computed fields
        for particle in loaded_particles:
            self._process_particle(particle)

        # Sort by mass (ascending)
        loaded_particles.sort(key=lambda p: p.get('Mass_MeVc2', 0) or 0)

        # Store references
        self.particles = loaded_particles
        self.particles_by_name = {p['Name']: p for p in loaded_particles}
        self.particles_by_symbol = {p['Symbol']: p for p in loaded_particles if p.get('Symbol')}

        return loaded_particles

    def _load_particle_file(self, filepath: Path, is_antiparticle: bool = False) -> Optional[Dict]:
        """
        Load a single particle JSON file.

        Args:
            filepath: Path to the JSON file
            is_antiparticle: Whether this is from the antiparticles folder

        Returns:
            Dictionary containing particle data or None if loading fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove JavaScript-style comments (// ...)
                content = re.sub(r'//.*?(?=\n|$)', '', content)
                data = json.loads(content)

            # Add metadata
            data['_source_file'] = filepath.name
            data['_is_antiparticle'] = is_antiparticle

            return data

        except json.JSONDecodeError as e:
            print(f"Warning: JSON parse error in {filepath.name}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Failed to load {filepath.name}: {e}")
            return None

    def _process_particle(self, particle: Dict):
        """
        Process particle data to add computed fields.

        Args:
            particle: Particle dictionary to process
        """
        # Determine particle type
        classification = particle.get('Classification', [])
        particle['particle_type'] = ParticleType.from_classification(classification)
        particle['particle_type_name'] = particle['particle_type'].name

        # Determine generation
        name = particle.get('Name', '')
        particle['generation'] = QuarkGeneration.from_particle_name(name)
        particle['generation_num'] = particle['generation'].value

        # Normalize charge for display
        charge = particle.get('Charge_e', 0)
        if charge is not None:
            # Handle fractional charges
            if abs(charge - 2/3) < 0.001:
                particle['charge_display'] = "+2/3"
            elif abs(charge - (-1/3)) < 0.001:
                particle['charge_display'] = "-1/3"
            elif abs(charge - 1/3) < 0.001:
                particle['charge_display'] = "+1/3"
            elif abs(charge - (-2/3)) < 0.001:
                particle['charge_display'] = "-2/3"
            elif charge == int(charge):
                particle['charge_display'] = f"{int(charge):+d}" if charge != 0 else "0"
            else:
                particle['charge_display'] = f"{charge:+.2f}"
        else:
            particle['charge_display'] = "N/A"

        # Format mass for display
        mass = particle.get('Mass_MeVc2', 0)
        if mass is not None and mass > 0:
            if mass >= 1000:
                particle['mass_display'] = f"{mass/1000:.2f} GeV/c^2"
            elif mass >= 1:
                particle['mass_display'] = f"{mass:.2f} MeV/c^2"
            elif mass >= 0.001:
                particle['mass_display'] = f"{mass*1000:.2f} keV/c^2"
            else:
                particle['mass_display'] = f"{mass:.2e} MeV/c^2"
        elif mass == 0:
            particle['mass_display'] = "0 (massless)"
        else:
            particle['mass_display'] = "N/A"

        # Format spin for display
        spin = particle.get('Spin_hbar')
        if spin is not None:
            if spin == int(spin):
                particle['spin_display'] = str(int(spin))
            elif abs(spin - 0.5) < 0.001:
                particle['spin_display'] = "1/2"
            elif abs(spin - 1.5) < 0.001:
                particle['spin_display'] = "3/2"
            else:
                particle['spin_display'] = str(spin)
        else:
            particle['spin_display'] = "N/A"

        # Determine fermion/boson
        spin_val = particle.get('Spin_hbar', 0) or 0
        if spin_val == int(spin_val):
            particle['statistics'] = "Boson"
        else:
            particle['statistics'] = "Fermion"

        # Standard Model position (for layout)
        self._assign_standard_model_position(particle)

    def _assign_standard_model_position(self, particle: Dict):
        """
        Assign Standard Model grid position.
        Layout:
        - Row 0: Up quarks (u, c, t)
        - Row 1: Down quarks (d, s, b)
        - Row 2: Charged leptons (e, mu, tau)
        - Row 3: Neutrinos (ve, vmu, vtau)
        - Column 4: Gauge bosons (g, gamma, Z, W)
        - Column 5: Higgs

        Args:
            particle: Particle dictionary
        """
        name_lower = particle.get('Name', '').lower()
        classification = particle.get('Classification', [])
        classification_lower = [c.lower() for c in classification]

        # Default position
        particle['sm_row'] = -1
        particle['sm_col'] = -1

        # Quarks
        if 'quark' in classification_lower:
            # Up-type quarks (row 0)
            if 'up' in name_lower:
                particle['sm_row'] = 0
                particle['sm_col'] = 0
            elif 'charm' in name_lower:
                particle['sm_row'] = 0
                particle['sm_col'] = 1
            elif 'top' in name_lower:
                particle['sm_row'] = 0
                particle['sm_col'] = 2
            # Down-type quarks (row 1)
            elif 'down' in name_lower:
                particle['sm_row'] = 1
                particle['sm_col'] = 0
            elif 'strange' in name_lower:
                particle['sm_row'] = 1
                particle['sm_col'] = 1
            elif 'bottom' in name_lower:
                particle['sm_row'] = 1
                particle['sm_col'] = 2

        # Leptons
        elif 'lepton' in classification_lower:
            # Charged leptons (row 2)
            if 'electron' in name_lower and 'neutrino' not in name_lower:
                particle['sm_row'] = 2
                particle['sm_col'] = 0
            elif 'muon' in name_lower and 'neutrino' not in name_lower:
                particle['sm_row'] = 2
                particle['sm_col'] = 1
            elif 'tau' in name_lower and 'neutrino' not in name_lower:
                particle['sm_row'] = 2
                particle['sm_col'] = 2
            # Neutrinos (row 3)
            elif 'electron' in name_lower and 'neutrino' in name_lower:
                particle['sm_row'] = 3
                particle['sm_col'] = 0
            elif 'muon' in name_lower or 'mu' in name_lower:
                particle['sm_row'] = 3
                particle['sm_col'] = 1
            elif 'tau' in name_lower:
                particle['sm_row'] = 3
                particle['sm_col'] = 2

        # Bosons
        elif 'boson' in classification_lower or 'force carrier' in classification_lower:
            if 'gluon' in name_lower:
                particle['sm_row'] = 0
                particle['sm_col'] = 3
            elif 'photon' in name_lower:
                particle['sm_row'] = 1
                particle['sm_col'] = 3
            elif 'z boson' in name_lower or name_lower == 'z':
                particle['sm_row'] = 2
                particle['sm_col'] = 3
            elif 'w' in name_lower and 'boson' in name_lower:
                particle['sm_row'] = 3
                particle['sm_col'] = 3
            elif 'higgs' in name_lower:
                particle['sm_row'] = 0
                particle['sm_col'] = 4

    def get_particle_by_name(self, name: str) -> Optional[Dict]:
        """Get particle by name.

        Accepts exact names (``"Up Quark"``) or short names (``"Up"``).
        Short names are matched by case-insensitive prefix.
        """
        result = self.particles_by_name.get(name)
        if result is not None:
            return result
        # Fuzzy: case-insensitive prefix match (e.g. "Up" → "Up Quark")
        name_lower = name.lower()
        for pname, pdata in self.particles_by_name.items():
            if pname.lower().startswith(name_lower):
                return pdata
        return None

    def get_particle_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get particle by symbol"""
        return self.particles_by_symbol.get(symbol)

    def get_particles_by_type(self, particle_type: ParticleType) -> List[Dict]:
        """Get all particles of a specific type"""
        return [p for p in self.particles if p.get('particle_type') == particle_type]

    def get_particles_by_generation(self, generation: int) -> List[Dict]:
        """Get all particles of a specific generation"""
        return [p for p in self.particles if p.get('generation_num') == generation]

    def get_standard_model_particles(self) -> List[Dict]:
        """Get only Standard Model fundamental particles (no antiparticles or composites)"""
        return [p for p in self.particles
                if not p.get('_is_antiparticle', False)
                and not p.get('is_composite', False)]

    def get_all_particles(self) -> List[Dict]:
        """Get all loaded particles"""
        return self.particles

    def get_particle_count(self) -> int:
        """Get total number of loaded particles"""
        return len(self.particles)


# Singleton instance for convenience
_loader_instance: Optional[QuarkDataLoader] = None


def get_quark_loader() -> QuarkDataLoader:
    """Get or create the singleton loader instance"""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = QuarkDataLoader()
        _loader_instance.load_all_particles()
    return _loader_instance
