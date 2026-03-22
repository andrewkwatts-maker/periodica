"""
Optical Properties Module
=========================

Rigorous spectral representation of light-material interactions.

Instead of simple RGB colors, this module provides:
- Complex refractive index: n(lambda) + i*k(lambda)
- Dielectric function: epsilon(lambda) = epsilon_1 + i*epsilon_2
- Reflectance R(lambda), Transmittance T(lambda), Absorbance A(lambda)
- Fresnel equations for interface reflectance
- Drude model for metals
- Lorentz oscillator model for dielectrics
- Sellmeier equation for transparent materials
- Temperature-dependent optical properties

Physical Models:
    - Drude (free electrons): epsilon(omega) = 1 - omega_p^2 / (omega^2 + i*gamma*omega)
    - Lorentz (bound electrons): epsilon(omega) = sum_j [f_j * omega_j^2 / (omega_j^2 - omega^2 - i*gamma_j*omega)]
    - Sellmeier (refractive index): n^2 = 1 + sum_j [B_j * lambda^2 / (lambda^2 - C_j)]
    - Cauchy (simple n): n = A + B/lambda^2 + C/lambda^4

References:
    - M. Born & E. Wolf, "Principles of Optics"
    - M. Fox, "Optical Properties of Solids"
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import json


# Physical constants
C_LIGHT = 299792458  # m/s
H_PLANCK = 6.62607e-34  # J*s
HBAR = 1.054571e-34  # J*s
EV_TO_J = 1.602176e-19  # J/eV


class OpticalModel(Enum):
    """Available optical property models."""
    CONSTANT = "constant"
    TABULATED = "tabulated"
    CAUCHY = "cauchy"
    SELLMEIER = "sellmeier"
    DRUDE = "drude"
    LORENTZ = "lorentz"
    DRUDE_LORENTZ = "drude_lorentz"


@dataclass
class DrudeParameters:
    """
    Drude model parameters for metallic response.

    The Drude model describes free electron response:
        epsilon(omega) = epsilon_inf - omega_p^2 / (omega^2 + i*gamma*omega)

    Attributes:
        epsilon_inf: High-frequency dielectric constant (typically 1-10)
        plasma_frequency_eV: Plasma frequency in eV
        damping_eV: Collision/damping rate in eV
    """
    epsilon_inf: float = 1.0
    plasma_frequency_eV: float = 9.0  # Typical for metals
    damping_eV: float = 0.1

    def epsilon_at_energy(self, energy_eV: float) -> complex:
        """Calculate complex dielectric function at given photon energy."""
        omega = energy_eV
        omega_p = self.plasma_frequency_eV
        gamma = self.damping_eV

        if abs(omega) < 1e-10:
            # DC limit
            return complex(self.epsilon_inf, 0)

        denom = omega * omega + 1j * gamma * omega
        return self.epsilon_inf - (omega_p * omega_p) / denom

    def to_dict(self) -> Dict[str, float]:
        return {
            'epsilon_inf': self.epsilon_inf,
            'plasma_frequency_eV': self.plasma_frequency_eV,
            'damping_eV': self.damping_eV
        }


@dataclass
class LorentzOscillator:
    """
    Single Lorentz oscillator for bound electron response.

    The Lorentz model describes resonant absorption:
        epsilon_j(omega) = f_j * omega_j^2 / (omega_j^2 - omega^2 - i*gamma_j*omega)

    Attributes:
        oscillator_strength: Dimensionless oscillator strength f_j
        resonance_energy_eV: Resonance energy in eV
        damping_eV: Damping/broadening in eV
    """
    oscillator_strength: float = 1.0
    resonance_energy_eV: float = 4.0  # ~310 nm (UV)
    damping_eV: float = 0.5

    def epsilon_at_energy(self, energy_eV: float) -> complex:
        """Calculate contribution to dielectric function."""
        omega = energy_eV
        omega_0 = self.resonance_energy_eV
        gamma = self.damping_eV
        f = self.oscillator_strength

        denom = omega_0 * omega_0 - omega * omega - 1j * gamma * omega
        return f * omega_0 * omega_0 / denom

    def to_dict(self) -> Dict[str, float]:
        return {
            'oscillator_strength': self.oscillator_strength,
            'resonance_energy_eV': self.resonance_energy_eV,
            'damping_eV': self.damping_eV
        }


@dataclass
class SellmeierCoefficients:
    """
    Sellmeier equation coefficients for transparent materials.

    n^2(lambda) = 1 + sum_j [B_j * lambda^2 / (lambda^2 - C_j)]

    where lambda is in micrometers and C_j is in um^2.

    Attributes:
        B_coefficients: List of B_j values (dimensionless)
        C_coefficients: List of C_j values (um^2)
    """
    B_coefficients: List[float] = field(default_factory=lambda: [1.03961212])
    C_coefficients: List[float] = field(default_factory=lambda: [0.00600069])

    def n_squared(self, wavelength_nm: float) -> float:
        """Calculate n^2 at given wavelength."""
        lambda_um = wavelength_nm / 1000.0
        lambda_sq = lambda_um * lambda_um

        n_sq = 1.0
        for B, C in zip(self.B_coefficients, self.C_coefficients):
            if abs(lambda_sq - C) > 1e-10:
                n_sq += B * lambda_sq / (lambda_sq - C)
        return n_sq

    def refractive_index(self, wavelength_nm: float) -> float:
        """Calculate refractive index at given wavelength."""
        n_sq = self.n_squared(wavelength_nm)
        return math.sqrt(max(n_sq, 0))

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            'B_coefficients': self.B_coefficients,
            'C_coefficients': self.C_coefficients
        }


@dataclass
class CauchyCoefficients:
    """
    Cauchy equation for simple refractive index.

    n(lambda) = A + B/lambda^2 + C/lambda^4

    where lambda is in micrometers.

    Attributes:
        A: Constant term
        B: Coefficient for 1/lambda^2 term (um^2)
        C: Coefficient for 1/lambda^4 term (um^4)
    """
    A: float = 1.5
    B: float = 0.004
    C: float = 0.0

    def refractive_index(self, wavelength_nm: float) -> float:
        """Calculate refractive index at given wavelength."""
        lambda_um = wavelength_nm / 1000.0
        return (self.A +
                self.B / (lambda_um * lambda_um) +
                self.C / (lambda_um ** 4))

    def to_dict(self) -> Dict[str, float]:
        return {'A': self.A, 'B': self.B, 'C': self.C}


@dataclass
class SpectralDataPoint:
    """Single point of tabulated spectral data."""
    wavelength_nm: float
    value: float

    def to_dict(self) -> Dict[str, float]:
        return {'wavelength_nm': self.wavelength_nm, 'value': self.value}


class SpectralProperty:
    """
    Spectral property that varies with wavelength.

    Can be initialized from:
    - Constant value
    - Tabulated data (wavelength, value pairs)
    - Parametric model (Cauchy, Sellmeier, Drude, Lorentz)
    """

    def __init__(
        self,
        model: OpticalModel = OpticalModel.CONSTANT,
        constant_value: float = 1.0,
        tabulated_data: Optional[List[SpectralDataPoint]] = None,
        cauchy: Optional[CauchyCoefficients] = None,
        sellmeier: Optional[SellmeierCoefficients] = None,
        drude: Optional[DrudeParameters] = None,
        lorentz_oscillators: Optional[List[LorentzOscillator]] = None
    ):
        self.model = model
        self.constant_value = constant_value
        self.tabulated_data = tabulated_data or []
        self.cauchy = cauchy
        self.sellmeier = sellmeier
        self.drude = drude
        self.lorentz_oscillators = lorentz_oscillators or []

        # Sort tabulated data by wavelength for interpolation
        if self.tabulated_data:
            self.tabulated_data.sort(key=lambda p: p.wavelength_nm)

    def evaluate(self, wavelength_nm: float) -> float:
        """
        Evaluate the property at a given wavelength.

        Args:
            wavelength_nm: Wavelength in nanometers

        Returns:
            Property value at that wavelength
        """
        if self.model == OpticalModel.CONSTANT:
            return self.constant_value

        elif self.model == OpticalModel.TABULATED:
            return self._interpolate_tabulated(wavelength_nm)

        elif self.model == OpticalModel.CAUCHY:
            if self.cauchy:
                return self.cauchy.refractive_index(wavelength_nm)
            return self.constant_value

        elif self.model == OpticalModel.SELLMEIER:
            if self.sellmeier:
                return self.sellmeier.refractive_index(wavelength_nm)
            return self.constant_value

        else:
            # For Drude/Lorentz, return real part of n
            n_complex = self.complex_refractive_index(wavelength_nm)
            return n_complex.real

    def complex_refractive_index(self, wavelength_nm: float) -> complex:
        """
        Calculate complex refractive index n + i*k.

        k is the extinction coefficient (related to absorption).
        """
        epsilon = self.dielectric_function(wavelength_nm)
        # n + ik = sqrt(epsilon)
        return self._complex_sqrt(epsilon)

    def dielectric_function(self, wavelength_nm: float) -> complex:
        """
        Calculate complex dielectric function epsilon_1 + i*epsilon_2.

        This is the fundamental optical property from which others derive.
        """
        # Convert wavelength to photon energy in eV
        energy_eV = 1239.84 / wavelength_nm  # hc/lambda in eV*nm

        epsilon = complex(1.0, 0.0)

        if self.model in [OpticalModel.DRUDE, OpticalModel.DRUDE_LORENTZ]:
            if self.drude:
                epsilon = self.drude.epsilon_at_energy(energy_eV)

        if self.model in [OpticalModel.LORENTZ, OpticalModel.DRUDE_LORENTZ]:
            for osc in self.lorentz_oscillators:
                epsilon += osc.epsilon_at_energy(energy_eV)

        if self.model == OpticalModel.SELLMEIER and self.sellmeier:
            n_sq = self.sellmeier.n_squared(wavelength_nm)
            epsilon = complex(n_sq, 0)

        if self.model == OpticalModel.CAUCHY and self.cauchy:
            n = self.cauchy.refractive_index(wavelength_nm)
            epsilon = complex(n * n, 0)

        return epsilon

    def _complex_sqrt(self, z: complex) -> complex:
        """Compute principal square root of complex number."""
        r = abs(z)
        theta = math.atan2(z.imag, z.real)
        return complex(
            math.sqrt(r) * math.cos(theta / 2),
            math.sqrt(r) * math.sin(theta / 2)
        )

    def _interpolate_tabulated(self, wavelength_nm: float) -> float:
        """Linear interpolation of tabulated data."""
        if not self.tabulated_data:
            return self.constant_value

        # Boundary handling
        if wavelength_nm <= self.tabulated_data[0].wavelength_nm:
            return self.tabulated_data[0].value
        if wavelength_nm >= self.tabulated_data[-1].wavelength_nm:
            return self.tabulated_data[-1].value

        # Find bracketing points
        for i in range(len(self.tabulated_data) - 1):
            w1 = self.tabulated_data[i].wavelength_nm
            w2 = self.tabulated_data[i + 1].wavelength_nm
            if w1 <= wavelength_nm <= w2:
                v1 = self.tabulated_data[i].value
                v2 = self.tabulated_data[i + 1].value
                t = (wavelength_nm - w1) / (w2 - w1)
                return v1 + t * (v2 - v1)

        return self.constant_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            'model': self.model.value,
            'constant_value': self.constant_value
        }
        if self.tabulated_data:
            result['tabulated_data'] = [p.to_dict() for p in self.tabulated_data]
        if self.cauchy:
            result['cauchy'] = self.cauchy.to_dict()
        if self.sellmeier:
            result['sellmeier'] = self.sellmeier.to_dict()
        if self.drude:
            result['drude'] = self.drude.to_dict()
        if self.lorentz_oscillators:
            result['lorentz_oscillators'] = [o.to_dict() for o in self.lorentz_oscillators]
        return result


class OpticalMaterial:
    """
    Complete optical characterization of a material.

    Includes all spectral response functions needed for simulation:
    - Complex refractive index (n, k)
    - Reflectance, transmittance, absorbance
    - Emissivity for thermal radiation
    - Temperature dependence
    """

    # Standard wavelengths for visible spectrum (nm)
    VISIBLE_WAVELENGTHS = [380, 400, 450, 500, 550, 600, 650, 700, 750, 780]

    # CIE standard observer color matching functions (simplified)
    # For wavelengths: 400, 450, 500, 550, 600, 650, 700 nm
    CIE_X = [0.014, 0.336, 0.005, 0.433, 1.062, 0.283, 0.011]
    CIE_Y = [0.000, 0.038, 0.323, 0.995, 0.631, 0.107, 0.004]
    CIE_Z = [0.068, 1.772, 0.272, 0.009, 0.000, 0.000, 0.000]

    def __init__(
        self,
        name: str,
        refractive_index: Optional[SpectralProperty] = None,
        extinction_coefficient: Optional[SpectralProperty] = None,
        emissivity: float = 0.9,
        surface_roughness_um: float = 0.1
    ):
        self.name = name
        self.refractive_index = refractive_index or SpectralProperty(
            model=OpticalModel.CONSTANT, constant_value=1.5
        )
        self.extinction_coefficient = extinction_coefficient or SpectralProperty(
            model=OpticalModel.CONSTANT, constant_value=0.0
        )
        self.emissivity = emissivity
        self.surface_roughness_um = surface_roughness_um

    def reflectance_normal(self, wavelength_nm: float) -> float:
        """
        Calculate reflectance at normal incidence using Fresnel equations.

        R = |n - 1|^2 / |n + 1|^2 for normal incidence from air (n_air = 1)
        where n is complex: n_real + i*k
        """
        n = self.refractive_index.evaluate(wavelength_nm)
        k = self.extinction_coefficient.evaluate(wavelength_nm)

        n_complex = complex(n, k)
        numerator = abs(n_complex - 1) ** 2
        denominator = abs(n_complex + 1) ** 2

        if denominator < 1e-10:
            return 0.0
        return numerator / denominator

    def reflectance_angle(self, wavelength_nm: float,
                          angle_rad: float, polarization: str = 's') -> float:
        """
        Calculate reflectance at arbitrary angle using Fresnel equations.

        Args:
            wavelength_nm: Wavelength in nm
            angle_rad: Incidence angle from normal in radians
            polarization: 's' (perpendicular) or 'p' (parallel)

        Returns:
            Reflectance (0-1)
        """
        n1 = 1.0  # Air
        n2 = self.refractive_index.evaluate(wavelength_nm)

        cos_i = math.cos(angle_rad)
        sin_i = math.sin(angle_rad)

        # Snell's law for transmission angle
        sin_t = n1 * sin_i / n2
        if abs(sin_t) > 1:
            # Total internal reflection
            return 1.0
        cos_t = math.sqrt(1 - sin_t * sin_t)

        if polarization == 's':
            # s-polarization (TE)
            rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
            return rs * rs
        else:
            # p-polarization (TM)
            rp = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
            return rp * rp

    def transmittance(self, wavelength_nm: float, thickness_m: float) -> float:
        """
        Calculate transmittance through material of given thickness.

        Uses Beer-Lambert law: T = exp(-alpha * d)
        where alpha = 4*pi*k/lambda is the absorption coefficient.
        """
        k = self.extinction_coefficient.evaluate(wavelength_nm)
        lambda_m = wavelength_nm * 1e-9

        # Absorption coefficient
        alpha = 4 * math.pi * k / lambda_m

        # Beer-Lambert law
        return math.exp(-alpha * thickness_m)

    def absorbance(self, wavelength_nm: float, thickness_m: float) -> float:
        """Calculate absorbance A = 1 - R - T."""
        R = self.reflectance_normal(wavelength_nm)
        T = self.transmittance(wavelength_nm, thickness_m)
        return 1.0 - R - T

    def to_rgb(self, illuminant: str = 'D65') -> Tuple[int, int, int]:
        """
        Convert spectral reflectance to RGB color.

        Integrates reflectance weighted by CIE color matching functions
        and standard illuminant.

        Args:
            illuminant: 'D65' (daylight) or 'A' (incandescent)

        Returns:
            Tuple of (R, G, B) values 0-255
        """
        # Sample wavelengths
        wavelengths = [400, 450, 500, 550, 600, 650, 700]

        # Calculate reflectances
        reflectances = [self.reflectance_normal(w) for w in wavelengths]

        # Integrate with CIE functions (simplified)
        X = sum(r * x for r, x in zip(reflectances, self.CIE_X)) / len(wavelengths)
        Y = sum(r * y for r, y in zip(reflectances, self.CIE_Y)) / len(wavelengths)
        Z = sum(r * z for r, z in zip(reflectances, self.CIE_Z)) / len(wavelengths)

        # XYZ to sRGB transformation matrix
        R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

        # Gamma correction and clamp
        def gamma_correct(v):
            if v <= 0.0031308:
                return 12.92 * v
            return 1.055 * (v ** (1/2.4)) - 0.055

        R = max(0, min(1, gamma_correct(R)))
        G = max(0, min(1, gamma_correct(G)))
        B = max(0, min(1, gamma_correct(B)))

        return (int(R * 255), int(G * 255), int(B * 255))

    def blackbody_emission_rgb(self, temperature_K: float) -> Tuple[int, int, int]:
        """
        Calculate RGB color of thermal emission at given temperature.

        Uses Planck's law weighted by CIE color matching functions.
        """
        if temperature_K < 100:
            return (0, 0, 0)

        wavelengths = [400, 450, 500, 550, 600, 650, 700]

        # Planck function B(lambda, T)
        def planck(wavelength_nm, T):
            lambda_m = wavelength_nm * 1e-9
            c1 = 2 * H_PLANCK * C_LIGHT * C_LIGHT
            c2 = H_PLANCK * C_LIGHT / (1.38065e-23)  # hc/k
            exp_term = math.exp(c2 / (lambda_m * T))
            if exp_term > 1e30:
                return 0.0
            return c1 / (lambda_m ** 5 * (exp_term - 1))

        # Calculate spectral radiance weighted by emissivity
        radiances = [self.emissivity * planck(w, temperature_K) for w in wavelengths]

        # Normalize
        max_rad = max(radiances) if max(radiances) > 0 else 1.0
        radiances = [r / max_rad for r in radiances]

        # Integrate with CIE functions
        X = sum(r * x for r, x in zip(radiances, self.CIE_X)) / len(wavelengths)
        Y = sum(r * y for r, y in zip(radiances, self.CIE_Y)) / len(wavelengths)
        Z = sum(r * z for r, z in zip(radiances, self.CIE_Z)) / len(wavelengths)

        # XYZ to sRGB
        R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
        G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
        B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

        # Normalize and clamp
        max_val = max(R, G, B) if max(R, G, B) > 0 else 1.0
        R, G, B = R / max_val, G / max_val, B / max_val

        return (
            int(max(0, min(1, R)) * 255),
            int(max(0, min(1, G)) * 255),
            int(max(0, min(1, B)) * 255)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'name': self.name,
            'refractive_index': self.refractive_index.to_dict(),
            'extinction_coefficient': self.extinction_coefficient.to_dict(),
            'emissivity': self.emissivity,
            'surface_roughness_um': self.surface_roughness_um
        }


# Predefined optical materials
def create_steel_optical() -> OpticalMaterial:
    """Create optical properties for steel (iron-based)."""
    # Iron Drude parameters (approximate)
    drude = DrudeParameters(
        epsilon_inf=1.0,
        plasma_frequency_eV=4.0,
        damping_eV=0.6
    )

    n_property = SpectralProperty(
        model=OpticalModel.DRUDE,
        drude=drude
    )

    return OpticalMaterial(
        name="Steel",
        refractive_index=n_property,
        extinction_coefficient=SpectralProperty(
            model=OpticalModel.CONSTANT,
            constant_value=3.5
        ),
        emissivity=0.35,
        surface_roughness_um=0.5
    )


def create_glass_optical() -> OpticalMaterial:
    """Create optical properties for borosilicate glass."""
    sellmeier = SellmeierCoefficients(
        B_coefficients=[1.03961212, 0.231792344, 1.01046945],
        C_coefficients=[0.00600069867, 0.0200179144, 103.560653]
    )

    n_property = SpectralProperty(
        model=OpticalModel.SELLMEIER,
        sellmeier=sellmeier
    )

    return OpticalMaterial(
        name="Borosilicate Glass",
        refractive_index=n_property,
        extinction_coefficient=SpectralProperty(
            model=OpticalModel.CONSTANT,
            constant_value=1e-7  # Very low absorption in visible
        ),
        emissivity=0.92,
        surface_roughness_um=0.01
    )


def create_granite_optical() -> OpticalMaterial:
    """Create optical properties for granite (heterogeneous)."""
    # Tabulated reflectance data for typical gray granite
    tabulated = [
        SpectralDataPoint(400, 0.15),
        SpectralDataPoint(500, 0.20),
        SpectralDataPoint(600, 0.22),
        SpectralDataPoint(700, 0.25),
    ]

    n_property = SpectralProperty(
        model=OpticalModel.TABULATED,
        tabulated_data=tabulated
    )

    return OpticalMaterial(
        name="Granite",
        refractive_index=n_property,
        extinction_coefficient=SpectralProperty(
            model=OpticalModel.CONSTANT,
            constant_value=0.01
        ),
        emissivity=0.90,
        surface_roughness_um=10.0  # Rough natural surface
    )
