"""
Element Data Module
Contains all element property data dictionaries and helper functions.
"""

# NIST Ionization Energies (eV) - First ionization energy
IE_DATA = {
    "H": 13.59844, "He": 24.58738, "Li": 5.39172, "Be": 9.32270, "B": 8.29803, "C": 11.26030,
    "N": 14.53414, "O": 13.61806, "F": 17.42282, "Ne": 21.56454, "Na": 5.13908, "Mg": 7.64624,
    "Al": 5.98577, "Si": 8.15169, "P": 10.48669, "S": 10.36001, "Cl": 12.96764, "Ar": 15.75962,
    "K": 4.34066, "Ca": 6.11316, "Sc": 6.56149, "Ti": 6.82812, "V": 6.74619, "Cr": 6.76651,
    "Mn": 7.43402, "Fe": 7.90243, "Co": 7.88101, "Ni": 7.63980, "Cu": 7.72638, "Zn": 9.39405,
    "Ga": 5.99930, "Ge": 7.89943, "As": 9.78855, "Se": 9.75239, "Br": 11.81381, "Kr": 13.99961,
    "Rb": 4.17713, "Sr": 5.69485, "Y": 6.21726, "Zr": 6.63390, "Nb": 6.75885, "Mo": 7.09243,
    "Tc": 7.11938, "Ru": 7.36050, "Rh": 7.45890, "Pd": 8.33686, "Ag": 7.57624, "Cd": 8.99382,
    "In": 5.78636, "Sn": 7.34392, "Sb": 8.60839, "Te": 9.00966, "I": 10.45126, "Xe": 12.12984,
    "Cs": 3.89390, "Ba": 5.21170, "La": 5.5769, "Ce": 5.5387, "Pr": 5.473, "Nd": 5.5250,
    "Pm": 5.582, "Sm": 5.6437, "Eu": 5.6704, "Gd": 6.1498, "Tb": 5.8638, "Dy": 5.9389,
    "Ho": 6.0215, "Er": 6.1077, "Tm": 6.18431, "Yb": 6.25416, "Lu": 5.4259,
    "Hf": 6.82507, "Ta": 7.54957, "W": 7.86403, "Re": 7.83352, "Os": 8.43823, "Ir": 8.96702,
    "Pt": 8.95883, "Au": 9.22553, "Hg": 10.43750, "Tl": 6.10829, "Pb": 7.41663, "Bi": 7.28556,
    "Po": 8.41671, "At": 9.31751, "Rn": 10.74850, "Fr": 3.89939, "Ra": 5.27892,
    "Ac": 5.380226, "Th": 6.3067, "Pa": 5.89, "U": 6.19405, "Np": 6.2657, "Pu": 6.0260,
    "Am": 5.9738, "Cm": 5.9915, "Bk": 6.1979, "Cf": 6.2817, "Es": 6.42, "Fm": 6.50,
    "Md": 6.58, "No": 6.65, "Lr": 4.96, "Rf": 6.01, "Db": 6.8, "Sg": 7.8,
    "Bh": 7.7, "Hs": 7.6, "Mt": 8.0, "Ds": 8.3, "Rg": 9.0, "Cn": 11.3,
    "Nh": 7.4, "Fl": 8.5, "Mc": 7.0, "Lv": 5.3, "Ts": 7.1, "Og": 8.9
}

# Electronegativity (Pauling scale)
ELECTRONEGATIVITY = {
    "H": 2.20, "He": 0.0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Ne": 0.0, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58,
    "Cl": 3.16, "Ar": 0.0, "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01,
    "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.6, "Mo": 2.16, "Tc": 1.9, "Ru": 2.2, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.1, "I": 2.66, "Xe": 2.6, "Cs": 0.79, "Ba": 0.89,
    "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.2, "Gd": 1.20,
    "Tb": 1.1, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.1, "Lu": 1.27, "Hf": 1.3,
    "Ta": 1.5, "W": 2.36, "Re": 1.9, "Os": 2.2, "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
    "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.0, "At": 2.2, "Rn": 2.2, "Fr": 0.7, "Ra": 0.9,
    "Ac": 1.1, "Th": 1.3, "Pa": 1.5, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.13, "Cm": 1.28,
    "Bk": 1.3, "Cf": 1.3, "Es": 1.3, "Fm": 1.3, "Md": 1.3, "No": 1.3, "Lr": 1.3,
    "Rf": 1.3, "Db": 1.5, "Sg": 1.8, "Bh": 1.9, "Hs": 2.0, "Mt": 2.2, "Ds": 2.3, "Rg": 2.4,
    "Cn": 2.5, "Nh": 1.6, "Fl": 1.8, "Mc": 1.9, "Lv": 2.0, "Ts": 2.2, "Og": 2.4
}

# Primary Emission Wavelength (nm) - Most prominent spectral line for each element
# Based on flame tests, characteristic emission, or strongest visible line
PRIMARY_EMISSION_WAVELENGTH = {
    "H": 656.3,    # H-alpha (Balmer series, red)
    "He": 587.6,   # Yellow line
    "Li": 670.8,   # Deep red (flame test)
    "Be": 234.9,   # UV line
    "B": 518.0,    # Green
    "C": 247.9,    # UV
    "N": 500.5,    # Blue-green
    "O": 777.4,    # Red-orange
    "F": 685.6,    # Red
    "Ne": 640.2,   # Red-orange (neon sign color)
    "Na": 589.0,   # Yellow D-line (flame test)
    "Mg": 285.2,   # UV (518nm green for flame)
    "Al": 396.2,   # Violet
    "Si": 390.6,   # Violet
    "P": 253.6,    # UV
    "S": 545.4,    # Green
    "Cl": 479.5,   # Blue-green
    "Ar": 763.5,   # Red
    "K": 766.5,    # Violet (flame test)
    "Ca": 422.7,   # Violet-blue (flame test)
    "Sc": 402.0,   # Violet
    "Ti": 498.2,   # Blue-green
    "V": 437.9,    # Violet
    "Cr": 425.4,   # Violet-blue
    "Mn": 403.3,   # Violet
    "Fe": 385.9,   # UV-violet
    "Co": 345.4,   # UV
    "Ni": 352.5,   # UV
    "Cu": 324.8,   # UV (510-515nm for flame - blue-green)
    "Zn": 213.9,   # UV
    "Ga": 417.2,   # Violet
    "Ge": 303.9,   # UV
    "As": 234.0,   # UV
    "Se": 196.0,   # UV
    "Br": 827.2,   # Near-IR
    "Kr": 587.1,   # Yellow
    "Rb": 780.0,   # Red (flame test)
    "Sr": 460.7,   # Blue (flame test)
    "Y": 412.8,    # Violet
    "Zr": 360.1,   # UV
    "Nb": 405.9,   # Violet
    "Mo": 386.4,   # Violet
    "Tc": 403.2,   # Violet (estimated)
    "Ru": 372.8,   # UV
    "Rh": 369.2,   # UV
    "Pd": 340.5,   # UV
    "Ag": 328.1,   # UV (520nm for flame)
    "Cd": 228.8,   # UV
    "In": 451.1,   # Blue (flame test)
    "Sn": 380.1,   # Violet
    "Sb": 259.8,   # UV
    "Te": 214.3,   # UV
    "I": 546.5,    # Green
    "Xe": 467.1,   # Blue
    "Cs": 852.1,   # Near-IR (blue for flame)
    "Ba": 553.6,   # Green (flame test)
    "La": 550.1,   # Green
    "Ce": 413.8,   # Violet
    "Pr": 495.1,   # Blue-green
    "Nd": 463.4,   # Blue
    "Pm": 390.0,   # Violet (estimated)
    "Sm": 476.0,   # Blue
    "Eu": 459.4,   # Blue
    "Gd": 407.9,   # Violet
    "Tb": 432.6,   # Violet-blue
    "Dy": 421.2,   # Violet
    "Ho": 405.4,   # Violet
    "Er": 400.8,   # Violet
    "Tm": 371.8,   # UV
    "Yb": 398.8,   # Violet
    "Lu": 451.9,   # Blue
}

# Most prominent VISIBLE emission wavelength (nm) - what you'd see in flame tests or discharge tubes
# This is the brightest line in the visible spectrum, not necessarily the strongest overall line
VISIBLE_EMISSION_WAVELENGTH = {
    "H": 656.3,    # H-alpha (red)
    "He": 587.6,   # Yellow
    "Li": 670.8,   # Deep red
    "Be": 556.0,   # Green (estimated visible line)
    "B": 518.0,    # Green
    "C": 495.0,    # Blue-green (estimated)
    "N": 500.5,    # Blue-green
    "O": 777.4,    # Red-orange
    "F": 685.6,    # Red
    "Ne": 640.2,   # Red-orange
    "Na": 589.0,   # Yellow (sodium D-line)
    "Mg": 518.0,   # Green (flame test)
    "Al": 396.2,   # Violet
    "Si": 390.6,   # Violet
    "P": 450.0,    # Blue (estimated visible)
    "S": 545.4,    # Green
    "Cl": 479.5,   # Blue-green
    "Ar": 763.5,   # Red
    "K": 766.5,    # Violet
    "Ca": 422.7,   # Violet-blue
    "Sc": 402.0,   # Violet
    "Ti": 498.2,   # Blue-green
    "V": 437.9,    # Violet
    "Cr": 425.4,   # Violet-blue
    "Mn": 403.3,   # Violet
    "Fe": 385.9,   # Violet
    "Co": 535.0,   # Green (estimated visible)
    "Ni": 505.0,   # Green (estimated visible)
    "Cu": 515.0,   # Blue-green (flame test)
    "Zn": 468.0,   # Blue (estimated visible)
    "Ga": 417.2,   # Violet
    "Ge": 468.0,   # Blue (estimated)
    "As": 450.0,   # Blue (estimated)
    "Se": 473.0,   # Blue (estimated)
    "Br": 478.0,   # Blue-green (estimated visible)
    "Kr": 587.1,   # Yellow
    "Rb": 780.0,   # Red
    "Sr": 460.7,   # Blue
    "Y": 412.8,    # Violet
    "Zr": 468.0,   # Blue (estimated)
    "Nb": 405.9,   # Violet
    "Mo": 386.4,   # Violet
    "Tc": 403.2,   # Violet
    "Ru": 465.0,   # Blue (estimated)
    "Rh": 470.0,   # Blue (estimated)
    "Pd": 470.0,   # Blue (estimated)
    "Ag": 520.0,   # Green (flame test)
    "Cd": 480.0,   # Blue (estimated)
    "In": 451.1,   # Blue
    "Sn": 380.1,   # Violet
    "Sb": 450.0,   # Blue (estimated)
    "Te": 470.0,   # Blue (estimated)
    "I": 546.5,    # Green
    "Xe": 467.1,   # Blue
    "Cs": 455.5,   # Blue (flame test)
    "Ba": 553.6,   # Green
    "La": 550.1,   # Green
    "Ce": 413.8,   # Violet
    "Pr": 495.1,   # Blue-green
    "Nd": 463.4,   # Blue
    "Pm": 390.0,   # Violet
    "Sm": 476.0,   # Blue
    "Eu": 459.4,   # Blue
    "Gd": 407.9,   # Violet
    "Tb": 432.6,   # Violet-blue
    "Dy": 421.2,   # Violet
    "Ho": 405.4,   # Violet
    "Er": 400.8,   # Violet
    "Tm": 465.0,   # Blue (estimated)
    "Yb": 398.8,   # Violet
    "Lu": 451.9,   # Blue
}

# Electron Affinity (kJ/mol)
ELECTRON_AFFINITY = {
    "H": 72.8, "He": 0, "Li": 59.6, "Be": 0, "B": 26.7, "C": 121.8, "N": -7, "O": 141.0,
    "F": 328.0, "Ne": 0, "Na": 52.8, "Mg": 0, "Al": 42.5, "Si": 133.6, "P": 72.0, "S": 200.4,
    "Cl": 349.0, "Ar": 0, "K": 48.4, "Ca": 2.37, "Sc": 18.1, "Ti": 7.6, "V": 50.6, "Cr": 64.3,
    "Mn": 0, "Fe": 15.7, "Co": 63.7, "Ni": 112.0, "Cu": 118.4, "Zn": 0, "Ga": 28.9, "Ge": 119.0,
    "As": 78.0, "Se": 195.0, "Br": 324.6, "Kr": 0, "Rb": 46.9, "Sr": 5.03, "Y": 29.6, "Zr": 41.1,
    "Nb": 86.1, "Mo": 71.9, "Tc": 53, "Ru": 101.3, "Rh": 110.0, "Pd": 53.7, "Ag": 125.6, "Cd": 0,
    "In": 28.9, "Sn": 107.3, "Sb": 103.2, "Te": 190.2, "I": 295.2, "Xe": 0
}

# Density (g/cm³)
DENSITY = {
    "H": 0.00009, "He": 0.00018, "Li": 0.534, "Be": 1.85, "B": 2.34, "C": 2.26, "N": 0.00125, "O": 0.00143,
    "F": 0.0017, "Ne": 0.0009, "Na": 0.971, "Mg": 1.74, "Al": 2.70, "Si": 2.33, "P": 1.82, "S": 2.07,
    "Cl": 0.0032, "Ar": 0.0018, "K": 0.862, "Ca": 1.54, "Sc": 2.99, "Ti": 4.54, "V": 6.11, "Cr": 7.19,
    "Mn": 7.44, "Fe": 7.87, "Co": 8.90, "Ni": 8.91, "Cu": 8.96, "Zn": 7.14, "Ga": 5.91, "Ge": 5.32,
    "As": 5.73, "Se": 4.81, "Br": 3.12, "Kr": 0.0037, "Rb": 1.53, "Sr": 2.64, "Y": 4.47, "Zr": 6.52,
    "Nb": 8.57, "Mo": 10.28, "Tc": 11.50, "Ru": 12.37, "Rh": 12.41, "Pd": 12.02, "Ag": 10.49, "Cd": 8.65,
    "In": 7.31, "Sn": 7.29, "Sb": 6.68, "Te": 6.24, "I": 4.93, "Xe": 0.0059
}

# Boiling Point (Kelvin)
BOILING_POINT = {
    "H": 20, "He": 4.2, "Li": 1615, "Be": 2744, "B": 4273, "C": 4098, "N": 77, "O": 90,
    "F": 85, "Ne": 27, "Na": 1156, "Mg": 1363, "Al": 2792, "Si": 3538, "P": 550, "S": 718,
    "Cl": 239, "Ar": 87, "K": 1032, "Ca": 1757, "Sc": 3109, "Ti": 3560, "V": 3680, "Cr": 2944,
    "Mn": 2334, "Fe": 3134, "Co": 3200, "Ni": 3186, "Cu": 2835, "Zn": 1180, "Ga": 2477, "Ge": 3106
}

# Atomic Radius (pm - picometers)
ATOMIC_RADIUS = {
    "H": 53, "He": 31, "Li": 167, "Be": 112, "B": 87, "C": 67, "N": 56, "O": 48,
    "F": 42, "Ne": 38, "Na": 190, "Mg": 145, "Al": 118, "Si": 111, "P": 98, "S": 88,
    "Cl": 79, "Ar": 71, "K": 243, "Ca": 194, "Sc": 184, "Ti": 176, "V": 171, "Cr": 166,
    "Mn": 161, "Fe": 156, "Co": 152, "Ni": 149, "Cu": 145, "Zn": 142, "Ga": 136, "Ge": 125,
    "As": 114, "Se": 103, "Br": 94, "Kr": 88, "Rb": 265, "Sr": 219, "Y": 212, "Zr": 206,
    "Nb": 198, "Mo": 190, "Tc": 183, "Ru": 178, "Rh": 173, "Pd": 169, "Ag": 165, "Cd": 161,
    "In": 156, "Sn": 145, "Sb": 133, "Te": 123, "I": 115, "Xe": 108, "Cs": 298, "Ba": 253,
    "La": 195, "Ce": 185, "Pr": 185, "Nd": 206, "Pm": 205, "Sm": 185, "Eu": 231, "Gd": 233,  # Fixed Pr and Sm: were 247/238, now 185/185
    "Tb": 225, "Dy": 228, "Ho": 226, "Er": 226, "Tm": 222, "Yb": 222, "Lu": 217, "Hf": 208,
    "Ta": 200, "W": 193, "Re": 188, "Os": 185, "Ir": 180, "Pt": 177, "Au": 174, "Hg": 171,
    "Tl": 156, "Pb": 154, "Bi": 143, "Po": 135, "At": 127, "Rn": 120, "Fr": 348, "Ra": 283,
    "Ac": 195, "Th": 180, "Pa": 180, "U": 175, "Np": 175, "Pu": 175, "Am": 175, "Cm": 176,
    "Bk": 170, "Cf": 186, "Es": 186, "Fm": 200, "Md": 200, "No": 200, "Lr": 200,
    "Rf": 157, "Db": 149, "Sg": 143, "Bh": 141, "Hs": 134, "Mt": 129, "Ds": 128, "Rg": 121,
    "Cn": 122, "Nh": 136, "Fl": 143, "Mc": 162, "Lv": 175, "Ts": 165, "Og": 157
}

# Melting Point (Kelvin)
MELTING_POINT = {
    "H": 14, "He": 0.95, "Li": 453, "Be": 1560, "B": 2349, "C": 3823, "N": 63, "O": 54,
    "F": 53, "Ne": 24, "Na": 371, "Mg": 923, "Al": 933, "Si": 1687, "P": 317, "S": 388,
    "Cl": 171, "Ar": 83, "K": 336, "Ca": 1115, "Sc": 1814, "Ti": 1941, "V": 2183, "Cr": 2180,
    "Mn": 1519, "Fe": 1811, "Co": 1768, "Ni": 1728, "Cu": 1358, "Zn": 693, "Ga": 303, "Ge": 1211,
    "As": 1090, "Se": 494, "Br": 266, "Kr": 116, "Rb": 312, "Sr": 1050, "Y": 1799, "Zr": 2128,
    "Nb": 2750, "Mo": 2896, "Tc": 2430, "Ru": 2607, "Rh": 2237, "Pd": 1828, "Ag": 1235, "Cd": 594,
    "In": 430, "Sn": 505, "Sb": 904, "Te": 723, "I": 387, "Xe": 161, "Cs": 301, "Ba": 1000,
    "La": 1193, "Ce": 1071, "Pr": 1208, "Nd": 1297, "Pm": 1315, "Sm": 1345, "Eu": 1095, "Gd": 1585,
    "Tb": 1629, "Dy": 1680, "Ho": 1734, "Er": 1802, "Tm": 1818, "Yb": 1097, "Lu": 1925, "Hf": 2506,
    "Ta": 3290, "W": 3695, "Re": 3459, "Os": 3306, "Ir": 2719, "Pt": 2041, "Au": 1337, "Hg": 234,
    "Tl": 577, "Pb": 600, "Bi": 545, "Po": 527, "At": 575, "Rn": 202, "Fr": 300, "Ra": 973,
    "Ac": 1323, "Th": 2023, "Pa": 1841, "U": 1405, "Np": 912, "Pu": 913, "Am": 1449, "Cm": 1613,
    "Bk": 1323, "Cf": 1173, "Es": 1133, "Fm": 1800, "Md": 1100, "No": 1100, "Lr": 1900,
    "Rf": 2400, "Db": 2900, "Sg": 3000, "Bh": 2900, "Hs": 2700, "Mt": 2500, "Ds": 2300, "Rg": 2100,
    "Cn": 283, "Nh": 700, "Fl": 340, "Mc": 670, "Lv": 700, "Ts": 723, "Og": 350
}

# Stable isotopes (mass numbers and approximate natural abundance %)
ISOTOPES = {
    "H": [(1, 99.98), (2, 0.02)],
    "He": [(3, 0.0001), (4, 99.9999)],
    "Li": [(6, 7.5), (7, 92.5)],
    "C": [(12, 98.93), (13, 1.07)],  # Fixed: was 98.9/1.1
    "N": [(14, 99.632), (15, 0.368)],  # Fixed: was 99.6/0.4
    "O": [(16, 99.757), (17, 0.038), (18, 0.205)],  # Fixed minor precision
    "Ne": [(20, 90.48), (21, 0.27), (22, 9.25)],
    "Si": [(28, 92.23), (29, 4.67), (30, 3.10)],  # Fixed precision
    "S": [(32, 94.99), (33, 0.75), (34, 4.25), (36, 0.01)],  # Fixed minor precision
    "Cl": [(35, 75.76), (37, 24.24)],  # Fixed precision
    "Ar": [(36, 0.3336), (38, 0.0629), (40, 99.6035)],  # Fixed precision
    "Ca": [(40, 96.941), (42, 0.647), (43, 0.135), (44, 2.086), (46, 0.004), (48, 0.187)],  # Fixed precision
    "Fe": [(54, 5.845), (56, 91.754), (57, 2.119), (58, 0.282)],  # Fixed: all values corrected
    "Cu": [(63, 69.15), (65, 30.85)],  # Fixed: was 69.2/30.8
    "Zn": [(64, 49.17), (66, 27.73), (67, 4.04), (68, 18.45), (70, 0.61)],  # Fixed: all values corrected
    "Br": [(79, 50.69), (81, 49.31)],  # Fixed precision
    "Kr": [(78, 0.355), (80, 2.286), (82, 11.593), (83, 11.500), (84, 56.987), (86, 17.279)],  # Fixed precision
    "Sr": [(84, 0.56), (86, 9.86), (87, 7.00), (88, 82.58)],
    "Ag": [(107, 51.839), (109, 48.161)],  # Fixed: was 51.8/48.2
    "Sn": [(112, 0.97), (114, 0.66), (115, 0.34), (116, 14.54), (117, 7.68), (118, 24.22), (119, 8.59), (120, 32.58), (122, 4.63), (124, 5.79)],  # Fixed precision
    "Xe": [(124, 0.095), (126, 0.089), (128, 1.910), (129, 26.401), (130, 4.071), (131, 21.232), (132, 26.909), (134, 10.436), (136, 8.857)],  # Fixed precision
    "Ba": [(130, 0.106), (132, 0.101), (134, 2.417), (135, 6.592), (136, 7.854), (137, 11.232), (138, 71.698)],  # Fixed precision
}

# Orbital blocks
BLOCKS = {
    's': ["H", "He", "Li", "Be", "Na", "Mg", "K", "Ca", "Rb", "Sr", "Cs", "Ba", "Fr", "Ra"],
    'p': ["B", "C", "N", "O", "F", "Ne", "Al", "Si", "P", "S", "Cl", "Ar",
          "Ga", "Ge", "As", "Se", "Br", "Kr", "In", "Sn", "Sb", "Te", "I", "Xe",
          "Tl", "Pb", "Bi", "Po", "At", "Rn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"],
    'd': ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
          "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
          "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"],
    'f': ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
          "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
}

# Periods - which elements are in each period (1-7)
PERIODS = [
    ["H", "He"],
    ["Li", "Be", "B", "C", "N", "O", "F", "Ne"],
    ["Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"],
    ["K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
     "Ga", "Ge", "As", "Se", "Br", "Kr"],
    ["Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
     "In", "Sn", "Sb", "Te", "I", "Xe"],
    ["Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
     "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
     "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    ["Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
     "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
     "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
]


# Helper functions
def get_electron_config(z):
    """Get electron configuration for atomic number z"""
    # Simplified electron configurations
    configs = {
        1: "1s¹", 2: "1s²", 3: "[He] 2s¹", 4: "[He] 2s²", 5: "[He] 2s² 2p¹",
        6: "[He] 2s² 2p²", 7: "[He] 2s² 2p³", 8: "[He] 2s² 2p⁴", 9: "[He] 2s² 2p⁵",
        10: "[He] 2s² 2p⁶", 11: "[Ne] 3s¹", 12: "[Ne] 3s²", 13: "[Ne] 3s² 3p¹",
        14: "[Ne] 3s² 3p²", 15: "[Ne] 3s² 3p³", 16: "[Ne] 3s² 3p⁴", 17: "[Ne] 3s² 3p⁵",
        18: "[Ne] 3s² 3p⁶", 19: "[Ar] 4s¹", 20: "[Ar] 4s²"
    }
    return configs.get(z, f"Z={z}")


def get_electron_shell_distribution(z):
    """
    Get electron distribution across shells for atomic number z.
    Returns list of electron counts per shell [shell1, shell2, shell3, ...]
    Following the 2n² rule with actual filling order.
    """
    # Maximum electrons per shell: [2, 8, 18, 32, 32, 18, 8]
    max_per_shell = [2, 8, 18, 32, 32, 18, 8]
    shells = [0] * 7

    remaining = z

    # Fill shells in the actual order electrons fill
    # Simplified model: fill sequentially with some exceptions for d and f blocks
    if z <= 2:  # Period 1: 1s
        shells[0] = z
    elif z <= 10:  # Period 2: 2s, 2p
        shells[0] = 2
        shells[1] = z - 2
    elif z <= 18:  # Period 3: 3s, 3p
        shells[0] = 2
        shells[1] = 8
        shells[2] = z - 10
    elif z <= 36:  # Period 4: 4s, 3d, 4p
        shells[0] = 2
        shells[1] = 8
        shells[2] = 18
        shells[3] = z - 28
    elif z <= 54:  # Period 5: 5s, 4d, 5p
        shells[0] = 2
        shells[1] = 8
        shells[2] = 18
        shells[3] = 18
        shells[4] = z - 46
    elif z <= 86:  # Period 6: 6s, 4f, 5d, 6p
        shells[0] = 2
        shells[1] = 8
        shells[2] = 18
        shells[3] = 32
        shells[4] = 18
        shells[5] = z - 78
    else:  # Period 7: 7s, 5f, 6d, 7p
        shells[0] = 2
        shells[1] = 8
        shells[2] = 18
        shells[3] = 32
        shells[4] = 32
        shells[5] = 18
        shells[6] = z - 110

    # Return only filled shells
    return [s for s in shells if s > 0]


def get_electron_quantum_numbers(z):
    """
    Get quantum numbers (n, l, m) for each electron in an atom.
    Returns list of tuples: [(n, l, m), (n, l, m), ...]
    Fills orbitals according to Aufbau principle.

    Args:
        z: Atomic number

    Returns:
        List of (n, l, m) tuples for each electron
    """
    # Filling order: 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d, 7p
    filling_order = [
        (1, 0),  # 1s
        (2, 0),  # 2s
        (2, 1),  # 2p
        (3, 0),  # 3s
        (3, 1),  # 3p
        (4, 0),  # 4s
        (3, 2),  # 3d
        (4, 1),  # 4p
        (5, 0),  # 5s
        (4, 2),  # 4d
        (5, 1),  # 5p
        (6, 0),  # 6s
        (4, 3),  # 4f
        (5, 2),  # 5d
        (6, 1),  # 6p
        (7, 0),  # 7s
        (5, 3),  # 5f
        (6, 2),  # 6d
        (7, 1),  # 7p
    ]

    electrons = []
    remaining = z

    for n, l in filling_order:
        if remaining <= 0:
            break

        # Maximum electrons in this subshell: 2(2l + 1)
        max_electrons = 2 * (2 * l + 1)
        electrons_to_add = min(remaining, max_electrons)

        # Fill m values from -l to +l, with spin up and down
        for m in range(-l, l + 1):
            if electrons_to_add <= 0:
                break
            # Add spin up electron
            electrons.append((n, l, m))
            electrons_to_add -= 1
            remaining -= 1

            if electrons_to_add <= 0:
                break
            # Add spin down electron (same n, l, m but opposite spin)
            electrons.append((n, l, m))
            electrons_to_add -= 1
            remaining -= 1

    return electrons


def get_valence_electrons(z, block):
    """Get number of valence electrons"""
    if block == 's':
        return ((z - 1) % 2) + 1 if z <= 2 else ((z - 3) % 8) + 1
    elif block == 'p':
        return ((z - 5) % 8) + 1
    elif block == 'd':
        return min(((z - 21) % 18) + 1, 10)
    elif block == 'f':
        return min(((z - 57) % 32) + 1, 14)
    return 1


def get_block(symbol):
    """Get the orbital block for an element symbol"""
    for block, elements in BLOCKS.items():
        if symbol in elements:
            return block
    return 's'


def get_period(symbol):
    """Get the period number for an element symbol"""
    for period_idx, period in enumerate(PERIODS):
        if symbol in period:
            return period_idx + 1
    return 1


def get_atomic_number(symbol):
    """Get atomic number from element symbol"""
    for period_idx, period in enumerate(PERIODS):
        if symbol in period:
            elem_idx = period.index(symbol)
            return sum(len(p) for p in PERIODS[:period_idx]) + elem_idx + 1
    return 0


def get_property_metadata(property_name):
    """
    Get metadata for a property including units, data range, and type.

    Returns dict with:
        - unit: string for display (e.g., "eV", "K", "pm")
        - min_value: minimum value in dataset
        - max_value: maximum value in dataset
        - type: "color", "size", or "intensity"
    """
    metadata = {
        "atomic_number": {
            "unit": "",
            "min_value": 1,
            "max_value": 118,
            "type": "numeric"
        },
        "atomic_mass": {
            "unit": "u",
            "min_value": 1.008,
            "max_value": 294,
            "type": "numeric"
        },
        "ionization": {
            "unit": "eV",
            "min_value": min(IE_DATA.values()),
            "max_value": max(IE_DATA.values()),
            "type": "numeric"
        },
        "electronegativity": {
            "unit": "",
            "min_value": min(v for v in ELECTRONEGATIVITY.values() if v > 0),
            "max_value": max(ELECTRONEGATIVITY.values()),
            "type": "numeric"
        },
        "melting": {
            "unit": "K",
            "min_value": min(MELTING_POINT.values()),
            "max_value": max(MELTING_POINT.values()),
            "type": "numeric"
        },
        "boiling": {
            "unit": "K",
            "min_value": min(BOILING_POINT.values()),
            "max_value": max(BOILING_POINT.values()),
            "type": "numeric"
        },
        "radius": {
            "unit": "pm",
            "min_value": min(ATOMIC_RADIUS.values()),
            "max_value": max(ATOMIC_RADIUS.values()),
            "type": "numeric"
        },
        "density": {
            "unit": "g/cm³",
            "min_value": min(DENSITY.values()),
            "max_value": max(DENSITY.values()),
            "type": "numeric"
        },
        "electron_affinity": {
            "unit": "kJ/mol",
            "min_value": min(ELECTRON_AFFINITY.values()),
            "max_value": max(ELECTRON_AFFINITY.values()),
            "type": "numeric"
        },
        "valence": {
            "unit": "electrons",
            "min_value": 1,
            "max_value": 8,
            "type": "numeric"
        },
        "group": {
            "unit": "",
            "min_value": 1,
            "max_value": 18,
            "type": "numeric"
        },
        "period": {
            "unit": "",
            "min_value": 1,
            "max_value": 7,
            "type": "numeric"
        },
        "specific_heat": {
            "unit": "J/(g·K)",
            "min_value": 0.1,
            "max_value": 15,
            "type": "numeric"
        },
        "thermal_conductivity": {
            "unit": "W/(m·K)",
            "min_value": 0.01,
            "max_value": 500,
            "type": "numeric"
        },
        "electrical_conductivity": {
            "unit": "MS/m",
            "min_value": 0,
            "max_value": 65,
            "type": "numeric"
        },
        "wavelength": {
            "unit": "nm",
            "min_value": 0,
            "max_value": 1000,
            "type": "numeric"
        },
        "emission_wavelength": {
            "unit": "nm",
            "min_value": 0,
            "max_value": 1000,
            "type": "numeric"
        },
        "visible_emission_wavelength": {
            "unit": "nm",
            "min_value": 0,
            "max_value": 1000,
            "type": "numeric"
        },
        "ionization_wavelength": {
            "unit": "nm",
            "min_value": 0,
            "max_value": 1000,
            "type": "numeric"
        },
        "spectrum": {
            "unit": "nm",
            "min_value": 0,
            "max_value": 1000,
            "type": "numeric"
        },
        "block": {
            "unit": "",
            "min_value": None,
            "max_value": None,
            "type": "categorical"
        }
    }

    return metadata.get(property_name, {
        "unit": "",
        "min_value": 0,
        "max_value": 100,
        "type": "numeric"
    })
