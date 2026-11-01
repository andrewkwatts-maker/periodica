"""Generate the scipy cross-check fixture for periodica-qm complex Y_lm.

Run once with the global interpreter (scipy >= 1.15 for sph_harm_y):
    python gen_sph_harm_y_scipy.py sph_harm_y_scipy.json
"""
import json
import sys

import numpy as np
import scipy
from scipy.special import sph_harm_y

ANGLES = [
    (0.0, 0.0),
    (1e-3, 2.5),
    (0.3, 0.1),
    (0.7853981633974483, 1.0471975511965976),
    (1.2, -2.0),
    (1.5707963267948966, 0.0),
    (2.0, 4.0),
    (2.9, 5.9),
    (3.141592653589793, 1.0),
]
L_MAX = 8

cases = []
for l in range(L_MAX + 1):
    for m in range(-l, l + 1):
        for theta, phi in ANGLES:
            y = complex(sph_harm_y(l, m, theta, phi))
            cases.append({"l": l, "m": m, "theta": theta, "phi": phi,
                          "re": y.real, "im": y.imag})

doc = {
    "generator": f"scipy.special.sph_harm_y (scipy {scipy.__version__}, numpy {np.__version__})",
    "convention": "sph_harm_y(n=l, m, theta=polar, phi=azimuth); orthonormal on the sphere; Condon-Shortley phase included",
    "cases": cases,
}
with open(sys.argv[1], "w", newline="\n") as fh:
    json.dump(doc, fh, indent=1)
    fh.write("\n")
print(len(cases), "cases")
