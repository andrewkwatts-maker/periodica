"""3D backbone construction + structural validation for proteins.

All bond lengths, angles, and Ramachandran allowed regions live in
`data/config/folding_rules.json` -- this module reads them, never
hardcodes physics. Pure-numpy: no scipy or biopython dependency.

Public API
----------
- `build_backbone(sequence, phi_psi)`         NeRF construction (N, CA, C per residue)
- `build_backbone_from_entry(name)`           pulls phi/psi from `Get(name)` and folds
- `extract_phi_psi(entry)`                    helper: read residues[].phi/.psi
- `kabsch_rmsd(P, Q)`                         optimal-rotation RMSD between two coord sets
- `parse_pdb_backbone(path)`                  pure-Python PDB ATOM parser; returns N/CA/C
- `fetch_alphafold(uniprot_id)`               download AF2 PDB to user cache dir
- `load_alphafold_reference(uniprot_id)`      bundled-first, fallback to fetch
- `ramachandran_region(phi, psi)`             classify dihedral into a region name (or None)
- `folding_rules()`                           return parsed config dict
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlopen, urlretrieve

import numpy as np

from periodica.get import Get


_DATA_DIR = Path(__file__).parent / "data"
_CONFIG_PATH = _DATA_DIR / "config" / "folding_rules.json"
_BUNDLED_AF_DIR = _DATA_DIR / "reference" / "proteins_alphafold"


def folding_rules() -> dict:
    """Read and return the folding rules config (pure JSON, no // comments)."""
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────
# NeRF backbone construction
# ─────────────────────────────────────────────────────────────────────────

def _place_next(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                bond_len: float, bond_angle_deg: float, dihedral_deg: float) -> np.ndarray:
    """Place atom d given previous three atoms a, b, c using NeRF.

    Bond geometry: |c-d| = bond_len, angle b-c-d = bond_angle_deg,
    dihedral a-b-c-d = dihedral_deg.
    """
    bc = c - b
    bc_n = np.linalg.norm(bc)
    if bc_n == 0:
        bc_n = 1e-12
    bc_hat = bc / bc_n

    ab = b - a
    n_raw = np.cross(ab, bc_hat)
    n_norm = np.linalg.norm(n_raw)
    if n_norm == 0:
        # Collinear — pick an arbitrary perpendicular.
        n_raw = np.cross(bc_hat, np.array([1.0, 0.0, 0.0]))
        n_norm = np.linalg.norm(n_raw)
        if n_norm == 0:
            n_raw = np.cross(bc_hat, np.array([0.0, 1.0, 0.0]))
            n_norm = np.linalg.norm(n_raw)
    n_hat = n_raw / n_norm
    m_hat = np.cross(n_hat, bc_hat)

    # Internal bond angle (b-c-d). For angle theta: vec(c->d) projects -cos(theta)
    # along bc_hat (so c->d swings outward from b), with sin(theta) in the
    # perpendicular plane spanned by m_hat (in-plane) and n_hat (out-of-plane).
    theta = np.deg2rad(bond_angle_deg)
    phi = np.deg2rad(dihedral_deg)
    offset = bond_len * np.array([
        -np.cos(theta),
         np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
    ])
    M = np.column_stack([bc_hat, m_hat, n_hat])
    return c + M @ offset


def build_backbone(
    sequence: str,
    phi_psi: Sequence[Tuple[float, float]],
    *,
    rules: Optional[dict] = None,
) -> np.ndarray:
    """Construct backbone (N, CA, C) coords from a sequence and phi/psi list.

    Returns ndarray of shape (n_residues, 3, 3): [residue][atom in {N, CA, C}][xyz].
    Bond lengths and angles come from `folding_rules()`; omega defaults to 180.
    """
    rules = rules or folding_rules()
    bl = rules["bond_lengths_A"]
    ba = rules["bond_angles_deg"]
    omega = rules.get("omega_deg", 180.0)

    n = len(sequence)
    if n == 0:
        return np.zeros((0, 3, 3))
    if len(phi_psi) != n:
        raise ValueError(
            f"phi_psi length {len(phi_psi)} != sequence length {n}"
        )

    coords = np.zeros((n, 3, 3))

    # Place first residue: N at origin, CA along +x, C in +xy plane.
    # Internal angle N-CA-C between vectors CA->N (-x) and CA->C must equal ba["N_CA_C"].
    coords[0, 0] = np.array([0.0, 0.0, 0.0])
    coords[0, 1] = np.array([bl["N_CA"], 0.0, 0.0])
    theta = np.deg2rad(ba["N_CA_C"])
    coords[0, 2] = coords[0, 1] + bl["CA_C"] * np.array([-np.cos(theta), np.sin(theta), 0.0])

    for i in range(1, n):
        N_prev, CA_prev, C_prev = coords[i - 1, 0], coords[i - 1, 1], coords[i - 1, 2]
        psi_prev = float(phi_psi[i - 1][1])
        phi_curr, psi_curr = float(phi_psi[i][0]), float(phi_psi[i][1])

        # N_i: dihedral CA_{i-1} - C_{i-1} - N_i is psi_{i-1} (rotates around C-N forming bond).
        # Actually the dihedral N_{i-1}-CA_{i-1}-C_{i-1}-N_i is psi_{i-1}.
        N_i = _place_next(
            a=N_prev, b=CA_prev, c=C_prev,
            bond_len=bl["C_N"], bond_angle_deg=ba["CA_C_N"],
            dihedral_deg=psi_prev,
        )
        # CA_i: dihedral CA_{i-1} - C_{i-1} - N_i - CA_i is omega.
        CA_i = _place_next(
            a=CA_prev, b=C_prev, c=N_i,
            bond_len=bl["N_CA"], bond_angle_deg=ba["C_N_CA"],
            dihedral_deg=omega,
        )
        # C_i: dihedral C_{i-1} - N_i - CA_i - C_i is phi_i.
        C_i = _place_next(
            a=C_prev, b=N_i, c=CA_i,
            bond_len=bl["CA_C"], bond_angle_deg=ba["N_CA_C"],
            dihedral_deg=phi_curr,
        )
        coords[i, 0] = N_i
        coords[i, 1] = CA_i
        coords[i, 2] = C_i

    return coords


def extract_phi_psi(entry: dict) -> List[Tuple[float, float]]:
    """Read per-residue phi/psi from an `Get('Crambin')`-style entry's residues array."""
    residues = entry.get("residues") or entry.get("Residues")
    if not residues:
        raise ValueError("entry has no residues array (need per-residue phi/psi)")
    out: List[Tuple[float, float]] = []
    for r in residues:
        phi = r.get("phi", 0.0) if r.get("phi") is not None else 0.0
        psi = r.get("psi", 0.0) if r.get("psi") is not None else 0.0
        out.append((float(phi), float(psi)))
    return out


def build_backbone_from_entry(name: str) -> np.ndarray:
    """Look up a protein by name in the registry and fold it from phi/psi."""
    entry = Get(name)
    seq = entry.get("sequence") or entry.get("Sequence") or ""
    if not seq:
        # Fallback: derive sequence from residues array.
        residues = entry.get("residues") or entry.get("Residues") or []
        seq = "".join(r.get("residue", "X") for r in residues)
    phi_psi = extract_phi_psi(entry)
    if len(seq) != len(phi_psi):
        # Truncate to shorter to be lenient with imperfect data.
        n = min(len(seq), len(phi_psi))
        seq = seq[:n]
        phi_psi = phi_psi[:n]
    return build_backbone(seq, phi_psi)


# ─────────────────────────────────────────────────────────────────────────
# Kabsch / RMSD
# ─────────────────────────────────────────────────────────────────────────

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (Q_aligned_to_P, rmsd). Both P and Q are (N, 3) arrays.

    Standard Kabsch via SVD; mirror-corrected so the rotation is proper.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"kabsch_rmsd needs matching (N, 3) inputs, got {P.shape} vs {Q.shape}")

    Pc_mean = P.mean(0)
    Qc_mean = Q.mean(0)
    Pc = P - Pc_mean
    Qc = Q - Qc_mean

    # Standard Kabsch: H = P^T Q (covariance), SVD H = U S V^T,
    # optimal rotation R = V diag(1, 1, sign(det(V U^T))) U^T,
    # then aligned Q' = Q R approximates P.
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    if d == 0:
        d = 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    Qa = Qc @ R
    rmsd = float(np.sqrt(((Pc - Qa) ** 2).sum(1).mean()))
    return Qa + Pc_mean, rmsd


# ─────────────────────────────────────────────────────────────────────────
# PDB parser (CA-focused, minimal)
# ─────────────────────────────────────────────────────────────────────────

_BACKBONE_ATOMS = ("N", "CA", "C")


def parse_pdb_backbone(path) -> Dict[int, Dict[str, np.ndarray]]:
    """Read an ATOM-record PDB file and return per-residue backbone coords.

    Returns: dict[residue_index, dict[atom_name, np.ndarray(3,)]] for atoms
    in {N, CA, C}. Skips alt-locs other than the first occurrence per residue.
    """
    out: Dict[int, Dict[str, np.ndarray]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                atom_name = line[12:16].strip()
                resi = int(line[22:26])
                if atom_name not in _BACKBONE_ATOMS:
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            entry = out.setdefault(resi, {})
            if atom_name in entry:
                # Already have it (alt-loc); keep the first.
                continue
            entry[atom_name] = np.array([x, y, z])
    return out


def backbone_array_from_pdb(parsed: Dict[int, Dict[str, np.ndarray]]) -> np.ndarray:
    """Convert parse_pdb_backbone() output to ndarray of shape (n_residues, 3, 3).

    Skips any residue missing one of N/CA/C.
    """
    rows: List[np.ndarray] = []
    for resi in sorted(parsed):
        atoms = parsed[resi]
        if not all(a in atoms for a in _BACKBONE_ATOMS):
            continue
        rows.append(np.stack([atoms["N"], atoms["CA"], atoms["C"]]))
    if not rows:
        return np.zeros((0, 3, 3))
    return np.stack(rows)


# ─────────────────────────────────────────────────────────────────────────
# AlphaFold reference (bundle-first, network fallback)
# ─────────────────────────────────────────────────────────────────────────

def _default_cache_dir() -> Path:
    home = Path(os.environ.get("PERIODICA_CACHE", "")).expanduser()
    if str(home):
        return home / "alphafold"
    return Path.home() / ".cache" / "periodica" / "alphafold"


def _find_existing_pdb(directory: Path, uniprot_id: str) -> Optional[Path]:
    """Return the highest-version AF2 PDB in `directory` for a UniProt ID, if any."""
    if not directory.is_dir():
        return None
    rules = folding_rules()
    pattern = rules["alphafold_pdb_glob"].format(uniprot_id=uniprot_id)
    matches = sorted(directory.glob(pattern))
    matches = [p for p in matches if p.stat().st_size > 0]
    return matches[-1] if matches else None


def fetch_alphafold(uniprot_id: str, *, cache_dir: Optional[Path] = None) -> Path:
    """Download an AF2 PDB for a UniProt ID; returns the cached path.

    Uses the AlphaFold prediction API to discover the current model version
    URL, then urlretrieve to fetch. Pure stdlib. Cached on disk.
    """
    cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    existing = _find_existing_pdb(cache, uniprot_id)
    if existing is not None:
        return existing
    rules = folding_rules()
    api_url = rules["alphafold_api_template"].format(uniprot_id=uniprot_id)
    with urlopen(api_url, timeout=30) as resp:
        meta = json.loads(resp.read())[0]
    pdb_url = meta["pdbUrl"]
    target = cache / pdb_url.rsplit("/", 1)[-1]
    urlretrieve(pdb_url, target)
    return target


def load_alphafold_reference(
    uniprot_id: str,
    *,
    allow_fetch: bool = True,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Resolve an AF2 reference PDB: bundled first, then user cache, then network.

    Set `allow_fetch=False` to disable the network fallback (CI use).
    """
    bundled = _find_existing_pdb(_BUNDLED_AF_DIR, uniprot_id)
    if bundled is not None:
        return bundled
    cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cached = _find_existing_pdb(cache, uniprot_id)
    if cached is not None:
        return cached
    if allow_fetch:
        return fetch_alphafold(uniprot_id, cache_dir=cache)
    raise FileNotFoundError(
        f"No AlphaFold reference for {uniprot_id} (bundled dir {_BUNDLED_AF_DIR}, cache {cache}); "
        f"call with allow_fetch=True to download."
    )


# ─────────────────────────────────────────────────────────────────────────
# Ramachandran classification
# ─────────────────────────────────────────────────────────────────────────

def ramachandran_region(phi: float, psi: float, *, rules: Optional[dict] = None) -> Optional[str]:
    """Classify a (phi, psi) pair into a named region from folding_rules.json.

    Returns the region name, or None if outside all declared allowed regions.
    """
    rules = rules or folding_rules()
    regions = rules.get("ramachandran_allowed", {})
    for name, r in regions.items():
        phi_lo, phi_hi = r["phi"]
        psi_lo, psi_hi = r["psi"]
        if phi_lo <= phi <= phi_hi and psi_lo <= psi <= psi_hi:
            return name
    return None


def ramachandran_in_allowed(phi: float, psi: float, *, rules: Optional[dict] = None) -> bool:
    return ramachandran_region(phi, psi, rules=rules) is not None
