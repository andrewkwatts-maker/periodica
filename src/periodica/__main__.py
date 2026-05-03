"""periodica CLI entry point.

Run as ``python -m periodica`` or, after install, just ``periodica``.

Subcommands
-----------
  build [SCRIPT...]    Run default generator scripts (or only the named ones)
                       in dependency order. Use --extra <path> to pull in
                       additional input JSONs from your own folder/file.
  run PATH             Run a single custom input JSON.
  clear [TIER]         Wipe `data/derived/<tier>/`. Omit TIER to wipe all
                       generated data.
  list [TIER]          List registry tiers and their entry counts; pass
                       a tier name to list its entries.
  show NAME            Pretty-print the JSON for a registry entry by name.
  inputs               Print the path to every default input JSON.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from periodica.get import (
    Get,
    UnknownName,
    list_tiers,
    reload_registry,
)
from periodica.sample import data_sheet, sample
from periodica.scripts._runner import (
    _INPUTS_DIR,
    build_all,
    build_from_input,
    default_inputs,
)
from periodica.folding import (
    backbone_array_from_pdb,
    build_backbone_from_entry,
    kabsch_rmsd,
    load_alphafold_reference,
    parse_pdb_backbone,
)
from periodica.optimize import optimize_alloy, optimize_protein_folding
from periodica.export import (
    export_hlsl,
    export_obj,
    export_sdf_raw,
    export_stl,
    export_vtk_legacy,
)


def _derived_dir() -> Path:
    """Resolve `data/derived/` from the package."""
    from periodica.get import _derived_dir as _dd
    return _dd()


def _tier_dir(tier: str) -> Path:
    return _derived_dir() / tier


# ── subcommands ─────────────────────────────────────────────────────────

def cmd_build(args: argparse.Namespace) -> int:
    extra: list[Path] = []
    for ex in (args.extra or []):
        extra.append(Path(ex))
    only = args.scripts or None
    results = build_all(extra_dirs=extra, only=only, verbose=not args.quiet)
    total_ok = sum(r["ok"] for r in results)
    total_err = sum(len(r["errors"]) for r in results)
    print()
    print(f"Build complete: {len(results)} script(s), {total_ok} entries saved, {total_err} error(s).")
    return 0 if total_err == 0 else 1


def cmd_run(args: argparse.Namespace) -> int:
    result = build_from_input(args.path, verbose=not args.quiet)
    return 0 if not result["errors"] else 1


def cmd_clear(args: argparse.Namespace) -> int:
    derived = _derived_dir()
    if args.tier:
        target = _tier_dir(args.tier)
        if not target.exists():
            print(f"Nothing to clear: tier {args.tier!r} has no derived data at {target}")
            return 0
        if not args.yes:
            print(f"About to delete {target} ({sum(1 for _ in target.rglob('*.json'))} JSON files).")
            ans = input("Proceed? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 1
        shutil.rmtree(target)
        print(f"Cleared {target}")
    else:
        if not derived.exists():
            print(f"Nothing to clear: {derived} does not exist.")
            return 0
        tiers = sorted(p.name for p in derived.iterdir() if p.is_dir())
        if not tiers:
            print(f"Nothing to clear: {derived} is empty.")
            return 0
        if not args.yes:
            file_count = sum(1 for _ in derived.rglob("*.json"))
            print(f"About to delete {file_count} JSON file(s) across tiers: {tiers}")
            ans = input("Proceed? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 1
        for sub in derived.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)
        print(f"Cleared {derived}")
    reload_registry()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    reload_registry()
    if args.tier:
        target = _tier_dir(args.tier)
        if not target.exists():
            print(f"Tier {args.tier!r} has no entries (folder {target} not found).")
            return 1
        entries = sorted(p.stem for p in target.glob("*.json"))
        print(f"Tier {args.tier!r} ({len(entries)} entries):")
        for n in entries:
            print(f"  {n}")
        return 0
    tiers = list_tiers()
    print(f"Registry tiers ({len(tiers)}):")
    derived = _derived_dir()
    for t in tiers:
        d = _tier_dir(t)
        if d.is_dir():
            count = sum(1 for _ in d.glob("*.json"))
            origin = "derived"
        else:
            from periodica.get import _DATA_DIR, _tier_sources
            count = 0
            for tier_name, src, _is_active in _tier_sources():
                if tier_name == t and src.is_dir():
                    count = sum(1 for _ in src.glob("*.json"))
                    break
            origin = "active"
        print(f"  {t:20s} {origin:10s}  {count:5d} entries")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    reload_registry()
    try:
        entry = Get(args.name)
    except UnknownName:
        print(f"Unknown name: {args.name!r}", file=sys.stderr)
        return 1
    print(json.dumps(entry, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_point(s: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--at must be three comma-separated numbers (got {s!r})"
        )
    return tuple(float(p) for p in parts)


def cmd_sample(args: argparse.Namespace) -> int:
    reload_registry()
    try:
        if args.prop:
            value = sample(args.name, args.prop, at=args.at, scale_m=args.scale)
            if value is None:
                print(
                    f"Property {args.prop!r} not found on {args.name!r}.",
                    file=sys.stderr,
                )
                return 1
            print(value)
        else:
            sheet = data_sheet(args.name)
            if not sheet:
                print(f"No properties on {args.name!r}.", file=sys.stderr)
                return 1
            print(json.dumps(sheet, indent=2, ensure_ascii=False, sort_keys=True))
    except UnknownName:
        print(f"Unknown name: {args.name!r}", file=sys.stderr)
        return 1
    return 0


def cmd_inputs(args: argparse.Namespace) -> int:
    inputs = default_inputs()
    print(f"Default inputs directory: {_INPUTS_DIR}")
    for p in inputs:
        print(f"  {p.name}")
    return 0


def _coords_to_pdb(coords, sequence: str) -> str:
    """Render an (n,3,3) backbone array to a minimal PDB ATOM block."""
    lines = []
    serial = 1
    atom_names = ("N", "CA", "C")
    for i in range(coords.shape[0]):
        aa = sequence[i] if i < len(sequence) else "X"
        # Map 1-letter to 3-letter
        three = {
            "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN",
            "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
            "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
        }.get(aa, "UNK")
        for a, atom in enumerate(atom_names):
            x, y, z = coords[i, a]
            lines.append(
                f"ATOM  {serial:5d}  {atom:<3s} {three} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           "
                f"{atom[0]}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def cmd_fold(args: argparse.Namespace) -> int:
    reload_registry()
    try:
        coords = build_backbone_from_entry(args.name)
    except UnknownName:
        print(f"Unknown protein: {args.name!r}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Folding failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    entry = Get(args.name)
    sequence = entry.get("sequence") or "".join(
        r.get("residue", "X") for r in entry.get("residues") or []
    )

    if args.validate or args.uniprot:
        uid = args.uniprot
        if not uid:
            # Try common name -> UniProt mapping for the bundled set.
            uid = {
                "Crambin":         "P01542",
                "Insulin_Chain_A": "P01308",
                "Insulin":         "P01308",
                "Ubiquitin":       "P0CG48",
                "Lysozyme":        "P00698",
                "Beta_Defensin":   "P60022",
            }.get(args.name)
        if not uid:
            print(
                f"--validate requires --uniprot ID for {args.name!r} "
                "(no built-in mapping).",
                file=sys.stderr,
            )
            return 1
        try:
            ref_pdb = load_alphafold_reference(uid, allow_fetch=not args.offline)
        except Exception as e:
            print(f"Could not load AF2 reference: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        af_coords = backbone_array_from_pdb(parse_pdb_backbone(ref_pdb))
        n = min(coords.shape[0], af_coords.shape[0])
        if n < 3:
            print("Too few matching residues to align.", file=sys.stderr)
            return 1
        _, rmsd = kabsch_rmsd(coords[:n, 1, :], af_coords[:n, 1, :])
        print(f"{args.name} vs AlphaFold ({uid}): {n} residues, CA RMSD = {rmsd:.3f} A")
        return 0

    pdb_text = _coords_to_pdb(coords, sequence)
    if args.output:
        Path(args.output).write_text(pdb_text, encoding="utf-8")
        print(f"Wrote {coords.shape[0]} residues -> {args.output}")
    else:
        print(pdb_text)
    return 0


def _parse_kv_target(s: str) -> tuple[str, float]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            f"--target expects KEY=VALUE (e.g. YieldStrength_MPa_min=800), got {s!r}"
        )
    k, v = s.split("=", 1)
    return k.strip(), float(v.strip())


def cmd_optimize(args: argparse.Namespace) -> int:
    if args.subject == "folding":
        if not args.sequence:
            print("`optimize folding` requires --sequence", file=sys.stderr)
            return 1
        result = optimize_protein_folding(
            args.sequence,
            target=args.target_region or "helical",
            iterations=args.iterations,
            seed=args.seed,
        )
        print(f"Final energy: {result['final_energy']:.3f}")
        print(f"Target region: {result['target_region']}")
        print(f"phi/psi (first 5): {result['phi_psi'][:5].tolist()}")
        return 0
    elif args.subject == "alloy":
        if not args.target:
            print("`optimize alloy` requires at least one --target K=V", file=sys.stderr)
            return 1
        targets = dict(args.target)
        results = optimize_alloy(
            targets,
            base=args.base,
            candidates=args.candidates,
            top_k=args.top,
            seed=args.seed,
        )
        if not results:
            print("No candidate satisfied all constraints.")
            return 1
        for i, r in enumerate(results, 1):
            comp_str = ", ".join(f"{k}={v:.4f}" for k, v in r["composition"].items())
            print(f"#{i}  score={r['score']:.2f}  composition: {comp_str}")
            for k in sorted(r["estimated_properties"]):
                print(f"     {k}: {r['estimated_properties'][k]:.2f}")
        return 0
    else:
        print(f"Unknown optimize subject: {args.subject!r}", file=sys.stderr)
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Alias: `validate <name>` == `fold <name> --validate`."""
    args.validate = True
    args.uniprot = args.uniprot
    args.output = None
    args.offline = args.offline
    return cmd_fold(args)


def _parse_bounds_arg(s: str) -> tuple:
    """Parse 'X1,Y1,Z1,X2,Y2,Z2' into ((X1,Y1,Z1),(X2,Y2,Z2))."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"--bounds expects 6 comma-separated numbers, got {s!r}"
        )
    nums = [float(p) for p in parts]
    return ((nums[0], nums[1], nums[2]), (nums[3], nums[4], nums[5]))


def cmd_export(args: argparse.Namespace) -> int:
    reload_registry()
    fmt = args.format.lower()
    needs_grid = fmt in ("stl", "obj", "vtk", "sdf")
    bounds = args.bounds
    if needs_grid and bounds is None:
        # Default: 10×10×10 voxels at the field's macro length scale (or 1 unit).
        bounds = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    voxel_size = args.voxel_size or 1.0
    scale_m = args.scale

    try:
        if fmt == "stl":
            out = export_stl(
                args.name, args.output,
                bounds=bounds, voxel_size=voxel_size, scale_m=scale_m,
                binary=not args.ascii,
            )
        elif fmt == "obj":
            out, mtl = export_obj(
                args.name, args.output,
                bounds=bounds, voxel_size=voxel_size, scale_m=scale_m,
                properties=args.properties,
            )
            print(f"OBJ:  {out}")
            print(f"MTL:  {mtl}")
            return 0
        elif fmt == "vtk":
            out = export_vtk_legacy(
                args.name, args.output,
                bounds=bounds, voxel_size=voxel_size, scale_m=scale_m,
                properties=args.properties or ["Density_kgm3", "YoungsModulus_GPa"],
            )
        elif fmt == "sdf":
            out = export_sdf_raw(
                args.name, args.output,
                bounds=bounds, voxel_size=voxel_size, scale_m=scale_m,
                mode=args.sdf_mode,
            )
            print(f"SDF:    {out}")
            print(f"Header: {Path(str(out) + '.json')}")
            return 0
        elif fmt == "hlsl":
            out = export_hlsl(args.name, args.output, properties=args.properties)
        else:
            print(f"Unknown format {fmt!r}; valid: stl, obj, vtk, sdf, hlsl", file=sys.stderr)
            return 1
    except UnknownName:
        print(f"Unknown name: {args.name!r}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Export failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {out}")
    return 0


# ── parser ──────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="periodica",
        description="Periodica CLI: build/clear/run/list generated data tiers.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("build", help="Run default generator scripts.")
    sb.add_argument(
        "scripts",
        nargs="*",
        help="Optional: only run these input names (e.g. 'molecules' or 'isotopes.json'). "
             "If omitted, all defaults run in dependency order.",
    )
    sb.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="PATH",
        help="Extra input file or directory to include. Repeat for multiple.",
    )
    sb.add_argument("-q", "--quiet", action="store_true", help="Less verbose output.")
    sb.set_defaults(func=cmd_build)

    sr = sub.add_parser("run", help="Run a single custom input JSON.")
    sr.add_argument("path", help="Path to a custom input JSON.")
    sr.add_argument("-q", "--quiet", action="store_true")
    sr.set_defaults(func=cmd_run)

    sc = sub.add_parser("clear", help="Wipe generated derived/ data.")
    sc.add_argument("tier", nargs="?", help="Tier name to clear; omit for all.")
    sc.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt.")
    sc.set_defaults(func=cmd_clear)

    sl = sub.add_parser("list", help="List registry tiers / entries.")
    sl.add_argument("tier", nargs="?", help="Tier to expand; omit for summary.")
    sl.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="Pretty-print a registry entry by name.")
    sh.add_argument("name", help="Entry name (e.g. 'H', 'H2O', 'Proton').")
    sh.set_defaults(func=cmd_show)

    si = sub.add_parser("inputs", help="List the default input JSON files shipped with the package.")
    si.set_defaults(func=cmd_inputs)

    ss = sub.add_parser(
        "sample",
        help="Sample a property of an entry at a 3D point (homogeneous entries return bulk).",
    )
    ss.add_argument("name", help="Registry entry name (e.g. 'Steel-1018', 'CFRP', 'H2O').")
    ss.add_argument(
        "prop",
        nargs="?",
        help="Property name (e.g. 'Density_kgm3'). Omit to print the full data sheet.",
    )
    ss.add_argument(
        "--at",
        type=_parse_point,
        default=None,
        metavar="X,Y,Z",
        help="3D sample point (comma-separated, e.g. 0,0,0).",
    )
    ss.add_argument(
        "--scale",
        type=float,
        default=None,
        metavar="METRES",
        help="Characteristic length scale of the sample (for scale-dependent fields).",
    )
    ss.set_defaults(func=cmd_sample)

    sf = sub.add_parser(
        "fold",
        help="Build a 3D backbone for a registered protein from its phi/psi.",
    )
    sf.add_argument("name", help="Protein name (e.g. 'Crambin', 'Ubiquitin').")
    sf.add_argument(
        "--output",
        metavar="PATH",
        help="Write the backbone as a PDB file. Default: print to stdout.",
    )
    sf.add_argument(
        "--validate",
        action="store_true",
        help="Compare to AlphaFold reference and print Kabsch-aligned RMSD.",
    )
    sf.add_argument(
        "--uniprot",
        metavar="ID",
        help="UniProt ID for the AlphaFold reference (auto-resolved for built-in proteins).",
    )
    sf.add_argument(
        "--offline",
        action="store_true",
        help="Do not download AF2 from the network; require a bundled or cached copy.",
    )
    sf.set_defaults(func=cmd_fold)

    sv = sub.add_parser("validate", help="Alias for `fold <name> --validate`.")
    sv.add_argument("name")
    sv.add_argument("--uniprot", metavar="ID")
    sv.add_argument("--offline", action="store_true")
    sv.set_defaults(func=cmd_validate)

    so = sub.add_parser("optimize", help="Search a folding or alloy parameter space.")
    so_sub = so.add_subparsers(dest="subject", required=True)

    sof = so_sub.add_parser("folding", help="Simulated-annealing on phi/psi for a sequence.")
    sof.add_argument("--sequence", required=True, help="One-letter amino acid sequence.")
    sof.add_argument(
        "--target-region",
        dest="target_region",
        default="helical",
        help="Target region name or alias (default: helical).",
    )
    sof.add_argument("--iterations", type=int, default=2000)
    sof.add_argument("--seed", type=int, default=None)
    sof.set_defaults(func=cmd_optimize)

    soa = so_sub.add_parser("alloy", help="Search compositions for property targets.")
    soa.add_argument(
        "--target",
        action="append",
        type=_parse_kv_target,
        default=[],
        help="KEY=VALUE constraint, e.g. YieldStrength_MPa_min=800. Repeatable.",
    )
    soa.add_argument("--base", default="Fe", help="Base element symbol (default Fe).")
    soa.add_argument("--candidates", type=int, default=500)
    soa.add_argument("--top", type=int, default=5)
    soa.add_argument("--seed", type=int, default=None)
    soa.set_defaults(func=cmd_optimize)

    se = sub.add_parser(
        "export",
        help="Export an entry's microstructure / phase field as STL / OBJ / VTK / SDF / HLSL.",
    )
    se.add_argument("name", help="Registry entry name (e.g. 'Steel-1018', 'CFRP').")
    se.add_argument(
        "--format", required=True,
        choices=["stl", "obj", "vtk", "sdf", "hlsl"],
        help="Output format.",
    )
    se.add_argument("--output", required=True, metavar="PATH",
                    help="Output file path (extension is your choice).")
    se.add_argument(
        "--bounds", type=_parse_bounds_arg, default=None,
        metavar="X1,Y1,Z1,X2,Y2,Z2",
        help="Sample volume bounds in metres. Default: ((0,0,0),(10,10,10)).",
    )
    se.add_argument(
        "--voxel-size", type=float, default=1.0, metavar="METRES", dest="voxel_size",
        help="Voxel side length in metres. Default 1.0.",
    )
    se.add_argument(
        "--scale", type=float, default=None, metavar="METRES",
        help="Sampling scale_m. Defaults to voxel_size.",
    )
    se.add_argument(
        "--properties", action="append", default=None, metavar="PROP",
        help="Property name to include (repeatable). VTK/HLSL/OBJ use this list.",
    )
    se.add_argument(
        "--ascii", action="store_true",
        help="STL only: emit ASCII STL instead of binary.",
    )
    se.add_argument(
        "--sdf-mode", choices=["occupancy", "phase"], default="occupancy",
        dest="sdf_mode",
        help="SDF only: 'occupancy' (1.0/0.0) or 'phase' (float-cast phase id).",
    )
    se.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
