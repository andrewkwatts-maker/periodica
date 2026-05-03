"""Generator scripts that build derived/ entries from JSON-array inputs.

Each script is a tiny entry point: it points the shared `_runner` at one
JSON input file under `inputs/`. The runner is generic - it reads
{name, spec} rows, calls Get(spec), and Save(name, result). It does not
know what kind of object it is building.

Run order matters: build_hadrons -> build_periodic_table -> build_molecules
(each layer composes against names saved in earlier layers).
"""
