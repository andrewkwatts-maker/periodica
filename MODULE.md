# periodica

> **Crate:** `periodica_core / periodica-runtime` &nbsp;|&nbsp; **Engine plugin:** `pt-periodica`
> **Upstream repo:** EXISTS (live)

Engine bridge composing periodica material datasheets onto the SDF material, plus Fourier baking, GLSL emission, crystal SDF primitives, mass integration and fracture planes.

---

## Design

The engine-side bridge composes periodica material datasheets onto the SDF material and adds engine services: Fourier-texture bake metadata, GLSL sampler emission, crystal-lattice SDF primitives, mass/inertia integration and weakpoint fracture-plane search.

**Roughly half of the plugin is upstream work in the wrong repo.** `pt_periodica_fracture.rs` (550), `pt_periodica_mass.rs` (277) and `pt_periodica_sdf.rs` (193) are closure-based, depend only on thiserror and std, and reference no engine crate. They belong in `periodica-runtime`, which does not exist yet.

Three defects to fix while transplanting the fracture/mass code: the A* heuristic is multiplied by 0.0 (so it is Dijkstra, not A*); `FaceSide` has only a `Low` variant so it only ever searches low-to-high; and the mass integrator uses a bare LCG with no stratification, needing a Halton or Sobol sampler and a real eigensolver for the principal axes.

The live repo is at v2.3.0 with a new `periodica-mat` crate (canonical SI material model). The vendored copy in the engine is v2.0.0-alpha.0 from May, and carries a `with-arithmos` feature edge that was deleted upstream - a latent build break that fires on refresh.

## Responsibilities

- PTPeriodicaMaterial: composition over the SDF material with symbolic property fields
- Loader: DashMap cache, singleton, file-watcher hot reload
- Fourier bake config/validation and result metadata
- GLSL sampler emission with identifier sanitisation
- Crystal-lattice SDF primitive (FCC/BCC, 27-cell wraparound)
- Mass/inertia integration and weakpoint fracture-plane search

### Explicitly not this module's job

- Engine material composition, `PTExpression` storage and hot reload - those stay in `pt-periodica`
- Renderer texture handles

## Dependencies

**Upstream crates**
- Arithma (optional)
- metaphysica / physica_core (optional)

**Engine plugins** *(these disappear once extraction completes)*
- pt-optikos (1 ref)
- pt-arithmos (1 ref)
- pt-themelios (1 ref)
- pt-eml-bridge DEAD - 0 refs
- pt-physica DEAD - 0 refs

**External crates**
- serde
- serde_json
- once_cell
- parking_lot
- dashmap 5.5
- anyhow
- thiserror
- tracing
- unused: ndarray, cgmath, tempfile

## Current state

| | |
|---|---|
| Rust LOC | 2,162 |
| Tests | 51 |
| GPU-coupled files | 0 |
| Extractable | 47% belongs upstream |
| PyPI | Already on PyPI |
| Extraction risk | **Medium** |

**Verdict.** Not a thin wrapper - roughly half is upstream work in the wrong repo. ~1,020 LOC of fracture, mass and crystal-SDF code belongs in periodica-runtime, which does not exist yet. Best test density of the engine plugins (23.6/kLOC).

## Blockers

- Latent build break: with-arithmos forwards to periodica_core?/with-arithmos, a feature deleted upstream in 2.1+
- Vendored submodule is v2.0.0-alpha.0 (May) vs live v2.3.0 (Aug) - periodica-mat did not exist when this was written
- periodica-runtime does not exist yet - nowhere upstream to put fracture/mass/sdf
- PTLatticeKind defined twice with different derives
- FourierFieldConfig alias collides by name with the upstream type
- from_periodica_spec_real() unconditionally bails - every material is a 1000 kg/m^3 water stub
- Unused deps: pt-eml-bridge, pt-physica, ndarray, cgmath, serde_json

## Work items

> **Status 2026-10-02** (hand-maintained: the current `_gen_report.py` reads `modules.yaml`, which classes
> pt-periodica as an engine-side bridge, so this file can no longer be regenerated with these items).
> Upstream items 2-4 are **done**. Engine items 1, 5, 6, 7 are **deferred by owner decision**: the SVN engine
> has not built since 2026-09-13 (pt-themelios still imports modules Themelios moved to Rhoe/Metron/Ergon), and
> `H:\Github\PlayTow` is now the engine; Rust consumers use `periodica-runtime` directly.

1. *Deferred* - Update the submodule to 2.3.0 and delete the dead feature edge (the forward was removed 2026-09-04; the refresh also needs the engine workspace to stop listing `periodica_core` as a member)
2. **Done** - Land periodica-runtime in the live repo (`rust/periodica-runtime`)
3. **Done** - Move fracture + mass + crystal SDF (and their 24 tests) upstream (63 unit + 2 doc tests)
4. **Done** - Fix while transplanting: A* heuristic now remaining-cells x min-cost; `FaceSide::High` added; Halton + Cranley-Patterson sampling with a single parallel-axis pass; nalgebra eigensolver for principal axes; plus the plane-offset bug (A* now crosses the section, so the offset tracks the weak band)
5. *Deferred* - Collapse the duplicate PTLatticeKind (upstream `periodica_runtime::LatticeKind` is ready to re-export); drop the `FourierFieldConfig` alias, which aliases a bake *result*
6. *Deferred* - Re-point the material at periodica_mat::record (`periodica_core::material_record()` is ready)
7. *Deferred* - Prune unused deps (ndarray, cgmath, serde_json, tempfile, the dead `with-arithmos` feature); align dashmap to 6 (periodica's workspace is already on 6.1)

---

## Naming

Greek words, written in Latin letters. No Greek script in code, docs or reports.
See [INDEX.md 1b](../GitReview/extraction/INDEX.md#1b-naming-convention----keep-the-greek).

## References

- Master plan: `H:\Github\GitReview\extraction\INDEX.md`
- Per-module report: `H:\Github\GitReview\extraction\modules.html`
- Source of truth today: `h:\DaedalusSVN\PlayTowEngine\PlayTowEngine\plugins\pt-periodica`

*Generated 2026-09-03. Do not hand-copy code into this folder; follow the procedure
in INDEX.md so the rename and test migration stay verifiable.*
