# R1b: sheet coverage census by PlayTow domain (lane `census`)

Read `_common.md` first. Start this lane only after M1 has merged into `claude/sheets`.

Deliverable:
- a rerunnable `rust/periodica-mat/examples/domain_census.rs`;
- its committed output, `docs/dev/domain-census.json` and `docs/dev/domain-census.md`.

Both go on `claude/census`.

## What it measures
The census ingests every sheet through the real `periodica-mat` ingest path, the same as
`tests/corpus.rs`. For each sheet it reports coverage per domain, with the property count and the
`Source` mix (Datasheet/Derived/Estimated/UserOverride):
- mass;
- elastic;
- failure;
- contact;
- optical and spectral;
- thermal and phase;
- electrical;
- magnetic;
- acoustic.

Take the domains from PlayTow `docs/MATERIALS.md` and the asks in `docs/plan/playtow-sheets.md`.
Define the domain → `PropertyId` map once, in one table.

It also lists:
- keys that ingest drops (`provenance.unknown_keys`) and keys whose units are missing or
  ambiguous;
- values whose meaning depends on temperature (conductivity, heat capacity, resistivity, modulus)
  but whose sheet states no temperature;
- the sheets that `census::volume_readiness` reports as not ready.

The `.md` gives the totals per category (alloys, materials, elements, and so on) and the 20 worst
gaps that PlayTow's REQ-MAT, REQ-THERMAL, REQ-FIELDS and REQ-SPECTRAL would hit first.

Add a test that the example's domain table covers every `PropertyId`, or names each one that it
excludes on purpose.
