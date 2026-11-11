# F2d: a one-part mix returns the part's stored values; E and nu are the primary pair; a measure of stored elastic mismatch

Read `_common.md` first, including "Real quantities only". One deliverable: the three changes below in `periodica-mat`, their docs
and tests. No datasheet changes. Use `-p periodica-mat` for every cargo command.

## Context
PlayTow (the engine that reads periodica) reported this on 2026-11-11. Since 1f8c5fa `mix_tagged`
(`rust/periodica-mat/src/mix.rs`) takes E and nu from the Hill K and G. `kg_of` prefers a part's
stored K and G over its E and nu. `derived/alloys/Steel-1018.json` stores E 200 GPa, nu 0.29 and
G 79.3 GPa; those disagree by 2.3 % (`E / (2 (1 + nu))` is 77.5 GPa). K is derived from E and nu
at load, so a mix of that one sheet at fraction 1 returns E 203.9 GPa and nu 0.286, not 200 and
0.29. PlayTow reads E and nu as the primary pair; it reads neither G nor K.

## 1. A mix of one part is the part (`mix_tagged`)
After the fractions are checked: when every entry with a positive fraction is the same record
(the same reference, or records that compare equal with `==`), return that record's stored value
and its own source tag, `(bulk.get(prop), bulk.source(prop))`. This holds for every `PropertyId`,
including one with no mixing rule. A property the record lacks is `MixError::Missing`. `mix` goes
through `mix_tagged`, so it follows. Leave `bounds` alone. Put the test for "one record" in a
small helper; keep every function at 60 lines or fewer.

## 2. E and nu are the primary pair (`kg_of`)
Without this, the mix jumps: one part returns E 200 GPa, and the same part at fraction 0.999999
returns 203.9. New order in `kg_of`:
1. E and nu both present and `-1 < nu < 0.5`: K and G from E and nu (the isotropic identities).
2. Otherwise stored K and G, when both are present.
3. Otherwise `None`.
No new constant, and no change to `derive.rs`. Leave `k_of` and the other rules alone. Update the
module doc of `mix.rs` (the Young modulus row and the `mix_tagged` paragraph) to state both rules.

## 3. A measure of how far stored elastic constants disagree
An isotropic solid has two independent elastic constants. A sheet that states three can
contradict itself. New module `rust/periodica-mat/src/consistency.rs`, exported from `lib.rs`:
- `pub struct ElasticMismatch { pub shear: Option<f64>, pub bulk: Option<f64> }`
- `pub fn elastic_mismatch(t: &PropertyTable) -> ElasticMismatch`
  - `shear`: `|G - E / (2 (1 + nu))| / G` when E, nu and G are present and `-1 < nu < 0.5`.
  - `bulk`: `|K - E / (3 (1 - 2 nu))| / K` when E, nu and K are present and `-1 < nu < 0.5`.
  - `None` when an input is absent or nu is out of range.
It returns the relative difference and judges nothing: no tolerance constant. Call it on the
table as ingested, before derivation, so only constants the sheet itself states are compared
(a value derived at load agrees exactly).

Corpus record: `rust/periodica-mat/tests/oracles/elastic-mismatch.txt`, one line for every sheet
where `shear` or `bulk` is `Some`: the sheet's path relative to `src/periodica/data` with forward
slashes, then the two values in percent to two decimals (`-` for `None`). Sorted by path. A test
ingests every sheet the way `tests/ingest_census.rs` or `tests/corpus.rs` already walks the
corpus, builds the same lines, and compares the two sets of lines (a set comparison, so
directory order doesn't matter). On a difference the failure message prints the computed lines,
ready to paste. A new sheet that states a third constant then fails until its line is added, and
the reviewer sees its mismatch. Don't fix any sheet: which constant is right needs a cited
source.

## Tests: `rust/periodica-mat/tests/f2d_mix_identity.rs`
Load Steel-1018 by its path, `src/periodica/data/derived/alloys/Steel-1018.json`, with the ingest
the existing tests use.
- One part at fraction 1: for every property in `bulk.iter()`, `mix_tagged` returns the stored
  value bit for bit and the stored source. E is exactly 200e9, nu 0.29, G 79.3e9.
- The same record twice at 0.5 and 0.5, and the record with a clone at 0.3 and 0.7: the same.
- A property the record lacks: `MixError::Missing`. A property with no mixing rule: the stored
  value for one part, `NotMixable` for two different parts.
- Continuity: steel at `1 - 1e-6` with an in-memory aluminium (E 69e9, nu 0.33, density 2700) at
  `1e-6`: E within 1e-5 relative of 200e9, nu within 1e-5 of 0.29.
- Two different parts: E is still `Estimated`, density still `Derived`.
- A part with no nu: two in-memory records with different names and the same values (K 160e9,
  G 79e9, density 7850, no E and no nu) at 0.5 and 0.5 give `E = 9KG / (3K + G)`, computed in
  the test, within 1e-12 relative.
- `elastic_mismatch`: Steel-1018 `shear` equals `|G - E / (2 (1 + nu))| / G` computed in the test
  (about 0.0225); a consistent in-memory record is below 1e-12; no G gives `shear: None`; nu 0.5
  gives both `None`.
- The corpus record test above.

## Report (20 lines or fewer)
The commit hash; test counts; every file touched (`git diff --name-only`); the number of sheets
in `elastic-mismatch.txt` and the five largest mismatches of each kind; every number you added
that is not a measured value or a unit conversion; every existing test whose expected
numbers moved, with the reason (`tests/f3_mix.rs` is the likely one; change an expectation only
when rule 2 explains the move); which pair `elastic` uses to derive K for a sheet that states E,
nu and G (read it, don't guess); anything above you found wrong or ambiguous (say so; don't
change a rule on your own).
