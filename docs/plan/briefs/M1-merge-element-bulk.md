# M1: land the CRC element bulk properties on current main (lane `merge`)

Read `_common.md` first. Deliverable: commit d78bb77 ("Add measured element bulk properties (CRC
97th ed.) to the element sheets") lands on `claude/merge`, with none of origin's later changes
lost.

## Known
- d78bb77 exists only on the main checkout's local `main`, and was never pushed. Cherry-pick it by
  hash; the objects are shared across worktrees.
- `git merge-tree` reports conflicts in `rust/periodica-mat/src/canon.rs` and in about 20 element
  sheets in `src/periodica/data/active/elements/`: Mg, Al, Ti, V, Cr, Fe, Co, Ni, Cu, Zn, Zr, Nb,
  Mo, Ag, Cd, Sn, Au, Pb and more.
- Commit 33a84fb (CLAUDE.md shrink) is also unpushed. Compare it with the current CLAUDE.md and
  docs/claude-context.md. Cherry-pick it only if its content isn't already there.

## Steps
1. Run `git cherry-pick d78bb77` and list the conflicted files.
2. For each sheet, keep every key from both sides. Where both sides give the same key:
   - keep the value that carries a citation;
   - if both are cited and the values differ, keep origin's value and list the pair (key, both
     values, both sources) in your report.

   Never invent a value or a source.
3. For `canon.rs`, take the union of both sides' alias and unit tables. Check that no alias maps to
   two different properties.
4. Validate every touched sheet as JSON. Then run `cargo test -p periodica-mat` and the Python
   tests (`pytest tests -q -m "not slow"`).
5. Add one oracle test that ingests Fe and checks that the density is 7874 kg/m³ with
   `Source::Datasheet`. Add a second for one other CRC value from the commit, unless such tests
   already exist.
6. Commit. The message names d78bb77 and the conflict policy.
