# F4: optics models: gold and validity ranges (lane `optics`)

Read `_common.md` first. Source: `docs/plan/sheets-formula-review.md`, BUGs 3 and 5, U10 and U12.
Deliverable: two commits on `claude/optics`, with tests in a new file
`rust/periodica-mat/tests/f4_optics.rs`.

1. Gold (optics/drude_lorentz.rs:32-35, optics/dispersion.rs:55-66): the model is Drude-only with
   `epsilon_inf = 1`, so gold renders as silver. Add the interband Lorentz terms from a published
   fit and cite it in the code (for example Rakic et al., Appl. Opt. 37, 5271 (1998), or Etchegoin
   et al., J. Chem. Phys. 125, 164705 (2006)). Check your parameters against Johnson and Christy
   (1972) reference points before you commit; don't trust parameters from memory. Stop
   `EpsilonInfinity` defaulting silently to 1.0: an absent value is an error or an explicit,
   documented choice. Tests: gold R(450 nm) < 0.6, R(600 nm) > 0.9, and `base_color` r/b > 1.5.
   Check the other built-in metals the same way and report any that are flat.
2. Validity ranges (optics/sellmeier.rs:24-34, optics/dispersion.rs:75-85, optics/cauchy.rs): every
   dispersion model carries the wavelength range it was fitted over. Evaluating outside it returns
   an error and never a NaN or a pole value. Tests: BK7 at 9 um is an error; BK7 at 0.5876 um is
   1.5168 within 1e-4. Add a Planck test (optics/planck.rs:15-23) that small `lambda * T` returns 0
   and not NaN.

Run `cargo test -p periodica-mat`. If the range needs new `PropertyId`s, append them at the end of
the list so stored indices stay stable, and name them in your report.
