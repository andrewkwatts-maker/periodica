# F3: mixing-rule fixes (lane `mix`)

Read `_common.md` first. Source: `docs/plan/sheets-formula-review.md`, BUG 6, U1, and item 1 of
"missing". Deliverable: three fixes in `rust/periodica-mat/src/mix.rs`, each with its test, in one
commit per fix on `claude/mix`. Put the tests in a new file, `rust/periodica-mat/tests/f3_mix.rs`,
so the lanes don't collide.

1. Thermal expansion (mix.rs:77-82): use Turner, `alpha = sum(alpha_i K_i V_i) / sum(K_i V_i)`,
   when every component has a bulk modulus (given or derivable from E and nu). Otherwise keep the
   volume average and tag the result `Estimated`. Test: Al (23e-6 /K, K 68.6 GPa) and steel
   (12e-6 /K, K 167 GPa) at 50/50 give 15.3e-6 /K within 1 %.
2. Elastic consistency (mix.rs:83, 159-162): take the Hill average on K and G only, then derive E
   and nu from them (`E = 9KG/(3K+G)`, `nu = (3K-2G)/(2(3K+G))`), so the mixed constants satisfy
   the isotropic identities exactly. Test: the mixed E equals 9KG/(3K+G) to 1e-12 relative.
3. Refractive index: make `RefractiveIndexN` mixable by Lorentz-Lorenz,
   `(n^2-1)/(n^2+2) = sum f_i (n_i^2-1)/(n_i^2+2)` over volume fractions, for components with
   `ExtinctionK` absent or zero. If any component has k > 0, stay NotMixable. Test: n 1.5 and 1.0 at
   50/50 give 1.232 within 1e-3.

Don't change the conductivity, strength or surface-energy rules; those are owner decisions (U2, U4,
U5, U6). Fix the module doc where it disagrees with the code you touch. Run
`cargo test -p periodica-mat`. If an existing oracle changes value, update it and list the old and
new numbers in your report.
