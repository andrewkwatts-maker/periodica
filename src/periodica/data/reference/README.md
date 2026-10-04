# Reference data

Real-world hand-curated values used **only by the test suite** to validate
generated outputs.

The library's runtime code never reads from this folder. Generation always
composes from `data/active/` fundamentals through the generic `Get()` /
`Save()` flow. These files exist so that tests can compare a generated
result against an authoritative source (CODATA, NIST, IUPAC, PDG, etc.).

Layout mirrors `data/derived/`:

    reference/
      subatomic/   # composite hadrons (Proton, Neutron, ...)
      atoms/       # neutral atoms (H, He, ...)
      molecules/   # molecules (H2O, CO2, ...)

Each file has a flat schema with the canonical scalar properties plus a
`source` field citing the reference used.

## Exception: curated masses (`masses/`)

`masses/` holds the two mass references the generators consume at **build
time** (never at runtime -- the registry serves the generated files):

    masses/
      ciaaw2021_standard_atomic_weights.json  # CIAAW 2021 standard atomic weights
      ame2020_atomic_masses.json              # AME2020 atomic masses (subset used)

`data/config/composition_rules.json` (`mass_sources`) declares which generated
tier takes its mass from which file: atoms and ions from the CIAAW standard
atomic weight (the abridged value for the 14 interval elements; the AME2020
mass of the atom's own nuclide for the 34 elements CIAAW gives no weight for),
isotopes from AME2020. Each file carries a `_provenance` block with the
citation, source URL, retrieval date and the policy used.

