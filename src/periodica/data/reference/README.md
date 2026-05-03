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
