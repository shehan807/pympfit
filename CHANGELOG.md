# Changelog

All notable changes to PyMPFIT are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MBIS distributed-multipole support: `Psi4MBISGenerator`, `MBISSettings`,
  `MoleculeMBISRecord`, and a `MoleculeMBISStore`, alongside the existing GDMA path.
- A top-level `pympfit.electrostatics` subpackage for Cartesian multipole evaluation.
- `extract_mbis_charges()` for raw MBIS charges.
- Contribution, support, and security guidelines; GitHub issue templates; this changelog.

### Fixed
- Dropped the obsolete `libint=2.9` pin, which forced an old Psi4 build incompatible
  with current `einsums` and broke `import psi4` on Linux and macOS; corrected the
  install instructions to match.
- Documentation fixes: the `getting_started` install block, the `user_guide` `radius`
  example, and the `Psi4GDMAGenerator._generate_input` docstring parameters.

## [1.0.0] - 2026-02-04

First public release on PyPI. Multipole-based partial-charge fitting from Gaussian
Distributed Multipole Analysis (GDMA) data, with multi-molecule constrained fitting,
virtual sites, and Bayesian inference of virtual-site charges and geometries.

[Unreleased]: https://github.com/shehan807/pympfit/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/shehan807/pympfit/releases/tag/v1.0.0
