# Contributing to PyMPFIT

Thanks for your interest in improving PyMPFIT! Bug reports, feature requests, and
pull requests are all welcome.

## Development installation

PyMPFIT depends on Psi4/GDMA and the OpenFF stack, distributed via conda-forge. The
most reliable way to get a working development environment is to build it from the
same conda spec CI uses, with `conda`, `mamba`, or `micromamba`:

```bash
# create the environment CI tests against
micromamba create -n pympfit-dev -f devtools/conda-envs/test_env_rdkit.yaml python=3.12
# add the quantum-chemistry stack
micromamba install -n pympfit-dev psi4 pygdma -c conda-forge
# install pympfit in editable mode
micromamba run -n pympfit-dev python -m pip install -e .
```

Verify the install:

```bash
micromamba run -n pympfit-dev python -c "import pympfit; print(pympfit.__version__)"
```

## Running the tests

```bash
micromamba run -n pympfit-dev pytest pympfit/tests/ -n auto
```

The full matrix (Python 3.11 and 3.12 on Linux) plus coverage runs automatically in CI
on every pull request.

## Code style

PyMPFIT uses [ruff](https://docs.astral.sh/ruff/) and [pre-commit](https://pre-commit.com/).
Install the hooks once; they then run on every commit:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files   # check the whole tree
```

The CI "Lint" job runs these same hooks, so a clean `pre-commit run --all-files` means
CI lint will pass.

## Submitting a pull request

1. Fork the repository and create a feature branch off `main`.
2. Make your change, adding or updating tests where it makes sense.
3. Make sure `pre-commit run --all-files` and `pytest pympfit/tests/` pass locally.
4. Open a pull request against `main`, describing what changed and why.

Please keep pull requests focused — one logical change per PR is the easiest to review.

## Reporting issues and getting help

Please use the [issue templates](.github/ISSUE_TEMPLATE/) to file bugs and feature
requests, and see [SUPPORT.md](SUPPORT.md) for where to ask questions. All participation
is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
