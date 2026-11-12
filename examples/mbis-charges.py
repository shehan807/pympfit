import numpy as np

# From openff-recharge (dependency)
from openff.recharge.utilities.molecule import extract_conformers
from openff.toolkit.topology import Molecule

# From pympfit
from pympfit import (
    MBISSettings,
    MoleculeMBISRecord,
    Psi4MBISGenerator,
    extract_mbis_charges,
    generate_mpfit_charge_parameter,
)
from pympfit.mpfit.solvers import MPFITSVDSolver


def main():
    mbis_settings = MBISSettings(
        method="pbe0",
        basis="def2-SVP",
    )

    molecule: Molecule = Molecule.from_mapped_smiles(
        "[C:1]([H:5])([H:6])([H:8])[C:2]([H:7])([H:9])[O:3][H:4]"
    )
    molecule.generate_conformers(n_conformers=1)

    [input_conformer] = extract_conformers(molecule)

    coords, multipoles = Psi4MBISGenerator.generate(
        molecule, input_conformer, mbis_settings, minimize=True
    )

    mbis_record = MoleculeMBISRecord.from_molecule(
        molecule, coords, multipoles, mbis_settings
    )
    print("Multipole moments shape:", multipoles.shape)

    # Path 1, raw MBIS charges: the Psi4-emitted partial charges (the Q00
    # column of the multipoles), wrapped in a LibraryChargeParameter.
    mbis_parameter = extract_mbis_charges(mbis_record)

    # Path 2, refit: fit a new set of point charges to all the MBIS multipoles
    # with the same MPFIT SVD solver used for GDMA data.
    mpfit_solver = MPFITSVDSolver(svd_threshold=1.0e-4)
    fitted_parameter = generate_mpfit_charge_parameter([mbis_record], mpfit_solver)

    print(f"SMILES                   : {mbis_parameter.smiles}")
    print(f"MBIS CHARGES (UNIQUE)    : {np.round(mbis_parameter.value, 4)}")
    print(f"MPFIT-from-MBIS (UNIQUE) : {np.round(fitted_parameter.value, 4)}")
    print()

    # Print each atom with its raw MBIS charge
    print("Atom-by-atom MBIS charges:")
    from openff.units.elements import SYMBOLS

    for i, atom in enumerate(molecule.atoms):
        element = SYMBOLS[atom.atomic_number]
        print(f"Atom {i} ({element}): {mbis_parameter.value[i]:.4f}")


if __name__ == "__main__":
    main()
