import numpy as np
import pytest


class TestMBISMultipoleEvaluation:
    """Test MBIS multipole generation and electrostatic energy evaluation."""

    def test_mbis_dimer_multipoles_match_monomers(self):
        """Test that MBIS multipoles from dimer match monomer calculations.

        This test verifies that:
        1. Multipoles generated with max_moment parameter are correctly sized
        2. Multipole energies can be evaluated using Cartesian format
        3. Dimer MBIS multipoles match monomer MBIS multipoles within tolerance
        4. The electrostatic interaction energy is reasonable
        """
        pytest.importorskip("psi4")
        pytest.importorskip("qcelemental")

        import psi4
        import qcelemental as qcel
        from openff.recharge.utilities.molecule import extract_conformers
        from openff.toolkit import Molecule
        from openff.units import unit
        from qcelemental import constants

        from pympfit import (
            MBISSettings,
            MoleculeMBISRecord,
            MPFITSVDSolver,
            Psi4MBISGenerator,
            generate_mpfit_charge_parameter,
        )
        from pympfit.electrostatics import evaluate_dimer_interaction_energy
        from pympfit.mbis.multipole_transform import flat_to_cartesian_multipoles

        # Create water molecule
        molecule = Molecule.from_smiles("O")
        molecule.generate_conformers(n_conformers=1)
        [conformer] = extract_conformers(molecule)

        # Create dimer by shifting second molecule
        shift = 2.0  # Angstrom
        mol_dict = molecule.to_dict()
        mol_str = "0 1\n"
        mol_str += "\n".join(
            f"{atom['atomic_number']} "
            f"{conformer[i, 0].to(unit.angstrom).magnitude} "
            f"{conformer[i, 1].to(unit.angstrom).magnitude} "
            f"{conformer[i, 2].to(unit.angstrom).magnitude}"
            for i, atom in enumerate(mol_dict["atoms"])
        )
        mol_str += "\n--\n0 1\n"
        mol_str += "\n".join(
            f"{atom['atomic_number']} "
            f"{conformer[i, 0].to(unit.angstrom).magnitude + shift} "
            f"{conformer[i, 1].to(unit.angstrom).magnitude + shift} "
            f"{conformer[i, 2].to(unit.angstrom).magnitude}"
            for i, atom in enumerate(mol_dict["atoms"])
        )
        mol_str += "\nunits angstrom"
        qcel_mol = qcel.models.Molecule.from_data(mol_str)

        # Generate MBIS multipoles with Cartesian format
        settings = MBISSettings(
            max_radial_moment=3,
            max_moment=3,
            limit=3,
            method="hf",
            basis="aug-cc-pvdz",
            multipole_format="cartesian",
        )

        # Generate multipoles for first conformer
        coords, multipoles = Psi4MBISGenerator.generate(
            molecule, conformer, settings, n_threads=1, memory=2 * unit.gigabyte
        )

        # Shape: 3 atoms x (1 charge + 3 dipole + 6 quadrupole) = (3, 10)
        assert multipoles.shape == (3, 10)

        # Generate multipoles for shifted conformer
        conformer_2 = conformer.copy()
        conformer_2[:, 0] += shift * unit.angstrom
        conformer_2[:, 1] += shift * unit.angstrom
        coords_2, multipoles_2 = Psi4MBISGenerator.generate(
            molecule, conformer_2, settings, n_threads=1, memory=2 * unit.gigabyte
        )

        # Create record and verify charge fitting works
        record = MoleculeMBISRecord.from_molecule(
            molecule, coords, multipoles, settings
        )
        charges = generate_mpfit_charge_parameter([record], MPFITSVDSolver())
        assert len(charges.value) == 3

        # Convert multipoles to Cartesian tensors
        charges_a, dipoles_a, quadrupoles_a, _ = flat_to_cartesian_multipoles(
            multipoles, max_moment=3
        )
        charges_b, dipoles_b, quadrupoles_b, _ = flat_to_cartesian_multipoles(
            multipoles_2, max_moment=3
        )

        # Evaluate electrostatic interaction energy
        e_elst = evaluate_dimer_interaction_energy(
            qcel_mol,
            charges_a,
            dipoles_a,
            quadrupoles_a,
            charges_b,
            dipoles_b,
            quadrupoles_b,
        )

        # Energy should be negative (attractive) and reasonable magnitude
        assert e_elst < 0.0, f"Expected negative interaction energy, got {e_elst:.4f}"
        assert abs(e_elst) < 20.0, f"Interaction energy {e_elst:.4f} seems too large"

        # Run SAPT0 for comparison
        psi4.core.be_quiet()
        psi4.set_num_threads(1)
        psi4.set_memory("2 GB")
        psi4.set_options(
            {
                "basis": "aug-cc-pVDZ",
                "scf_type": "df",
                "freeze_core": True,
                "guess": "sad",
            }
        )
        psi4.geometry(mol_str)
        psi4.energy("sapt0")
        qcvars = psi4.core.variables()
        sapt0_elst = qcvars["SAPT0 ELST ENERGY"] * constants.hartree2kcalmol

        # MBIS multipole energy should be within ~50% of SAPT0
        # (exact match not expected due to different physical approximations)
        assert abs(e_elst - sapt0_elst) / abs(sapt0_elst) < 0.5

        # Calculate monomer MBIS multipoles directly with psi4
        psi4.set_options(
            {
                "basis": "aug-cc-pVDZ",
                "scf_type": "df",
                "freeze_core": True,
                "guess": "sad",
                "mbis_radial_points": 99,
                "mbis_spherical_points": 590,
                "mbis_d_convergence": 9,
                "max_radial_moment": 3,
            }
        )

        # Monomer A
        psi4.geometry(qcel_mol.get_fragment(0).to_string("psi4"))
        _, wfn = psi4.energy("hf", return_wfn=True)
        psi4.oeprop(wfn, "mbis_charges")
        wfn_vars = wfn.variables()
        mbis_mon_a_q = wfn_vars["MBIS CHARGES"].flatten()
        mbis_mon_a_mu = wfn_vars["MBIS DIPOLES"].reshape(-1, 3)
        mbis_mon_a_theta = wfn_vars["MBIS QUADRUPOLES"].reshape(-1, 3, 3)

        # Make quadrupoles traceless (Psi4 doesn't return them traceless)
        for i in range(mbis_mon_a_theta.shape[0]):
            trace = np.trace(mbis_mon_a_theta[i])
            mbis_mon_a_theta[i, 0, 0] -= trace / 3.0
            mbis_mon_a_theta[i, 1, 1] -= trace / 3.0
            mbis_mon_a_theta[i, 2, 2] -= trace / 3.0

        # Monomer B
        psi4.geometry(qcel_mol.get_fragment(1).to_string("psi4"))
        _, wfn = psi4.energy("hf", return_wfn=True)
        psi4.oeprop(wfn, "mbis_charges")
        wfn_vars = wfn.variables()
        mbis_mon_b_q = wfn_vars["MBIS CHARGES"].flatten()
        mbis_mon_b_mu = wfn_vars["MBIS DIPOLES"].reshape(-1, 3)
        mbis_mon_b_theta = wfn_vars["MBIS QUADRUPOLES"].reshape(-1, 3, 3)

        # Make quadrupoles traceless (Psi4 doesn't return them traceless)
        for i in range(mbis_mon_b_theta.shape[0]):
            trace = np.trace(mbis_mon_b_theta[i])
            mbis_mon_b_theta[i, 0, 0] -= trace / 3.0
            mbis_mon_b_theta[i, 1, 1] -= trace / 3.0
            mbis_mon_b_theta[i, 2, 2] -= trace / 3.0

        # Verify monomer multipoles match dimer multipoles
        np.testing.assert_allclose(
            mbis_mon_a_q,
            charges_a,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Monomer A charges don't match dimer charges",
        )
        np.testing.assert_allclose(
            mbis_mon_a_mu,
            dipoles_a,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Monomer A dipoles don't match dimer dipoles",
        )
        np.testing.assert_allclose(
            mbis_mon_a_theta,
            quadrupoles_a,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Monomer A quadrupoles don't match dimer quadrupoles",
        )

        # Evaluate monomer-based interaction energy
        e_elst_monomers = evaluate_dimer_interaction_energy(
            qcel_mol,
            mbis_mon_a_q,
            mbis_mon_a_mu,
            mbis_mon_a_theta,
            mbis_mon_b_q,
            mbis_mon_b_mu,
            mbis_mon_b_theta,
        )

        # Monomer and dimer energies should match closely
        np.testing.assert_allclose(
            e_elst_monomers,
            e_elst,
            rtol=1e-3,
            atol=1e-3,
            err_msg=(
                f"Monomer energy {e_elst_monomers:.4f} doesn't match "
                f"dimer energy {e_elst:.4f}"
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__])
