"""Shared record-type alias used across the fitting and optimize layers."""

from pympfit.gdma.storage import MoleculeGDMARecord
from pympfit.mbis.storage import MoleculeMBISRecord

# A per-molecule multipole record from either the GDMA or the MBIS path.
MultipoleRecord = MoleculeGDMARecord | MoleculeMBISRecord

__all__ = ["MultipoleRecord"]
