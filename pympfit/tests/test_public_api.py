"""Lightweight checks on the top-level ``pympfit`` public API surface."""

import typing

import pympfit


def test_all_exports_importable():
    """Every name in ``pympfit.__all__`` is actually attached to the package."""
    for name in pympfit.__all__:
        assert hasattr(pympfit, name), f"pympfit.{name} is missing"


def test_stores_exposed_at_top_level():
    """The GDMA and MBIS stores are reachable from the top-level namespace."""
    from pympfit import MoleculeGDMAStore, MoleculeMBISStore
    from pympfit.gdma.storage import MoleculeGDMAStore as GDMAStore
    from pympfit.mbis.storage import MoleculeMBISStore as MBISStore

    assert MoleculeGDMAStore is GDMAStore
    assert MoleculeMBISStore is MBISStore


def test_multipole_record_is_single_object():
    """MultipoleRecord is defined once and re-exported, not duplicated per module."""
    import pympfit._records
    import pympfit.mpfit
    import pympfit.optimize

    canonical = pympfit._records.MultipoleRecord
    assert pympfit.MultipoleRecord is canonical
    assert pympfit.mpfit.MultipoleRecord is canonical
    assert pympfit.optimize.MultipoleRecord is canonical


def test_multipole_record_unions_gdma_and_mbis():
    """The alias is the union of the GDMA and MBIS record types."""
    from pympfit.gdma.storage import MoleculeGDMARecord
    from pympfit.mbis.storage import MoleculeMBISRecord

    members = set(typing.get_args(pympfit.MultipoleRecord))
    assert members == {MoleculeGDMARecord, MoleculeMBISRecord}
