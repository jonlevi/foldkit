import pytest
import foldkit
import os
from numpy.testing import assert_array_equal

#  PYTHONPATH=src pytest tests/


def test_example():
    aa = foldkit.AF3Result.load_result("tests/test_data/structure1")

    foldkit.save_af3_result(aa, "tests/test_data/structure1.npz")

    bb = foldkit.load_af3_result("tests/test_data/structure1.npz")

    assert aa.id == bb.id
    assert aa.chains == bb.chains
    assert_array_equal(aa.residue_chain_ids, bb.residue_chain_ids)
    assert_array_equal(aa.atom_chain_ids, bb.atom_chain_ids)
    assert_array_equal(aa.plddt, bb.plddt)
    assert_array_equal(aa.pae, bb.pae)
    assert_array_equal(aa.contact_probs, bb.contact_probs)
    assert aa.global_ptm == bb.global_ptm
    assert_array_equal(aa.chain_pair_iptm, bb.chain_pair_iptm)
    assert_array_equal(aa.chain_ptm, bb.chain_ptm)
