import pytest
import foldkit
import os


def test_example():
    aa = foldkit.AF3Result.load_result("tests/test_data/structure1")
    assert aa.get_ptm() == 0.66
    assert aa.get_iptm() == 0.6
    assert aa.get_ptm(chain="A") == 0.68
    assert aa.get_iptm(chain1="A", chain2="B") == 0.75
    assert aa.get_ipae(chain1="A", chain2="B") == pytest.approx(12.88, rel=1e-2)

    with pytest.raises(ValueError) as excinfo:
        aa.get_iptm(chain1="A", chain2="X")
    assert "Chain X not in" in str(excinfo.value)
