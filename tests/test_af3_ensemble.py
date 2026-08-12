import json

import numpy as np
import pytest

import foldkit
from foldkit.af3_ensemble import AF3Ensemble


PAE = [
    [0.0, 5.0, 6.0],
    [5.0, 0.0, 2.0],
    [6.0, 2.0, 0.0],
]
CONTACT_PROBS = [
    [1.0, 0.5, 0.4],
    [0.5, 1.0, 0.9],
    [0.4, 0.9, 1.0],
]
PLDDT = [90.0, 91.0, 92.0]

CIF_TEMPLATE = (
    "_atom_site.group_PDB\n"
    "_atom_site.id\n"
    "_atom_site.label_atom_id\n"
    "_atom_site.label_comp_id\n"
    "_atom_site.label_asym_id\n"
    "_atom_site.label_seq_id\n"
    "_atom_site.Cartn_x\n"
    "_atom_site.Cartn_y\n"
    "_atom_site.Cartn_z\n"
    "ATOM 1 CA ALA A 1 0.0 0.0 0.0\n"
    "ATOM 2 CB ALA A 1 1.0 0.0 0.0\n"
    "ATOM 3 CA GLY B 1 5.0 0.0 0.0\n"
    "ATOM 4 CA GLY B 2 6.0 0.0 0.0\n"
)

RANKING_TEMPLATE = """seed,sample,ranking_score
1,0,0.8945341059820476
1,1,0.8887886611031466
2,0,0.891021758050243
"""


def _write_af3_sample_dir(dir_path, ptm, iptm):
    """Write one synthetic AF3Result-loadable directory (2 chains: A, B)."""
    dir_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "ptm": ptm,
        "iptm": iptm,
        "chain_pair_iptm": [[0.0, iptm], [iptm, 0.0]],
        "chain_ptm": [ptm, ptm],
    }
    (dir_path / "job_summary_confidences.json").write_text(json.dumps(summary))

    full = {
        "token_chain_ids": ["A", "B", "B"],
        "atom_chain_ids": ["A", "B", "B"],
        "atom_plddts": PLDDT,
        "pae": PAE,
        "contact_probs": CONTACT_PROBS,
    }
    (dir_path / "job_confidences.json").write_text(json.dumps(full))

    (dir_path / "job_model.cif").write_text(CIF_TEMPLATE)


# seed/sample -> (ptm, iptm); deliberately spread out for aggregation tests
SAMPLES = {
    "seed-1_sample-0": (0.5, 0.4),
    "seed-1_sample-1": (0.7, 0.6),
    "seed-2_sample-0": (0.9, 0.8),
}


@pytest.fixture
def ensemble_dir(tmp_path):
    top = tmp_path / "my_ensemble"
    for name, (ptm, iptm) in SAMPLES.items():
        _write_af3_sample_dir(top / name, ptm, iptm)
    with open(top / "ranking_scores.csv", "w", newline="") as f:
        f.write(RANKING_TEMPLATE)
    return top


@pytest.fixture
def ensemble(ensemble_dir):
    return foldkit.AF3Ensemble.load_af3_result(str(ensemble_dir))


def test_load_af3_result_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        foldkit.AF3Ensemble.load_af3_result("tests/test_data/does_not_exist")


def test_load_af3_result_uses_dirname_as_default_id(ensemble_dir, ensemble):
    assert ensemble.id == "my_ensemble"


def test_load_af3_result_accepts_explicit_id(ensemble_dir):
    ens = foldkit.AF3Ensemble.load_af3_result(str(ensemble_dir), id="custom_id")
    assert ens.id == "custom_id"


def test_load_af3_result_extracts_seeds_and_samples(ensemble):
    assert ensemble.seeds == [1, 2]
    assert ensemble.samples == [0, 1]
    assert ensemble.size == 3


def test_load_af3_result_populates_af3_results(ensemble):
    assert set(ensemble.af3_results.keys()) == set(SAMPLES.keys())
    for key, af3_result in ensemble.af3_results.items():
        assert isinstance(af3_result, foldkit.AF3Result)


def test_load_af3_result_rejects_badly_named_subdirectory(ensemble_dir, tmp_path):
    top = tmp_path / "my_ensemble"
    _write_af3_sample_dir(top / "not_a_valid_name", 0.5, 0.4)
    with pytest.raises(Exception) as excinfo:
        foldkit.AF3Ensemble.load_af3_result(str(top))
    assert "Unable to extract seed and sample" in str(excinfo.value)


def test_extract_result_objects_skips_if_already_initialized(ensemble, capsys):
    before = dict(ensemble.af3_results)
    ensemble._extract_result_objects(compressed=False)
    captured = capsys.readouterr()
    assert "Already initialized" in captured.out
    assert ensemble.af3_results == before


def test_get_result_by_seed_and_sample(ensemble):
    res = ensemble.get_result_by_seed_and_sample(seed=1, sample=1)
    assert isinstance(res, foldkit.AF3Result)
    assert res.get_ptm() == pytest.approx(0.7)


def test_get_result_by_seed_and_sample_invalid_seed(ensemble):
    with pytest.raises(ValueError) as excinfo:
        ensemble.get_result_by_seed_and_sample(seed=99, sample=0)
    assert "Seed 99 not in ensemble" in str(excinfo.value)


def test_get_result_by_seed_and_sample_invalid_sample(ensemble):
    with pytest.raises(ValueError) as excinfo:
        ensemble.get_result_by_seed_and_sample(seed=1, sample=99)
    assert "Sample 99 not in ensemble" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ptm
# ---------------------------------------------------------------------------


def test_get_all_ptm(ensemble):
    all_ptm = ensemble.get_all_ptm()
    assert all_ptm == {
        "seed-1_sample-0": pytest.approx(0.5),
        "seed-1_sample-1": pytest.approx(0.7),
        "seed-2_sample-0": pytest.approx(0.9),
    }


def test_get_all_ptm_by_chain(ensemble):
    all_ptm_a = ensemble.get_all_ptm(chain="A")
    assert all_ptm_a["seed-1_sample-0"] == pytest.approx(0.5)


def test_get_all_ptm_invalid_chain_propagates(ensemble):
    with pytest.raises(ValueError):
        ensemble.get_all_ptm(chain="Z")


def test_get_ensemble_ptm_default_mean(ensemble):
    expected = np.mean([0.5, 0.7, 0.9])
    assert ensemble.get_ensemble_ptm() == pytest.approx(expected)


def test_get_ensemble_ptm_custom_agg(ensemble):
    assert ensemble.get_ensemble_ptm(ensemble_agg=np.max) == pytest.approx(0.9)
    assert ensemble.get_ensemble_ptm(ensemble_agg=np.min) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# iptm
# ---------------------------------------------------------------------------


def test_get_all_iptm(ensemble):
    all_iptm = ensemble.get_all_iptm(chain1="A", chain2="B")
    assert all_iptm == {
        "seed-1_sample-0": pytest.approx(0.4),
        "seed-1_sample-1": pytest.approx(0.6),
        "seed-2_sample-0": pytest.approx(0.8),
    }


def test_get_ensemble_iptm_default_mean(ensemble):
    expected = np.mean([0.4, 0.6, 0.8])
    assert ensemble.get_ensemble_iptm(chain1="A", chain2="B") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# pae
# ---------------------------------------------------------------------------


def test_get_all_pae_global(ensemble):
    all_pae = ensemble.get_all_pae()
    assert set(all_pae.keys()) == set(SAMPLES.keys())
    for value in all_pae.values():
        assert value >= 0


def test_get_all_pae_same_across_samples(ensemble):
    # PAE matrix is identical across all synthetic samples in this fixture
    all_pae = ensemble.get_all_pae(chain="A")
    values = list(all_pae.values())
    assert values == pytest.approx([values[0]] * len(values))


def test_get_ensemble_pae_matches_manual_mean(ensemble):
    all_pae = ensemble.get_all_pae()
    expected = np.mean(list(all_pae.values()))
    assert ensemble.get_ensemble_pae() == pytest.approx(expected)


def test_get_all_pae_rejects_chain_and_tokens_together(ensemble):
    with pytest.raises(ValueError):
        ensemble.get_all_pae(chain="A", tokens=[0, 1])


# ---------------------------------------------------------------------------
# ipae
# ---------------------------------------------------------------------------


def test_get_all_ipae(ensemble):
    all_ipae = ensemble.get_all_ipae(chain1="A", chain2="B")
    assert set(all_ipae.keys()) == set(SAMPLES.keys())
    for value in all_ipae.values():
        assert value >= 0


def test_get_ensemble_ipae_matches_manual_mean(ensemble):
    all_ipae = ensemble.get_all_ipae(chain1="A", chain2="B")
    expected = np.mean(list(all_ipae.values()))
    assert ensemble.get_ensemble_ipae(chain1="A", chain2="B") == pytest.approx(expected)


def test_get_all_ipae_requires_selector(ensemble):
    with pytest.raises(ValueError):
        ensemble.get_all_ipae()


# ---------------------------------------------------------------------------
# ipsae
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d0", ["d0res", "d0dom", "d0chn"])
def test_get_all_ipsae_d0_options(ensemble, d0):
    all_ipsae = ensemble.get_all_ipsae(chain1="A", chain2="B", d0=d0)
    assert set(all_ipsae.keys()) == set(SAMPLES.keys())
    for value in all_ipsae.values():
        assert 0.0 <= value <= 1.0


def test_get_ensemble_ipsae_matches_manual_mean(ensemble):
    all_ipsae = ensemble.get_all_ipsae(chain1="A", chain2="B")
    expected = np.mean(list(all_ipsae.values()))
    assert ensemble.get_ensemble_ipsae(chain1="A", chain2="B") == pytest.approx(
        expected
    )


def test_get_all_ipsae_rejects_same_chain(ensemble):
    with pytest.raises(AssertionError):
        ensemble.get_all_ipsae(chain1="A", chain2="A")


# ---------------------------------------------------------------------------
# contact probabilities
# ---------------------------------------------------------------------------


def test_get_all_contact_probs_range(ensemble):
    all_probs = ensemble.get_all_contact_probs(chain1="A", chain2="B")
    for value in all_probs.values():
        assert 0.0 <= value <= 1.0


def test_get_ensemble_contact_probs_default_agg_is_mean_of_maxes(ensemble):
    all_probs = ensemble.get_all_contact_probs(chain1="A", chain2="B")
    expected = np.mean(list(all_probs.values()))
    assert ensemble.get_ensemble_contact_probs(chain1="A", chain2="B") == pytest.approx(
        expected
    )


# ---------------------------------------------------------------------------
# plddt
# ---------------------------------------------------------------------------


def test_get_all_plddt_range(ensemble):
    all_plddt = ensemble.get_all_plddt()
    for value in all_plddt.values():
        assert 0.0 <= value <= 100.0


def test_get_all_plddt_by_chain(ensemble):
    all_plddt_a = ensemble.get_all_plddt(chain="A")
    for value in all_plddt_a.values():
        assert value == pytest.approx(90.0)


def test_get_ensemble_plddt_matches_manual_mean(ensemble):
    all_plddt = ensemble.get_all_plddt()
    expected = np.mean(list(all_plddt.values()))
    assert ensemble.get_ensemble_plddt() == pytest.approx(expected)


def test_get_all_plddt_rejects_chain_and_atoms_together(ensemble):
    with pytest.raises(ValueError):
        ensemble.get_all_plddt(chain="A", atoms=[0, 1])
