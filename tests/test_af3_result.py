import json

import numpy as np
import pytest

import foldkit

TEST_DATA_DIR = "tests/test_data/multi_seed_1/seed-1_sample-3"


@pytest.fixture(scope="module")
def result():
    return foldkit.AF3Result.load_af3_result(TEST_DATA_DIR)


def test_example():
    aa = foldkit.AF3Result.load_af3_result(TEST_DATA_DIR)
    assert aa.get_ptm() == 0.81
    assert aa.get_iptm() == 0.84
    assert aa.get_ptm(chain="A") == 0.82
    assert aa.get_iptm(chain1="A", chain2="B") == 0.88
    assert aa.get_ipae(chain1="A", chain2="B") == pytest.approx(6.1, rel=1e-2)

    with pytest.raises(ValueError) as excinfo:
        aa.get_iptm(chain1="A", chain2="X")
    assert "Chain X not in" in str(excinfo.value)


def test_get_ptm_invalid_chain_raises(result):
    with pytest.raises(ValueError) as excinfo:
        result.get_ptm(chain="Z")
    assert "Chain Z not in" in str(excinfo.value)


def test_get_iptm_requires_both_chains(result):
    # only one of chain1/chain2 provided
    with pytest.raises(ValueError):
        result.get_iptm(chain1="A")
    with pytest.raises(ValueError):
        result.get_iptm(chain2="B")


def test_get_iptm_is_directional(result):
    # chain_pair_iptm is symmetric
    ab = result.get_iptm(chain1="A", chain2="B")
    ba = result.get_iptm(chain1="B", chain2="A")
    assert isinstance(ab, float)
    assert isinstance(ba, float)
    assert ab == ba


def test_get_pae_global(result):
    pae = result.get_pae()
    assert isinstance(pae, float)
    assert pae >= 0


def test_get_pae_by_chain(result):
    pae_a = result.get_pae(chain="A")
    assert pae_a >= 0


def test_get_pae_by_tokens(result):
    tokens = result._chain_residue_indices("A")[:3]
    pae = result.get_pae(tokens=tokens)
    assert pae >= 0


def test_get_pae_cannot_provide_both_chain_and_tokens(result):
    with pytest.raises(ValueError) as excinfo:
        result.get_pae(chain="A", tokens=[0, 1, 2])
    assert "Cannot provide both" in str(excinfo.value)


def test_get_pae_custom_agg(result):
    max_pae = result.get_pae(chain="A", agg=np.max)
    mean_pae = result.get_pae(chain="A", agg=np.mean)
    assert max_pae >= mean_pae


def test_get_ipae_requires_selector(result):
    with pytest.raises(ValueError) as excinfo:
        result.get_ipae()
    assert "Must provide either" in str(excinfo.value)


def test_get_ipae_rejects_partial_chains(result):
    with pytest.raises(ValueError) as excinfo:
        result.get_ipae(chain1="A")
    assert "Must provide both chains" in str(excinfo.value)


def test_get_ipae_rejects_partial_tokens(result):
    tokens = result._chain_residue_indices("A")[:2]
    with pytest.raises(ValueError) as excinfo:
        result.get_ipae(tokens1=tokens)
    assert "Must provide both tokens" in str(excinfo.value)


def test_get_ipae_rejects_mixed_chain_and_token_args(result):
    tokens = result._chain_residue_indices("B")[:2]
    with pytest.raises(ValueError) as excinfo:
        result.get_ipae(chain1="A", chain2="B", tokens1=tokens, tokens2=tokens)
    assert "Cannot provide both" in str(excinfo.value)


def test_get_ipae_by_tokens_matches_by_chain(result):
    tokens_a = result._chain_residue_indices("A")
    tokens_b = result._chain_residue_indices("B")
    by_chain = result.get_ipae(chain1="A", chain2="B")
    by_tokens = result.get_ipae(tokens1=tokens_a, tokens2=tokens_b)
    assert by_chain == pytest.approx(by_tokens, rel=1e-6)


def test_get_ipae_by_tokens_handles_unequal_length_lists(result):
    # tokens1/tokens2 represent independent, possibly non-contiguous residue
    # selections (e.g. a CDR loop vs. a whole partner chain) and should not
    # be required to be the same length - the full cross-product block is
    # what's aggregated, not an elementwise pairing.
    tokens_a = result._chain_residue_indices("A")[:5]  # e.g. a short CDR loop
    tokens_b = result._chain_residue_indices("B")  # full chain, likely longer
    assert len(tokens_a) != len(tokens_b)

    score = result.get_ipae(tokens1=tokens_a, tokens2=tokens_b)
    manual = np.mean(
        np.concatenate(
            (
                result.pae[np.ix_(tokens_a, tokens_b)],
                result.pae[np.ix_(tokens_b, tokens_a)],
            ),
            axis=None,
        )
    )
    assert score == pytest.approx(manual)


def test_get_contact_probs_requires_selector(result):
    with pytest.raises(ValueError):
        result.get_contact_probs()


def test_get_contact_probs_range(result):
    prob = result.get_contact_probs(chain1="A", chain2="B")
    assert 0.0 <= prob <= 1.0


def test_get_contact_probs_default_agg_is_max(result):
    max_prob = result.get_contact_probs(chain1="A", chain2="B")
    mean_prob = result.get_contact_probs(chain1="A", chain2="B", agg=np.mean)
    assert max_prob >= mean_prob


def test_get_plddt_global(result):
    plddt = result.get_plddt()
    assert 0.0 <= plddt <= 100.0


def test_get_plddt_by_chain(result):
    plddt_a = result.get_plddt(chain="A")
    assert 0.0 <= plddt_a <= 100.0


def test_get_plddt_cannot_provide_both_chain_and_atoms(result):
    with pytest.raises(ValueError) as excinfo:
        result.get_plddt(chain="A", atoms=[0, 1, 2])
    assert "Cannot provide both" in str(excinfo.value)


@pytest.mark.parametrize("d0", ["d0res", "d0dom", "d0chn"])
def test_get_ipsae_d0_options(result, d0):
    score = result.get_ipsae(chain1="A", chain2="B", d0=d0)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_get_ipsae_is_asymmetric_key_lookup(result):
    ab = result.get_ipsae(chain1="A", chain2="B")
    ba = result.get_ipsae(chain1="B", chain2="A")
    # asymmetric metric - not required to be equal, but both must resolve
    assert isinstance(ab, float)
    assert isinstance(ba, float)


def test_get_ipsae_rejects_same_chain(result):
    with pytest.raises(AssertionError):
        result.get_ipsae(chain1="A", chain2="A")


def test_get_ipsae_rejects_invalid_chain(result):
    with pytest.raises(AssertionError):
        result.get_ipsae(chain1="A", chain2="Z")


def test_get_ipsae_rejects_invalid_d0(result):
    with pytest.raises(AssertionError):
        result.get_ipsae(chain1="A", chain2="B", d0="not_a_real_option")


def test_load_af3_result_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        foldkit.AF3Result.load_af3_result("tests/test_data/does_not_exist")


def test_load_af3_result_wrong_number_of_cif_files(tmp_path):
    # no .cif file present at all
    (tmp_path / "job_summary_confidences.json").write_text("{}")
    (tmp_path / "job_confidences.json").write_text("{}")
    with pytest.raises(FileNotFoundError) as excinfo:
        foldkit.AF3Result.load_af3_result(str(tmp_path))
    assert "cif" in str(excinfo.value)


def test_load_af3_result_missing_summary_json(tmp_path):
    (tmp_path / "job_model.cif").write_text("")
    (tmp_path / "job_confidences.json").write_text("{}")
    with pytest.raises(FileNotFoundError) as excinfo:
        foldkit.AF3Result.load_af3_result(str(tmp_path))
    assert "summary_confidences" in str(excinfo.value)


def test_load_af3_result_missing_full_json(tmp_path):
    (tmp_path / "job_model.cif").write_text("")
    (tmp_path / "job_summary_confidences.json").write_text("{}")
    with pytest.raises(FileNotFoundError) as excinfo:
        foldkit.AF3Result.load_af3_result(str(tmp_path))
    assert "confidences.json" in str(excinfo.value)


def test_load_af3_result_uses_dirname_as_default_id(tmp_path):
    result_dir = tmp_path / "my_job"
    result_dir.mkdir()

    summary = {
        "ptm": 0.5,
        "iptm": 0.4,
        "chain_pair_iptm": [[0.0, 0.4], [0.4, 0.0]],
        "chain_ptm": [0.5, 0.5],
    }
    (result_dir / "job_summary_confidences.json").write_text(json.dumps(summary))

    full = {
        "token_chain_ids": ["A", "A", "B", "B"],
        "atom_chain_ids": ["A", "A", "B", "B"],
        "atom_plddts": [90.0, 91.0, 92.0, 93.0],
        "pae": [[0, 1, 2, 3], [1, 0, 2, 3], [2, 2, 0, 1], [3, 3, 1, 0]],
        "contact_probs": [
            [1, 0.9, 0.1, 0.1],
            [0.9, 1, 0.1, 0.1],
            [0.1, 0.1, 1, 0.9],
            [0.1, 0.1, 0.9, 1],
        ],
    }
    (result_dir / "job_confidences.json").write_text(json.dumps(full))

    cif_content = (
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
        "ATOM 4 CA GLY B 1 6.0 0.0 0.0\n"
    )
    (result_dir / "job_model.cif").write_text(cif_content)

    res = foldkit.AF3Result.load_af3_result(str(result_dir))
    assert res.id == "my_job"
    assert res.chains == ["A", "B"]
    assert res.get_chain_seq(chain="A") == "A"
    assert res.get_chain_seq(chain="B") == "GG"
    assert res.get_ptm() == pytest.approx(0.5)
    assert res.get_iptm(chain1="A", chain2="B") == pytest.approx(0.4)
