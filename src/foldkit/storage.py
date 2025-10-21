import numpy as np
from foldkit import AF3Result
from pathlib import Path


def save_af3_result(af3_result: AF3Result, path: str):
    """
    Save this AF3Result to a compressed NPZ file.
    """
    path = Path(path)
    np.savez_compressed(
        path,
        id=af3_result.id,
        chains=np.array(af3_result.chains),
        residue_chain_ids=af3_result.residue_chain_ids,
        atom_chain_ids=af3_result.atom_chain_ids,
        plddt=af3_result.plddt,
        pae=af3_result.pae,
        contact_probs=af3_result.contact_probs,
        global_ptm=af3_result.global_ptm,
        global_iptm=af3_result.global_iptm,
        chain_pair_iptm=af3_result.chain_pair_iptm,
        chain_ptm=af3_result.chain_ptm,
    )


def load_af3_result(path: str) -> "AF3Result":
    """
    Load an AF3Result from a saved NPZ file.
    """
    data = np.load(path, allow_pickle=True)
    obj = AF3Result()
    obj.id = str(data["id"])

    # Restore arrays
    obj.chains = list(data["chains"])
    obj.residue_chain_ids = data["residue_chain_ids"]
    obj.atom_chain_ids = data["atom_chain_ids"]
    obj.plddt = data["plddt"]
    obj.pae = data["pae"]
    obj.contact_probs = data["contact_probs"]
    obj.global_ptm = float(data["global_ptm"])
    obj.global_iptm = float(data["global_iptm"])
    obj.chain_pair_iptm = data["chain_pair_iptm"]
    obj.chain_ptm = data["chain_ptm"]

    return obj
