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
