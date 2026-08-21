# Adapted from https://github.com/DunbrackLab/IPSAE
#
# Trimmed down to keep only the code path used by ipsae()

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

## constants
RESIDUE_SET = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    # the following seem wrong...
    "DA",
    "DC",
    "DT",
    "DG",
    "A",
    "C",
    "U",
    "G",
    "ligand",
}
NUC_RESIDUE_SET = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}


# Define the parse_atom_line function for PDB lines (by column) and mmCIF lines (split by white_space)
# parsed_line = parse_atom_line(line)
# line = "ATOM    123  CA  ALA A  15     11.111  22.222  33.333  1.00 20.00           C"
def parse_pdb_atom_line(line):
    atom_num = line[6:11].strip()
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    chain_id = line[21].strip()
    residue_seq_num = line[22:26].strip()
    x = line[30:38].strip()
    y = line[38:46].strip()
    z = line[46:54].strip()

    # Convert string numbers to integers or floats as appropriate
    atom_num = int(atom_num)
    residue_seq_num = int(residue_seq_num)
    x = float(x)
    y = float(y)
    z = float(z)

    return {
        "atom_num": atom_num,
        "atom_name": atom_name,
        "residue_name": residue_name,
        "chain_id": chain_id,
        "residue_seq_num": residue_seq_num,
        "x": x,
        "y": y,
        "z": z,
    }


def parse_cif_atom_line(line, fielddict):
    """Interpret line of CIF file.

    See docs/cif_parsing_reference.txt for further information.
    """
    linelist = line.split()
    atom_num = int(linelist[fielddict["id"]])
    atom_name = linelist[fielddict["label_atom_id"]]
    residue_name = linelist[fielddict["label_comp_id"]]
    chain_id = linelist[fielddict["label_asym_id"]]
    residue_seq_num = linelist[fielddict["label_seq_id"]]
    x = float(linelist[fielddict["Cartn_x"]])
    y = float(linelist[fielddict["Cartn_y"]])
    z = float(linelist[fielddict["Cartn_z"]])

    if residue_seq_num == ".":  # ligand atom
        return {
            "atom_num": atom_num,
            "atom_name": atom_name,
            "residue_name": "ligand",  # each atom in a ligand is a token
            "chain_id": chain_id,
            "residue_seq_num": residue_seq_num,
            "x": x,
            "y": y,
            "z": z,
        }
    else:
        return {
            "atom_num": atom_num,
            "atom_name": atom_name,
            "residue_name": residue_name,
            "chain_id": chain_id,
            "residue_seq_num": int(residue_seq_num),
            "x": x,
            "y": y,
            "z": z,
        }


def read_pdb(
    pdb_file: Iterable[str],
    protein_file_type,
):
    """Load residues from AlphaFold PDB or mmCIF file into lists; each residue is a dictionary
    Read PDB file to get CA coordinates, chainids, and residue numbers
    Convert to np arrays, and calculate distances"""
    chains = []
    token_mask = []
    residues = []
    cb_residues = []

    # contains order of atom_site fields in mmCIF files; handles any mmCIF field order
    atomsitefield_dict = {}
    atomsitefield_num = 0
    for i, line in enumerate(pdb_file.readlines()):
        if protein_file_type == "cif" and isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith("_atom_site."):
            line = line.strip()
            atomsite, fieldname = line.split(".")
            atomsitefield_dict[fieldname] = atomsitefield_num
            atomsitefield_num += 1

        if line.startswith("ATOM") or line.startswith("HETATM"):
            if protein_file_type == "cif":
                atom = parse_cif_atom_line(line, atomsitefield_dict)
            else:
                atom = parse_pdb_atom_line(line)
            if atom is None:  # ligand atom
                token_mask.append(0)
                continue

            if (
                atom["atom_name"] == "CA"
                or "C1" in atom["atom_name"]
                or atom["residue_name"] == "ligand"
            ):
                token_mask.append(1)
                residues.append(
                    {
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": atom["residue_name"],
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    }
                )
                chains.append(atom["chain_id"])

            if (
                atom["atom_name"] == "CB"
                or "C3" in atom["atom_name"]
                or (atom["residue_name"] == "GLY" and atom["atom_name"] == "CA")
                or atom["residue_name"] == "ligand"
            ):
                cb_residues.append(
                    {
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": atom["residue_name"],
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    }
                )

            # add nucleic acids and non-CA atoms in PTM residues to tokens (as 0), whether labeled as "HETATM" (af3) or as "ATOM" (boltz1)
            if (
                atom["atom_name"] != "CA"
                and "C1" not in atom["atom_name"]
                and atom["residue_name"] not in RESIDUE_SET
            ):
                token_mask.append(0)

    return residues, cb_residues, token_mask, np.array(chains)


def d0_matrix(chain_length_sum_matrix, chain_type_matrix):
    """d0 values for numbers/arrays; minimum value = 1.0 for protein, 2.0 for nucleic acid.

    From Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702-710 (2004)
    """
    # minimum value for d0: 1 for protein, 2 for NA
    twos = np.invert(chain_type_matrix.astype(bool)).astype(int) * 2
    min_d0_matrix = chain_type_matrix + twos
    L = np.maximum(27, chain_length_sum_matrix)
    return np.maximum(min_d0_matrix, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def classify_chains(chains, residue_types):
    """Identify each chain as either protein or nucleic acid (NA).

    Inputs:
    :chains:
    :residue_types:

    Outputs:
    Vector (n_chains) where each entry is either
    """
    # Get unique chains and iterate over them
    chain_types = []
    unique_chains = np.unique(
        chains,
    )
    for chain in unique_chains:
        # Find indices where the current chain is located
        indices = np.where(chains == chain)[0]
        # Get the residues for these indices
        chain_residues = residue_types[indices]
        # Count nucleic acid residues

        # Determine if the chain is a nucleic acid or protein
        chain_types.append(
            int(sum([residue in NUC_RESIDUE_SET for residue in chain_residues]) > 0)
        )

    # calculate is_nucleic for each chain, then cross and do OR
    chain_types = np.array([chain_types] * len(unique_chains))
    return np.invert(np.bitwise_or(chain_types, chain_types.T))


def _grouped_nunique_offdiag_sum(pair_matrix, group_starts):
    """Get sum of active rows and columns in contiguous groups."""

    # sum over chain2 residue axis (columns), grouping by rows in each chain 1: (n_residues, n_chains)
    row_x_cat = np.add.reduceat(pair_matrix, group_starts, axis=1) > 0
    # sum again over chain1 residue axis (rows), grouping by chains: (n_chains, n_chains)
    row_group_sum = np.add.reduceat(row_x_cat, group_starts, axis=0)
    row_group_sum = row_group_sum.astype(int)
    np.fill_diagonal(row_group_sum, 0)

    # sum over chain1 residue axis (rows), grouping by rows in each chain 2: (n_chains, n_residues)
    col_x_cat = np.add.reduceat(pair_matrix, group_starts, axis=0) > 0
    col_x_cat = col_x_cat.transpose()
    # sum again over chain2 residue axis (rows after transpose), grouping by chains: (n_chains, n_chains)
    col_group_sum = np.add.reduceat(col_x_cat, group_starts, axis=0)
    col_group_sum = col_group_sum.T.astype(int)
    np.fill_diagonal(col_group_sum, 0)
    return row_group_sum, col_group_sum


def get_residue_info(chains, residue_matrix):
    _, idx_start = np.unique(
        chains,
        return_index=True,
    )
    row_sums = np.add.reduceat(residue_matrix, idx_start, axis=0)
    pair_count_matrix = np.add.reduceat(row_sums, idx_start, axis=1)
    np.fill_diagonal(pair_count_matrix, 0)

    nunique_residues_chain1, nunique_residues_chain2 = _grouped_nunique_offdiag_sum(
        residue_matrix, idx_start
    )
    return pair_count_matrix, nunique_residues_chain1, nunique_residues_chain2


def reshape_vectorized_output(vectorized_output, chain_indicator_matrix):
    c, _, r = vectorized_output.shape
    cat = np.argmax(chain_indicator_matrix, axis=0)
    X_prime = vectorized_output[
        np.arange(c)[:, None], cat[None, :], np.arange(r)[None, :]
    ]
    return np.moveaxis(X_prime, 0, -1)  # shape (r, c)


def ipsae_reslevel(chains, residue_types, pae_matrix, pae_cutoff):
    """"""
    # get chain matrices
    unique_chains = pd.unique(chains)
    unique_chain_lengths = pd.Series(chains).value_counts(sort=False).values
    length_array = np.array([unique_chain_lengths] * len(unique_chain_lengths))

    chain_length_sum_array = length_array + length_array.T
    chain_type_matrix = classify_chains(chains, residue_types)

    chain_indicator_matrix = np.array([chains == c for c in unique_chains])
    nchains = len(unique_chains)

    masked_pae_matrix = np.where(
        np.bitwise_and(
            (pae_matrix < pae_cutoff),
            np.invert(chain_indicator_matrix.T @ chain_indicator_matrix),  # self-pairs
        ),
        pae_matrix,
        np.nan,
    )

    d0chn = d0_matrix(chain_length_sum_array, chain_type_matrix)

    ipsae_residue_matrix = np.where(
        pae_matrix < pae_cutoff, 1, 0
    )  # (n_residues, n_residues), same as valid_pairs_matrix
    # ipsae_pair_counts: (n_chains, n_chains): number of residue pairs for non-self chain pairs meeting pae cutoff
    # ipsae_nunique_residues_chain1/2: number of unique residues in respective chain participating in each set of pairs
    ipsae_pair_counts, ipsae_nunique_residues_chain1, ipsae_nunique_residues_chain2 = (
        get_residue_info(chains, ipsae_residue_matrix)
    )
    # n0dom: numres in chain pair with PAEs < cutoff, ie. number of residues contributing to ipSAE calculation
    n0dom = (
        ipsae_nunique_residues_chain1 + ipsae_nunique_residues_chain2
    )  # this is correct
    # d0dom: chain-chain d0 value (should be none for self)
    d0dom = d0_matrix(
        n0dom, chain_type_matrix
    )  # this is correct -- axis 0 is chain 1, axis 1 is chain 2

    n0res = (pae_matrix < pae_cutoff) @ chain_indicator_matrix.T.astype(int)  # correct

    d0res = d0_matrix(
        n0res, chain_indicator_matrix.astype(int).T @ chain_type_matrix
    )  # correct. axis0 = chain1 residue, axis1 = chain2

    ipSAE_d0chn = []
    ipsae_d0dom_byres = []
    ipsae_d0res_byres = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice")

        for i, chain1 in enumerate(unique_chains):
            # shape: unique_chains, residues, len(chain1)
            chain1_masked_ptms = 1.0 / (
                1.0
                + (
                    masked_pae_matrix[np.newaxis, :, chain_indicator_matrix[i]]
                    / d0chn[i, :, np.newaxis, np.newaxis]
                )
                ** 2.0
            )
            ipSAE_d0chn.append(np.nanmean(chain1_masked_ptms, axis=-1))

            # shape: unique_chains, residues, len(chain1)
            chain1_masked_ptms = 1.0 / (
                1.0
                + (
                    masked_pae_matrix[np.newaxis, :, chain_indicator_matrix[i]]
                    / d0dom[:, i, np.newaxis, np.newaxis]
                )
                ** 2.0
            )
            ipsae_d0dom_byres.append(np.nanmean(chain1_masked_ptms, axis=-1))
            chain1_byres_masked_ptms = 1.0 / (
                1.0
                + (
                    masked_pae_matrix[:, chain_indicator_matrix[i], np.newaxis]
                    / d0res[:, np.newaxis, :]
                )
                ** 2.0
            )
            ipsae_d0res_byres.append(
                np.nanmean(chain1_byres_masked_ptms, axis=1)
            )  # chain1 residues, chain2

    ipSAE_d0chn = reshape_vectorized_output(
        np.array(ipSAE_d0chn) * chain_indicator_matrix[None, :, :],
        chain_indicator_matrix,
    )
    ipsae_d0dom_byres = reshape_vectorized_output(
        np.array(ipsae_d0dom_byres) * chain_indicator_matrix[None, :, :],
        chain_indicator_matrix,
    )

    # for reasons I don't totally understand, this has to be handled differently
    ipsae_d0res_byres = np.swapaxes(
        np.array(ipsae_d0res_byres) * np.invert(chain_indicator_matrix).T, 1, 2
    )
    ipsae_d0res_byres = np.moveaxis(
        ipsae_d0res_byres[np.arange(nchains), np.arange(nchains), :], 0, -1
    )
    return {
        "ipsae_d0chn": ipSAE_d0chn,
        "ipsae_d0dom": ipsae_d0dom_byres,
        "ipsae_d0res": ipsae_d0res_byres,
    }


def block_max(chains, input_matrix):
    """Group input_matrix using label vector `chains` and take max over each group.

    `chains`: array-like, shape (r,) marking c unique contiguous and monotonic-increasing blocks
    `input_matrix`: matrix-like, shape (r, c)

    Output shape (c,c)
    """
    # Find boundaries of each contiguous block
    _, start_idx = np.unique(
        chains,
        return_index=True,
    )
    start_idx.sort()  # handle >26 chains case
    start_idx = np.append(start_idx, len(chains))  # add end marker

    # Allocate result array
    out = np.maximum.reduceat(
        np.nan_to_num(input_matrix, nan=0), start_idx[:-1], axis=0
    ).astype(float)
    np.fill_diagonal(out, np.nan)
    return out


def chainlevel_outputs(
    chains,
    reslevel,
):
    """Aggregate residue-level metrics (ipSAE, ipTM, etc.) at chain level."""
    unique_chains = pd.unique(
        chains,
    )
    chainlevel_asym = {
        metric: pd.DataFrame(
            block_max(chains, reslevel[metric]),
            index=unique_chains,
            columns=unique_chains,
        )
        for metric in reslevel
    }
    return chainlevel_asym


def ipsae(residues, chains, pae_matrix, pae_cutoff: float) -> dict:
    """Calculate asymmetric chain-pairwise ipSAE.

    Args:
    :residues: dictionary of residue information pulled from `read_pdb` and similar
    :chains:
    :pae_matrix: (np.array(float)) PAE matrix for AF3 output
    :pae_cutoff: (float) max PAE to consider a residue pair
    """

    # calculate residue-level metrics
    reslevel = ipsae_reslevel(
        chains, np.array([res["res"] for res in residues]), pae_matrix, pae_cutoff
    )
    # calculate chain-level metrics
    return chainlevel_outputs(chains, reslevel)
