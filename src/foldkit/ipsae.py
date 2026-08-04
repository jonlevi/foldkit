import sys
import os
import math
import json
import logging
import warnings
import gzip
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


# Define the ptm and d0 functions
def ptm_func(x, d0):
    return 1.0 / (1 + (x / d0) ** 2.0)


ptm_func_vec = np.vectorize(ptm_func)  # vector version


# Define the d0 functions for numbers and arrays; minimum value = 1.0; from Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702–710 (2004)
def calc_d0(L, pair_type):
    L = float(L)
    if L < 27:
        L = 27
    min_value = 1.0
    if pair_type == "nucleic_acid":
        min_value = 2.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return max(min_value, d0)


def d0_matrix(chain_length_sum_matrix, chain_type_matrix):
    # minimum value for d0: 1 for protein, 2 for NA
    twos = np.invert(chain_type_matrix.astype(bool)).astype(int) * 2
    min_d0_matrix = chain_type_matrix + twos
    L = np.maximum(27, chain_length_sum_matrix)
    return np.maximum(min_d0_matrix, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


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


# Function for printing out residue numbers in PyMOL scripts
def contiguous_ranges(numbers):
    if not numbers:  # Check if the set is empty
        return

    sorted_numbers = sorted(numbers)  # Sort the numbers
    start = sorted_numbers[0]
    end = start
    ranges = []  # List to store ranges

    def format_range(start, end):
        if start == end:
            return f"{start}"
        else:
            return f"{start}-{end}"

    for number in sorted_numbers[1:]:
        if number == end + 1:
            end = number
        else:
            ranges.append(format_range(start, end))
            start = end = number

    # Append the last range after the loop
    ranges.append(format_range(start, end))

    # Join all ranges with a plus sign and print the result
    string = "+".join(ranges)
    return string


# Initializes a nested dictionary with all values set to 0
def init_chainpairdict_zeros(chainlist):
    return {
        chain1: {chain2: 0 for chain2 in chainlist if chain1 != chain2}
        for chain1 in chainlist
    }


# Initializes a nested dictionary with NumPy arrays of zeros of a specified size
def init_chainpairdict_npzeros(chainlist, arraysize):
    return {
        chain1: {
            chain2: np.zeros(arraysize) for chain2 in chainlist if chain1 != chain2
        }
        for chain1 in chainlist
    }


# Initializes a nested dictionary with empty sets.
def init_chainpairdict_set(chainlist):
    return {
        chain1: {chain2: set() for chain2 in chainlist if chain1 != chain2}
        for chain1 in chainlist
    }


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
        # chain_types[chain] = "nucleic_acid" if nuc_count > 0 else "protein"
        chain_types.append(
            int(sum([residue in NUC_RESIDUE_SET for residue in chain_residues]) > 0)
        )

    # calculate is_nucleic for each chain, then cross and do OR
    chain_types = np.array([chain_types] * len(unique_chains))
    return np.invert(np.bitwise_or(chain_types, chain_types.T))


def load_af2_data(pae_file, numres):
    if pae_file.name.endswith(".pkl"):
        data = np.load(pae_file, allow_pickle=True)
    elif pae_file.name.endswith(".gz"):
        with gzip.open(pae_file.name, "r") as f:
            data = json.load(f)
    else:
        data = json.load(pae_file)

    if "iptm" in data:
        iptm = float(data["iptm"])
    else:
        iptm = -1.0
    if "ptm" in data:
        ptm = float(data["ptm"])
    else:
        ptm = -1.0

    if "plddt" in data:
        plddt = np.array(data["plddt"])
        cb_plddt = np.array(data["plddt"])  # for pDockQ
    else:
        plddt = np.zeros(numres)
        cb_plddt = np.zeros(numres)

    if "pae" in data:
        pae_matrix = np.array(data["pae"])
    elif "predicted_aligned_error" in data:
        pae_matrix = np.array(data["predicted_aligned_error"])

    return pae_matrix, plddt, cb_plddt, iptm


def load_boltz1_data(pae_file, token_array, unique_chains):
    # Boltz1 filenames:
    # AURKA_TPX2_model_0.cif
    # confidence_AURKA_TPX2_model_0.json
    # pae_AURKA_TPX2_model_0.npz
    # plddt_AURKA_TPX2_model_0.npz

    plddt_file_path = pae_file.name.replace("pae", "plddt")
    if os.path.exists(plddt_file_path):
        data_plddt = np.load(plddt_file_path)
        plddt_boltz1 = np.array(100.0 * data_plddt["plddt"])
        plddt = plddt_boltz1[np.ix_(token_array.astype(bool))]
        cb_plddt = plddt_boltz1[np.ix_(token_array.astype(bool))]
    else:
        ntokens = np.sum(token_array)
        plddt = np.zeros(ntokens)
        cb_plddt = np.zeros(ntokens)

    data_pae = np.load(pae_file.name)
    pae_matrix_boltz1 = np.array(data_pae["pae"])
    pae_matrix = pae_matrix_boltz1[
        np.ix_(token_array.astype(bool), token_array.astype(bool))
    ]

    summary_file_path = pae_file.name.replace("pae", "confidence")
    summary_file_path = summary_file_path.replace(".npz", ".json")
    iptm = {
        chain1: {chain2: 0 for chain2 in unique_chains if chain1 != chain2}
        for chain1 in unique_chains
    }
    with open(summary_file_path, "r") as file:
        data_summary = json.load(file)

        boltz1_chain_pair_iptm_data = data_summary["pair_chains_iptm"]
        for chain1 in unique_chains:
            nchain1 = ord(chain1) - ord("A")  # map A,B,C... to 0,1,2...
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue
                nchain2 = ord(chain2) - ord("A")
                iptm[chain1][chain2] = boltz1_chain_pair_iptm_data[str(nchain1)][
                    str(nchain2)
                ]

    return pae_matrix, plddt, cb_plddt, iptm


def load_af3_data(
    pae_file, residues, cb_residues, token_mask, unique_chains, summary_file=None
):
    # Example Alphafold3 server filenames
    #   fold_aurka_0_tpx2_0_full_data_0.json
    #   fold_aurka_0_tpx2_0_summary_confidences_0.json
    #   fold_aurka_0_tpx2_0_model_0.cif
    # Example AlphaFold3 downloadable code filenames
    #   confidences.json
    #   summary_confidences.json
    #   model1.cif
    data = json.load(pae_file)
    atom_plddts = np.array(data["atom_plddts"])
    CA_atom_num = np.array(
        [res["atom_num"] - 1 for res in residues]
    )  # for AF3 atom indexing from 0
    CB_atom_num = np.array(
        [res["atom_num"] - 1 for res in cb_residues]
    )  # for AF3 atom indexing from 0
    plddt = atom_plddts[CA_atom_num]  # pull out residue plddts from Calpha atoms
    cb_plddt = atom_plddts[
        CB_atom_num
    ]  # pull out residue plddts from Cbeta atoms for pDockQ

    # Get pairwise residue PAE matrix by identifying one token per protein residue.
    # Modified residues have separate tokens for each atom, so need to pull out Calpha atom as token
    # Skip ligands
    if "pae" in data:
        pae_matrix_af3 = np.array(data["pae"])
    else:
        logging.info("no PAE data in AF3 json file; quitting")
        sys.exit()

    # Set pae_matrix for AF3 from subset of full PAE matrix from json file
    token_array = np.array(token_mask)
    pae_matrix = pae_matrix_af3[
        np.ix_(token_array.astype(bool), token_array.astype(bool))
    ]
    # Get iptm matrix from AF3 summary_confidences file
    iptm = {
        chain1: {chain2: 0 for chain2 in unique_chains if chain1 != chain2}
        for chain1 in unique_chains
    }

    if summary_file is not None:  # todo: this will break for more than 26 chains
        data_summary = json.load(summary_file)
        af3_chain_pair_iptm_data = data_summary["chain_pair_iptm"]
        for chain1 in unique_chains:
            nchain1 = ord(chain1) - ord("A")  # map A,B,C... to 0,1,2...
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue
                nchain2 = ord(chain2) - ord("A")
                iptm[chain1][chain2] = af3_chain_pair_iptm_data[nchain1][nchain2]

    return pae_matrix, plddt, cb_plddt, iptm


def _grouped_nunique_offdiag_sum(pair_matrix, group_starts):
    """Get sum of active rows and columns in contiguous groups."""

    row_x_cat = np.add.reduceat(pair_matrix, group_starts, axis=1) > 0
    row_group_sum = np.add.reduceat(row_x_cat, group_starts, axis=0)
    row_group_sum = row_group_sum.astype(int)
    np.fill_diagonal(row_group_sum, 0)

    col_x_cat = np.add.reduceat(pair_matrix, group_starts, axis=0) > 0
    col_x_cat = col_x_cat.transpose()
    col_group_sum = np.add.reduceat(col_x_cat, group_starts, axis=0)
    col_group_sum = col_group_sum.T.astype(int)
    np.fill_diagonal(col_group_sum, 0)
    return row_group_sum, col_group_sum


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
            (atomsite, fieldname) = line.split(".")
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


def calculate_distances(cb_residues):
    logging.info("Calculating distance matrix")
    # Calculate distance matrix using NumPy broadcasting
    coordinates = np.array([res["coor"] for res in cb_residues])
    return np.sqrt(
        ((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]) ** 2).sum(
            axis=2
        )
    )


def calculate_external_metrics(
    chains, unique_chains, distances, pae_matrix, cb_plddt, numres
):
    """pDockQ[2], LIS"""
    pDockQ = init_chainpairdict_zeros(unique_chains)
    pDockQ2 = init_chainpairdict_zeros(unique_chains)
    LIS = init_chainpairdict_zeros(unique_chains)
    pDockQ_unique_residues = init_chainpairdict_set(unique_chains)

    # pDockQ
    logging.info("Calculating pDockQ scores")
    pDockQ_cutoff = 8.0

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = (chains == chain2) & (distances[i] <= pDockQ_cutoff)
                npairs += np.sum(valid_pairs)
                if valid_pairs.any():
                    pDockQ_unique_residues[chain1][chain2].add(i)
                    chain2residues = np.where(valid_pairs)[0]

                    for residue in chain2residues:
                        pDockQ_unique_residues[chain1][chain2].add(residue)

            if npairs > 0:
                nres = len(list(pDockQ_unique_residues[chain1][chain2]))
                mean_plddt = cb_plddt[
                    list(pDockQ_unique_residues[chain1][chain2])
                ].mean()
                x = mean_plddt * math.log10(npairs)
                pDockQ[chain1][chain2] = (
                    0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
                )
            else:
                mean_plddt = 0.0
                x = 0.0
                pDockQ[chain1][chain2] = 0.0
                nres = 0

    # pDockQ2
    logging.info("Calculating pDockQ2 scores")
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            sum = 0.0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = (chains == chain2) & (distances[i] <= pDockQ_cutoff)
                if valid_pairs.any():
                    npairs += np.sum(valid_pairs)
                    pae_list = pae_matrix[i][valid_pairs]
                    pae_list_ptm = ptm_func_vec(pae_list, 10.0)
                    sum += pae_list_ptm.sum()

            if npairs > 0:
                nres = len(list(pDockQ_unique_residues[chain1][chain2]))
                mean_plddt = cb_plddt[
                    list(pDockQ_unique_residues[chain1][chain2])
                ].mean()
                mean_ptm = sum / npairs
                x = mean_plddt * mean_ptm
                pDockQ2[chain1][chain2] = (
                    1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
                )
            else:
                mean_plddt = 0.0
                x = 0.0
                nres = 0
                pDockQ2[chain1][chain2] = 0.0

    # LIS
    logging.info("Calculating LIS scores")
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            mask = (chains[:, None] == chain1) & (
                chains[None, :] == chain2
            )  # Select residues for (chain1, chain2)
            selected_pae = pae_matrix[mask]  # Get PAE values for this pair

            if selected_pae.size > 0:  # Ensure we have values
                valid_pae = selected_pae[selected_pae <= 12]  # Apply the threshold
                if valid_pae.size > 0:
                    scores = (12 - valid_pae) / 12  # Compute scores
                    avg_score = np.mean(scores)  # Average score for (chain1, chain2)
                    LIS[chain1][chain2] = avg_score
                else:
                    LIS[chain1][chain2] = 0.0  # No valid values
            else:
                LIS[chain1][chain2] = 0.0

    return pDockQ, pDockQ2, LIS


def configure(pdb_path, pae_file_path, pae_cutoff, dist_cutoff):
    """Handle arguments from command line."""
    pae_string = f"{'0' if pae_cutoff < 10 else ''}{int(pae_cutoff)}"
    dist_string = f"{'0' if dist_cutoff < 10 else ''}{int(dist_cutoff)}"

    if ".pdb" in pdb_path:
        pdb_stem = pdb_path.replace(".pdb", "")
        path_stem = f"{pdb_stem}_{pae_string}_{dist_string}"
        algorithm = "af2"
        protein_file_type = "pdb"
    elif ".cif" in pdb_path:
        pdb_stem = pdb_path.replace(".cif", "")
        path_stem = f"{pdb_stem}_{pae_string}_{dist_string}"
        protein_file_type = "cif"
        if pae_file_path.endswith(".json"):
            algorithm = "af3"
        elif pae_file_path.endswith(".npz"):
            algorithm = "boltz1"
        else:
            raise ValueError(
                f"PAE file {pae_file_path} extension is incorrect, must be one of (.json, .npz)"
            )
    else:
        raise ValueError(
            f"PDB file {pdb_path} extension is incorrect, must be one of (.pdb, .cif)"
        )

    return algorithm, protein_file_type, pdb_stem, path_stem, pae_string, dist_string


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


def ipsae_reslevel(chains, residue_types, pae_matrix, pae_cutoff):
    """"""
    # get chain matrices
    # unique_chains, unique_chain_lengths = np.unique(
    #     chains, return_counts=True, sorted=False
    # )
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
    # logging.debug("n0dom, d0dom, n0res, d0res")
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

    # logging.debug("Calculating residue-level ipSAE in four flavors")
    # ipTM_pae = []
    ipSAE_d0chn = []
    ipsae_d0dom_byres = []
    ipsae_d0res_byres = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice")

        for i, chain1 in enumerate(unique_chains):
            # shape: unique_chains, residues, len(chain1)
            # chain1_ptms = 1 / (1 + pae/d0chn)^2
            # chain1_ptms = 1.0 / (
            #     1.0
            #     + (
            #         pae_matrix[np.newaxis, :, chain_indicator_matrix[i]]
            #         / d0chn[i, :, np.newaxis, np.newaxis]
            #     )
            #     ** 2.0
            # )
            # ipTM_pae.append(chain1_ptms.mean(axis=-1))
            # now do it again, with the masked PAE matrix
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
            # now do it again, with the masked PAE matrix
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

    # ipTM_pae = reshape_vectorized_output(
    #     np.array(ipTM_pae) * chain_indicator_matrix[None, :, :], chain_indicator_matrix
    # )
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
        # "iptm_d0chn": ipTM_pae,
        "ipsae_d0chn": ipSAE_d0chn,
        "ipsae_d0dom": ipsae_d0dom_byres,
        "ipsae_d0res": ipsae_d0res_byres,
    }


def reshape_vectorized_output(vectorized_output, chain_indicator_matrix):
    c, _, r = vectorized_output.shape
    cat = np.argmax(chain_indicator_matrix, axis=0)
    X_prime = vectorized_output[
        np.arange(c)[:, None], cat[None, :], np.arange(r)[None, :]
    ]
    return np.moveaxis(X_prime, 0, -1)  # shape (r, c)


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
    start_idx.sort() # handle >26 chains case
    start_idx = np.append(start_idx, len(chains))  # add end marker

    # Allocate result array
    out = np.maximum.reduceat(
        np.nan_to_num(input_matrix, nan=0), start_idx[:-1], axis=0
    ).astype(float)
    np.fill_diagonal(out, np.nan)
    return out


def robust_argmax(block):
    block_mask = np.all(np.isnan(block), axis=0)
    return np.where(
        block_mask,
        np.nan,
        np.nanargmax(
            np.where(
                np.broadcast_to(block_mask, block.shape), np.zeros_like(block), block
            ),
            axis=0,
        ),
    )


def block_argmax(chains, input_matrix, argmax_method=robust_argmax):
    """Group input_matrix using label vector `chains` and return index of each group's max.

    `chains`: array-like, shape (r,) marking c unique contiguous and monotonic-increasing blocks
    `input_matrix`: matrix-like, shape (r, c)

    Output shape (c,c)
    """
    # Get category block boundaries
    cats, start_idx = np.unique(
        chains,
        return_index=True,
    )
    start_idx = np.append(start_idx, len(chains))  # add sentinel at end

    # Initialize array for argmax row indices
    argmax_idx = np.empty((len(cats), len(cats)), dtype=float)

    # Compute argmax per category and per column
    for i in range(len(cats)):
        s, e = start_idx[i], start_idx[i + 1]
        block = input_matrix[s:e]
        argmax_idx[i] = s + argmax_method(block)

    return argmax_idx


def symmetric_upper_max(X):
    # Ensure float dtype so np.nan works
    X = np.asarray(X, dtype=float)

    # Elementwise maximum with transpose
    Xp = np.maximum(X, X.T)

    # Mask out lower triangle and diagonal
    mask = np.triu(np.ones_like(Xp, dtype=bool), k=1)
    Xp[~mask] = np.nan

    return Xp


def chainlevel_outputs(
    chains,
    reslevel,
):
    """Aggregate residue-level metrics (ipSAE, ipTM, etc.) at chain level."""
    # logging.debug("Calculating chain-level outputs")
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
    chainlevel_max = {
        metric: pd.DataFrame(
            symmetric_upper_max(asym_matrix), index=unique_chains, columns=unique_chains
        )
        for metric, asym_matrix in chainlevel_asym.items()
    }

    # # not currently saved, WIP
    # ipsae_d0res_byres_maxindex = np.nan_to_num(
    #     block_argmax(chains, reslevel["ipsae_d0res_byres"], np.argmax), nan=0
    # ).astype(int)
    # n0res = n0res[
    #     ipsae_d0res_byres_maxindex,
    #     np.arange(len(chains)),
    # ]
    # d0res = d0res[
    #     ipsae_d0res_byres_maxindex,
    #     np.arange(len(chains)),
    # ]

    return chainlevel_asym, chainlevel_max


def setup(
    pdb_file: Iterable[str],
    pae_file: Iterable[str],
    protein_file_type,
    algorithm,
):
    """Format inputs.

    `pdb_file` and `pae_file` must be I/O stream objects that allow reading.
    """

    # read pdb file
    residues, cb_residues, token_mask, chains = read_pdb(pdb_file, protein_file_type)

    # handle chain uniqueness
    last_chain_label = ""
    unique_chain_index = -1
    new_chains = []
    for chain_label in chains:
        if chain_label != last_chain_label:
            unique_chain_index += 1
            last_chain_label = chain_label
        new_chains.append(unique_chain_index)
    chains = np.array(new_chains)

    logging.info(
        f"Running ipsae with {len(residues)} residues and {len(np.unique(chains))} chains"
    )
    # calculate distances
    distances = calculate_distances(cb_residues)

    # load PAE matrices
    logging.info("Loading PAE data")
    # Load AF2, AF3, or BOLTZ1 data and extract plddt and pae_matrix (and ptm_matrix if available)
    if algorithm == "af2":
        pae_matrix, plddt, cb_plddt, iptm = load_af2_data(pae_file, len(residues))
    elif algorithm == "boltz1":
        pae_matrix, plddt, cb_plddt, iptm = load_boltz1_data(
            pae_file,
            np.array(token_mask),
            np.unique(
                chains,
            ),
        )
    elif algorithm == "af3":
        pae_matrix, plddt, cb_plddt, iptm = load_af3_data(
            pae_file,
            residues,
            cb_residues,
            token_mask,
            np.unique(
                chains,
            ),
        )

    return residues, chains, distances, pae_matrix, cb_plddt


def externals(chains, distances, pae_matrix, cb_plddt, residues):
    """Calculate chain-pairwise external metrics for all chain pairs.
    
    TODO: figure out which output format makes the most sense. 
    (I don't currently use these and have not optimized the internal code at all.)
    """

    # calculate pDockQ and friends
    pDockQ, pDockQ2, LIS = calculate_external_metrics(
        chains, np.unique(chains), distances, pae_matrix, cb_plddt, len(residues)
    )
    return pDockQ, pDockQ2, LIS
    


def ipsae(residues, chains, pae_matrix, pae_cutoff: float):
    """Calculate chain-pairwise ipSAE.

    Args:
    :residues: dictionary of residue information pulled from `read_pdb` and similar
    :chains: 
    :pae_matrix: (np.array(float)) PAE matrix for AF3 output
    :pae_cutoff: (float) max PAE to consider a residue pair
    """
    # logging.debug(f"calculating ipSAE on input with ntokens = {len(residues)}")

    # calculate residue-level metrics
    reslevel = ipsae_reslevel(
        chains, np.array([res["res"] for res in residues]), pae_matrix, pae_cutoff
    )
    chain_columns = pd.unique(
        chains,
    )
    residue_index = pd.MultiIndex.from_tuples([(r['chainid'],r['res'],r['resnum']) for r in residues])
    # logging.debug("residue-level complete, aggregating results")
    # calculate chain-level metrics
    return (
        chainlevel_outputs(chains, reslevel), 
        { name: pd.DataFrame(
                res_array, 
                index=residue_index, 
                columns=chain_columns
            ) for name, res_array in reslevel.items()
        }
    )