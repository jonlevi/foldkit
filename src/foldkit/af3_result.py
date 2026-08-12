"""
foldkit.core
Core functionality for working with AlphaFold3 results.
 
Features:
- Load AF3 results from metadata (.json).
- Interface for global and local confidence metrics, including ipSAE
"""

import json
import re
from typing import Optional, Callable, Generator
from pathlib import Path
import logging
import shutil

import numpy as np
import pandas as pd

from .ipsae import ipsae, read_pdb


class AF3Result:

    residue_chain_ids: Optional[np.ndarray] = None
    atom_chain_ids: Optional[np.ndarray] = None
    chains: Optional[list[str]] = None

    # Confidence data
    plddt: Optional[np.ndarray] = None
    pae: Optional[np.ndarray] = None
    contact_probs: Optional[np.ndarray] = None

    global_ptm: Optional[float] = None
    global_iptm: Optional[float] = None
    chain_pair_iptm: Optional[np.ndarray] = None
    chain_ptm: Optional[np.ndarray] = None

    # Structural data needed for ipSAE
    cif_residues: Optional[np.ndarray] = None
    cif_chains: Optional[np.ndarray] = None

    # Metadata
    id: str = None
    cif_path: Optional[Path] = None
    summary_json_path: Optional[Path] = None
    full_json_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------

    def _build_from_af3_output(
        self, id: str, summary_json_path: Path, full_json_path: Path, cif_path: Path
    ):
        self.id = id
        self.cif_path = Path(cif_path)
        self.summary_json_path = Path(summary_json_path)
        self.full_json_path = Path(full_json_path)

        self._extract_confidences()
        self._load_cif_information()

    def _load_cif_information(self):
        with open(self.cif_path, "r") as f:
            residues, _, _, chains = read_pdb(f, "cif")
            self.cif_residues = residues
            self.cif_chains = chains

    def _extract_confidences(self):
        with open(self.summary_json_path, "r") as jfile1:
            summary_confidence = json.load(jfile1)
            self.global_ptm = float(summary_confidence["ptm"])
            self.global_iptm = float(summary_confidence["iptm"])
            self.chain_pair_iptm = np.array(summary_confidence["chain_pair_iptm"])
            self.chain_ptm = np.array(summary_confidence["chain_ptm"])

        with open(self.full_json_path, "r") as jfile2:
            confidence = json.load(jfile2)
            self.residue_chain_ids = np.array(confidence["token_chain_ids"])
            self.atom_chain_ids = np.array(confidence["atom_chain_ids"])
            # Note: pd.unique instead of np.unique because it preserves order, and so chains are in the right order
            self.chains = list(pd.unique(self.residue_chain_ids))
            self.plddt = np.array(confidence["atom_plddts"])
            self.pae = np.array(confidence["pae"])
            self.contact_probs = np.array(confidence["contact_probs"])

    @staticmethod
    def _from_npz_data(data) -> "AF3Result":
        """Build an AF3Result from an already-loaded .npz archive (shared by
        `load_result` and `load_stored_webserver_results`)."""
        obj = AF3Result()
        obj.id = str(data["id"])
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

    # ------------------------------------------------------------------
    # Shared validation / indexing helpers
    # ------------------------------------------------------------------

    def _validate_chain(self, chain: str) -> None:
        """Raise ValueError if `chain` isn't a known chain."""
        if chain not in self.chains:
            raise ValueError(f"Chain {chain} not in {self.chains}")

    def _chain_residue_indices(self, chain: str) -> list[int]:
        """Residue (token) indices belonging to `chain`."""
        self._validate_chain(chain)
        return [int(idx) for idx in np.where(self.residue_chain_ids == chain)[0]]

    def _chain_atom_indices(self, chain: str) -> list[int]:
        """Atom indices belonging to `chain`."""
        self._validate_chain(chain)
        return [int(idx) for idx in np.where(self.atom_chain_ids == chain)[0]]

    def _validate_pair_selector(
        self,
        chain1: Optional[str],
        chain2: Optional[str],
        tokens1: Optional[list[int]],
        tokens2: Optional[list[int]],
    ) -> tuple[str, tuple]:
        """
        Validate the chain1/chain2 vs. tokens1/tokens2 arguments shared by
        `get_ipae` and `get_contact_probs`, and report which selector is active.

        Returns
        -------
        ("chains", (chain1, chain2)) or ("tokens", (tokens1, tokens2))

        Raises
        ------
        ValueError
            If neither pair is fully specified, both pairs are specified, or
            only one member of a pair is given.
        """
        both_chains = chain1 is not None and chain2 is not None
        one_chain_only = (chain1 is not None) != (chain2 is not None)
        both_tokens = tokens1 is not None and tokens2 is not None
        one_token_only = (tokens1 is not None) != (tokens2 is not None)

        if one_chain_only:
            raise ValueError("Must provide both chains for chain-based aggregation")
        if one_token_only:
            raise ValueError(
                "Must provide both tokens lists for token-based aggregation"
            )
        if not both_chains and not both_tokens:
            raise ValueError(
                "Must provide either chain1 and chain2, or tokens1 and tokens2"
            )
        if both_chains and both_tokens:
            raise ValueError(
                "Cannot provide both `chain` and `tokens` at the same time."
            )

        if both_chains:
            self._validate_chain(chain1)
            self._validate_chain(chain2)
            return "chains", (chain1, chain2)
        return "tokens", (tokens1, tokens2)

    def _pairwise_submatrices(
        self, matrix: np.ndarray, mode: str, selector: tuple
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract the two off-diagonal blocks of `matrix` between a pair of
        chains or a pair of token lists (shared by `get_ipae` and
        `get_contact_probs`).
        """
        if mode == "chains":
            chain1, chain2 = selector
            residues_1 = self._chain_residue_indices(chain1)
            residues_2 = self._chain_residue_indices(chain2)
            sub1 = matrix[
                residues_1[0] : residues_1[-1] + 1, residues_2[0] : residues_2[-1] + 1
            ]
            sub2 = matrix[
                residues_2[0] : residues_2[-1] + 1, residues_1[0] : residues_1[-1] + 1
            ]
        else:  # mode == "tokens"
            tokens1, tokens2 = selector
            sub1 = matrix[np.ix_(tokens1, tokens2)]
            sub2 = matrix[np.ix_(tokens2, tokens1)]
        return sub1, sub2

    # ------------------------------------------------------------------
    # Confidence metrics
    # ------------------------------------------------------------------

    def get_ptm(self, chain: Optional[str] = None) -> float:
        """
        Return the ptm score.

        Parameters
        ----------
        chain : str, optional
            If provided, computes the ptm for this specific chain.
            If None, returns the global score across all chains in the structure.

        Returns
        -------
        float
           ptm score.

        Raises
        ------
        ValueError
            If `chain` is invalid based on the data
        """
        if chain is None:
            return self.global_ptm
        self._validate_chain(chain)
        return float(self.chain_ptm[self.chains.index(chain)])

    def get_iptm(
        self, chain1: Optional[str] = None, chain2: Optional[str] = None
    ) -> float:
        """
        Return the interchain ptm (iptm) score.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the iptm for this specific pair of chains. Otherwise, returns the global iptm score for all pairs of chains

        Returns
        -------
        float
           iptm score.

        Raises
        ------
        ValueError
            If `chain` is invalid based on the data
        """
        if chain1 is None and chain2 is None:
            return self.global_iptm
        if chain1 is None or chain2 is None:
            raise ValueError("Must provide both chains for chain-based aggregation")

        self._validate_chain(chain1)
        self._validate_chain(chain2)
        i = self.chains.index(chain1)
        j = self.chains.index(chain2)
        return float(self.chain_pair_iptm[i][j])

    def get_pae(
        self,
        chain: Optional[str] = None,
        tokens: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the pae (predicted alignment error) score.

        Parameters
        ----------
        chain : str, optional
            If provided, computes the pae for this specific chain.
            If None, returns the score across all tokens (residues) in the structure.

        tokens : list[int], optional
            If provided, computes the pae for specific tokens.
            If None, returns the score across all tokens (residues) in the structure.
            Cannot be used in combination with "chain"

        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.

        Returns
        -------
        float
            Aggregated pae score.

        Raises
        ------
        ValueError
            If both `chain` and `tokens` are provided, or if either argument is invalid based on the data
        """
        if chain is not None and tokens is not None:
            raise ValueError(
                "Cannot provide both `chain` and `tokens` at the same time."
            )

        if chain is not None:
            residues = self._chain_residue_indices(chain)
            sub_pae = self.pae[
                residues[0] : residues[-1] + 1, residues[0] : residues[-1] + 1
            ]
        elif tokens is not None:
            sub_pae = self.pae[tokens, tokens]
        else:
            sub_pae = self.pae

        return agg(sub_pae)

    def get_ipae(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        tokens1: Optional[list[int]] = None,
        tokens2: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the interface pae (predicted alignment error) score.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the ipae for this specific pair of chains.
            Only optional if `tokens1` and `tokens2` are provided

        tokens1, tokens2 : list[int], optional
            If provided, computes the ipae for this specific pair of token lists.
             Only optional if `chain1` and `chain2`  are provided

        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.

        Returns
        -------
        float
            Aggregated ipae score.

        Raises
        ------
        ValueError
            If neither `chain` and `tokens` are provided, both are provided, or if either argument is invalid based on the data
        """
        mode, selector = self._validate_pair_selector(chain1, chain2, tokens1, tokens2)
        sub1, sub2 = self._pairwise_submatrices(self.pae, mode, selector)
        return agg(np.concatenate((sub1, sub2), axis=None))

    def get_contact_probs(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        tokens1: Optional[list[int]] = None,
        tokens2: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the contact probs score.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the contact probs for this specific pair of chains.
            Only optional if `tokens1` and `tokens2` are provided

        tokens1, tokens2 : list[int], optional
            If provided, computes the contact probs for this specific pair of token lists.
             Only optional if `chain1` and `chain2`  are provided

        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.

        Returns
        -------
        float
            Aggregated contact probs score.

        Raises
        ------
        ValueError
            If neither `chain` and `tokens` are provided, both are provided, or if either argument is invalid based on the data
        """
        mode, selector = self._validate_pair_selector(chain1, chain2, tokens1, tokens2)
        sub1, sub2 = self._pairwise_submatrices(self.contact_probs, mode, selector)
        return agg(np.concatenate((sub1, sub2), axis=None))

    def get_plddt(
        self,
        chain: Optional[str] = None,
        atoms: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the plddt score.

        Parameters
        ----------
        chain : str, optional
            If provided, computes the plddt for this specific chain.
            If None, returns the score across all tokens (residues) in the structure.

        atoms : list[int], optional
            If provided, computes the plddt for specific atoms.
            If None, returns the score across all atoms in the structure.
            Cannot be used in combination with "chain"

        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.

        Returns
        -------
        float
            Aggregated plddt score.

        Raises
        ------
        ValueError
            If both `chain` and `atoms` are provided, or if either argument is invalid based on the data
        """
        if chain is not None and atoms is not None:
            raise ValueError(
                "Cannot provide both `chain` and `atoms` at the same time."
            )

        if chain is not None:
            atoms = self._chain_atom_indices(chain)

        sub_plddt = self.plddt if atoms is None else self.plddt[atoms]
        return agg(sub_plddt)

    def get_ipsae(
        self, chain1: str, chain2: str, pae_cutoff: float = 10, d0="d0res"
    ) -> float:
        """
        Return the asymmetric ipSAE score.

        Parameters
        ----------
        chain1, chain2 : str
            Computes the ipSAE for this specific pair of chains.
            Note that the score is asymmetric so the order of chains matters.
        pae_cutoff : float, default=10
            PAE cutoff (in Angstroms) passed through to the ipSAE calculation.
        d0: str, default=d0res
            d0 used for calculation. Options are:
            d0res: d0 determined by number of residues in 2nd chain with PAE<cutoff
            d0chn: sum of chain lengths
            d0dom: d0 = total number of residues in both chains with any interchain PAE<cutoff

        Returns
        -------
        float
            ipSAE score.
        """
        cif_chains_unique = pd.unique(self.cif_chains)
        assert (
            chain1 in cif_chains_unique
        ), f"{chain1} not found in .cif chains: {cif_chains_unique}"
        assert (
            chain2 in cif_chains_unique
        ), f"{chain2} not found in .cif chains: {cif_chains_unique}"
        assert chain1 != chain2, "ipSAE is undefined for the case chain1==chain2"
        d0_values = ["d0res", "d0dom", "d0chn"]
        assert d0 in d0_values, f"Invalid d0 value {d0}. Valid choices are: {d0_values}"

        ipsae_dict = ipsae(
            residues=self.cif_residues,
            chains=self.cif_chains,
            pae_matrix=self.pae,
            pae_cutoff=pae_cutoff,
        )
        key = f"ipsae_{d0}"
        ipsae_df = ipsae_dict.get(key)
        return ipsae_df.loc[chain1, chain2]

    # ------------------------------------------------------------------
    # Object Creation Factory Methods
    # ------------------------------------------------------------------

    @staticmethod
    def load_af3_result(result_dir: str, id: Optional[str] = None) -> "AF3Result":
        """
        Loads a new single AF3 Result from AF3 output directory (`result_dir`), with optional name (`id`)
        """

        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(
                f"Result directory not found: {result_dir} does not exist or is not a directory"
            )

        # Find CIF
        cif_files = list(result_dir.glob("*.cif"))
        if len(cif_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *.cif file, found {len(cif_files)}"
            )
        cif_file = cif_files[0]

        # Find summary JSON
        summary_files = list(result_dir.glob("*summary_confidences.json"))
        if len(summary_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *summary_confidences.json, found {len(summary_files)}. For data in .npz format, use flag 'from_npz=True'."
            )

        # Find full JSON
        full_files = [
            f for f in result_dir.glob("*confidences.json") if not "summary" in f.name
        ]
        if len(full_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one confidences.json (non-summary), found {len(full_files)}"
            )

        if not id:
            id = Path(result_dir).name

        res = AF3Result()
        res._build_from_af3_output(
            id=id,
            summary_json_path=summary_files[0],
            full_json_path=full_files[0],
            cif_path=cif_file,
        )
        return res

    @staticmethod
    def load_webserver_result(
        result_dir: str,
        sample: int,
        seed: Optional[int] = None,
        id_stem: Optional[str] = None,
    ) -> "AF3Result":
        """Generate an AF3Result file for one of the 5 samples from a run on the AF3 webserver.

        Inputs:
        result_dir: path to directory storing results.
        sample: which sample to load
        seed: Model seed. If unset, will try to infer these from a request file in result_dir.
        id_stem: Job name. If unset, will try to infer these from a request file in result_dir.
        """
        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(
                f"Result directory not found: {result_dir} does not exist or is not a directory"
            )

        summary_files = list(result_dir.glob("*summary_confidences*.json"))

        if seed is None or id_stem is None:
            request_files = list(result_dir.glob("*request*.json"))
            if len(request_files) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one request.json, found {len(request_files)}."
                )
            with open(request_files[0], "r") as f:
                request_data = json.load(f)
            if len(request_data) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one request inside the JSON, found {len(request_data)}."
                )
            request = request_data[0]
            id_stem = request["name"] if id_stem is None else id_stem
            seed = request["modelSeeds"][0] if seed is None else seed

        num_samples = len(summary_files)  # usually this is 5
        if sample >= num_samples:
            raise ValueError(
                f"Sample {sample} not found. Job was only run with {num_samples} and is 0-indexed."
            )

        # Find full JSON
        full_files = list(result_dir.glob("*full_data*.json"))
        if len(full_files) != num_samples:
            raise FileNotFoundError(
                f"Expected exactly {num_samples} full_data*.json (non-summary), found {len(full_files)}"
            )
        structure_files = list(result_dir.glob("*.cif"))
        if len(structure_files) != num_samples:
            raise FileNotFoundError(
                f"Expected exactly {num_samples} *.cif files, found {len(structure_files)}"
            )

        for summary_file, full_file, structure_file in zip(
            sorted(summary_files), sorted(full_files), sorted(structure_files)
        ):
            match = re.search(r"summary_confidences_(\d)\.json", summary_file.name)
            if not match:
                raise ValueError(
                    f"Improperly formatted summary_confidences filename:\n{summary_file.name}"
                )
            sample_num = int(match.group(1))

            if sample_num != sample:
                continue

            sample_stem = f"{id_stem}_seed{seed}_sample{sample_num}"
            res = AF3Result()
            res._build_from_af3_output(
                id=sample_stem,
                summary_json_path=summary_file,
                full_json_path=full_file,
                cif_path=structure_file,
            )
            return res

        raise FileNotFoundError(
            f"Sample {sample} not found among parsed summary_confidences files in {result_dir}."
        )

    @staticmethod
    def load_compressed_result(
        result_dir: str,
    ) -> "AF3Result":

        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(
                f"Result directory not found: {result_dir} does not exist or is not a directory"
            )

        # Find npz
        npz_files = list(result_dir.glob("*.npz"))
        if len(npz_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *.npz, found {len(npz_files)}. For data not in compressed format, use regular load function."
            )
        # Find CIF
        cif_files = list(result_dir.glob("*.cif"))
        if len(cif_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *.cif file, found {len(cif_files)}"
            )
        cif_file = cif_files[0]

        data = np.load(npz_files[0], allow_pickle=True)
        obj = AF3Result._from_npz_data(data)
        obj.cif_path = Path(cif_file)
        obj._load_cif_information()
        return obj
