"""
foldkit.core
Core functionality for working with AlphaFold3 results.

Features:
- Load AF3 results from structure (.cif) and metadata (.json).
- Interface for global and local confidence metrics
- Interface for predicted structure (TODO: coming soon!)
"""

import json
from typing import Optional, Callable
from pathlib import Path
import numpy as np
import pandas as pd


class AF3Result:
    # Confidence data
    chains: Optional[list[str]] = None
    plddt: Optional[np.ndarray] = None
    pae: Optional[np.ndarray] = None
    contact_probs: Optional[np.ndarray] = None

    global_ptm: Optional[float] = None
    global_iptm: Optional[float] = None
    chain_pair_iptm: Optional[np.ndarray] = None
    chain_ptm: Optional[np.ndarray] = None

    # Metadata
    id: str = None
    cif_path: Path = None
    summary_json_path: Path = None
    full_json_path: Path = None

    def _build_from_af3_output(
        self, id: str, cif_path: Path, summary_json_path: Path, full_json_path: Path
    ):
        self.id = id
        self.cif_path = Path(cif_path)
        self.summary_json_path = Path(summary_json_path)
        self.full_json_path = Path(full_json_path)

        # TODO: Access structures
        # self.residues: List[ResidueConfidence] = []
        # self.chain_residue_idx: Dict[str, List[int]] = {}

        # Load data immediately
        # TODO: self._load_cif()
        self._extract_confidences()

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

    def get_ptm(self, chain: Optional[str] = None) -> float:
        """
        Return the ptm score.

        Parameters
        ----------
        chain : str, optional
            If provided, computes the pae for this specific chain.
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

        if chain is not None:
            if chain not in self.chains:
                raise ValueError(f"Chain {chain} not in {self.chains}")

            index = self.chains.index(chain)
            return float(self.chain_ptm[index])

        else:
            return self.global_ptm

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

        both_chains_present = chain1 is not None and chain2 is not None
        chain_one_only = chain1 is not None and chain2 is None
        chain_two_only = chain1 is None and chain2 is not None
        neither_chains_present = chain1 is None and chain2 is None

        if neither_chains_present:
            return self.global_iptm

        if chain_one_only or chain_two_only:
            raise ValueError("Must provide both chains for chain-based aggregation")

        if both_chains_present:
            if chain1 not in self.chains:
                raise ValueError(f"Chain {chain1} not in {self.chains}")
            if chain2 not in self.chains:
                raise ValueError(f"Chain {chain2} not in {self.chains}")

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
            if chain not in self.chains:
                raise ValueError(f"Chain {chain} not in {self.chains}")

            residues = [
                int(idx) for idx in np.where(self.residue_chain_ids == chain)[0]
            ]
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
        Return the pae (predicted alignment error) score.

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

        both_chains_present = chain1 is not None and chain2 is not None
        chain_one_only = chain1 is not None and chain2 is None
        chain_two_only = chain1 is None and chain2 is not None
        neither_chains_present = chain1 is None and chain2 is None

        both_tokens_present = tokens1 is not None and tokens2 is not None
        tokens_one_only = tokens1 is not None and tokens2 is None
        tokens_two_only = tokens1 is None and tokens2 is not None
        neither_tokens_present = tokens1 is None and tokens2 is None

        if neither_chains_present and neither_tokens_present:
            raise ValueError(
                "Must provide either chain1 and chain2, or tokens1 and tokens2"
            )

        if chain_one_only or chain_two_only:
            raise ValueError("Must provide both chains for chain-based aggregation")

        if tokens_one_only or tokens_two_only:
            raise ValueError(
                "Must provide both tokens lists for token-based aggregation"
            )

        if (both_chains_present and not neither_tokens_present) or (
            both_tokens_present and not neither_chains_present
        ):
            raise ValueError(
                "Cannot provide both `chain` and `tokens` at the same time."
            )

        if both_chains_present:
            if chain1 not in self.chains:
                raise ValueError(f"Chain {chain1} not in {self.chains}")
            if chain2 not in self.chains:
                raise ValueError(f"Chain {chain2} not in {self.chains}")

            residues_1 = [
                int(idx) for idx in np.where(self.residue_chain_ids == chain1)[0]
            ]
            residues_2 = [
                int(idx) for idx in np.where(self.residue_chain_ids == chain2)[0]
            ]
            sub_pae1 = self.pae[
                residues_1[0] : residues_1[-1] + 1, residues_2[0] : residues_2[-1] + 1
            ]
            sub_pae2 = self.pae[
                residues_2[0] : residues_2[-1] + 1, residues_1[0] : residues_1[-1] + 1
            ]

        else:
            sub_pae1 = self.pae[tokens1, tokens2]
            sub_pae2 = self.pae[tokens2, tokens1]

        pae_submatrix = np.concatenate((sub_pae1, sub_pae2), axis=None)
        return agg(pae_submatrix)

    def get_contact_probs(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        tokens1: Optional[list[int]] = None,
        tokens2: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.max,
    ) -> float:
        """
        Return the contact probs score.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the contact probs for this specific pair of chains.
            Only optional if `tokens1` and `tokens2` are provided

        tokens1, tokens2 : list[int], optional
            If provided, computes the ipae for this specific pair of token lists.
             Only optional if `chain1` and `chain2`  are provided

        agg : callable, default=np.max
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

        both_chains_present = chain1 is not None and chain2 is not None
        chain_one_only = chain1 is not None and chain2 is None
        chain_two_only = chain1 is None and chain2 is not None
        neither_chains_present = chain1 is None and chain2 is None

        both_tokens_present = tokens1 is not None and tokens2 is not None
        tokens_one_only = tokens1 is not None and tokens2 is None
        tokens_two_only = tokens1 is None and tokens2 is not None
        neither_tokens_present = tokens1 is None and tokens2 is None

        if neither_chains_present and neither_tokens_present:
            raise ValueError(
                "Must provide either chain1 and chain2, or tokens1 and tokens2"
            )

        if chain_one_only or chain_two_only:
            raise ValueError("Must provide both chains for chain-based aggregation")

        if tokens_one_only or tokens_two_only:
            raise ValueError(
                "Must provide both tokens lists for token-based aggregation"
            )

        if (both_chains_present and not neither_tokens_present) or (
            both_tokens_present and not neither_chains_present
        ):
            raise ValueError(
                "Cannot provide both `chain` and `tokens` at the same time."
            )

        if both_chains_present:
            if chain1 not in self.chains:
                raise ValueError(f"Chain {chain1} not in {self.chains}")
            if chain2 not in self.chains:
                raise ValueError(f"Chain {chain2} not in {self.chains}")

            residues_1 = [
                int(idx) for idx in np.where(self.residue_chain_ids == chain1)[0]
            ]
            residues_2 = [
                int(idx) for idx in np.where(self.residue_chain_ids == chain2)[0]
            ]
            sub_contacts1 = self.contact_probs[
                residues_1[0] : residues_1[-1] + 1, residues_2[0] : residues_2[-1] + 1
            ]
            sub_contacts2 = self.contact_probs[
                residues_2[0] : residues_2[-1] + 1, residues_1[0] : residues_1[-1] + 1
            ]

        else:
            sub_contacts1 = self.pae[tokens1, tokens2]
            sub_contacts2 = self.pae[tokens2, tokens1]

        contacts_submatrix = np.concatenate((sub_contacts1, sub_contacts2), axis=None)
        return agg(contacts_submatrix)

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
            If provided, computes the pae for this specific chain.
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
            if chain not in self.chains:
                raise ValueError(f"Chain {chain} not in {self.chains}")

            atoms = [int(idx) for idx in np.where(self.atom_chain_ids == chain)[0]]
            sub_plddt = self.plddt[atoms]

        elif atoms is not None:
            sub_plddt = self.plddt[atoms]

        else:
            sub_plddt = self.plddt

        return agg(sub_plddt)

    # --------------------
    # Object Creation Factory Method
    # see https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md
    # --------------------
    @staticmethod
    def load_result(result_dir: str, id: Optional[str] = None) -> "AF3Result":

        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Result directory not found: {result_dir}")

        # Find CIF
        cif_files = list(result_dir.glob("*.cif"))
        if len(cif_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *.cif file, found {len(cif_files)}"
            )

        # Find summary JSON
        summary_files = list(result_dir.glob("*summary_confidences.json"))
        if len(summary_files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one *summary_confidences.json, found {len(summary_files)}"
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
            cif_path=cif_files[0],
            summary_json_path=summary_files[0],
            full_json_path=full_files[0],
        )
        return res

    #  TODO: Add structures
    # def _load_cif(self) -> None:
    #     pass
    # mm = MMCIF2Dict(str(self.cif_path))
    # chains = mm.get("atom_site.auth_asym_id", [])
    # resseqs = mm.get("atom_site.label_seq_id", [])
    # bvals = mm.get("atom_site.B_iso_or_equiv", [])

    # seen = set()
    # residues: List[ResidueConfidence] = []
    # for i, chain in enumerate(chains):
    #     resseq = resseqs[i]
    #     bval = bvals[i] if i < len(bvals) else None
    #     key = (chain, resseq)
    #     if key not in seen:
    #         seen.add(key)
    #         try:
    #             resid = int(resseq)
    #         except Exception:
    #             resid = len(seen)
    #         residues.append(
    #             ResidueConfidence(
    #                 chain_id=str(chain),
    #                 resseq=str(resseq),
    #                 resid=resid,
    #                 plddt=float(bval) if bval not in (".", "?", None) else None,
    #             )
    #         )
    # self.residues = residues
