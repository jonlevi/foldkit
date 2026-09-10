"""
foldkit.core
Core object for an ensemble of multiple AlphaFold3 results, with the same sequence but varying seeds/samples

Features:
- Load Ensemble of AF3 results from directory of multiple results.
- Interface for aggregation of results across the ensemble
"""

import json
from typing import Optional, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

from .af3_result import AF3Result

SEED_SAMPLE_REGEX = r"seed-(\d+)_sample-(\d+)"


class AF3Ensemble:

    def __init__(self):
        # store individual results mapped by seed/sample name
        self.af3_results: dict[str, AF3Result] = dict()
        self.af3_ranking_scores: dict[str, float] = dict()
        self.seeds: list[int] = []
        self.samples: list[int] = []
        self.size: int = []
        self.chains: list[str] = []

        # Metadata
        self.id: str = None
        self.top_level_directory: Path = None

    def _build_from_af3_output(
        self, id: str, top_level_directory: Path, compressed: bool = False
    ):
        self.id = id
        self.top_level_directory = Path(top_level_directory)
        self._extract_result_objects(compressed=compressed)

    def _extract_result_objects(self, compressed: bool):
        if len(self.af3_results) > 0:
            print("Error: Already initialized. Skipping...")
            return

        seeds = set()
        samples = set()
        ranking_files = list(self.top_level_directory.glob("*ranking_scores.csv"))
        if len(ranking_files) == 0:
            raise Exception("Could not find ranking_scores.csv file")
        if len(ranking_files) > 1:
            raise Exception(
                "Found multiple ranking_scores.csv files and don't know which to use."
            )

        ranking_file = ranking_files[0]
        rankings = pd.read_csv(ranking_file)
        subdirectories = [x for x in self.top_level_directory.iterdir() if x.is_dir()]
        for subdir in tqdm(subdirectories):
            match = re.search(SEED_SAMPLE_REGEX, subdir.name)
            if match:
                seed = int(match.group(1))
                seeds.add(seed)
                sample = int(match.group(2))
                samples.add(sample)

                if compressed:
                    self.af3_results[f"seed-{seed}_sample-{sample}"] = (
                        AF3Result.load_compressed_result(subdir)
                    )
                else:
                    self.af3_results[f"seed-{seed}_sample-{sample}"] = (
                        AF3Result.load_af3_result(subdir, subdir.name)
                    )

                ranking = rankings[
                    (rankings["seed"] == seed) & (rankings["sample"] == sample)
                ]
                assert (
                    len(ranking) == 1
                ), f"ranking CSV does not contain one row for seed={seed} and sample={sample}"
                self.af3_ranking_scores[f"seed-{seed}_sample-{sample}"] = float(
                    ranking["ranking_score"].values[0]
                )
            else:
                raise Exception(
                    f"Unable to extract seed and sample information from directory name: {subdir.name}. Subdirectories should have names that fit the pattern `seed-X_sample-Y`"
                )

        self.size = len(self.af3_results)
        self.seeds = sorted(list(seeds))
        self.samples = sorted(list(samples))

        results_chains = [result.chains for result in self.af3_results.values()]
        first = results_chains[0]
        assert all(result_chains == first for result_chains in results_chains)
        self.chains = first

    def get_result_by_seed_and_sample(self, seed: int, sample: int) -> AF3Result:
        seed = int(seed)
        sample = int(sample)
        if seed not in self.seeds:
            raise ValueError(f"Seed {seed} not in ensemble (seeds: {self.seeds})")
        if sample not in self.samples:
            raise ValueError(
                f"Sample {sample} not in ensemble (samples: {self.samples})"
            )

        return self.af3_results[f"seed-{seed}_sample-{sample}"]

    def get_top_ranked_result(self) -> AF3Result:
        top_structure = max(self.af3_ranking_scores, key=self.af3_ranking_scores.get)
        return self.af3_results[top_structure]

    def _get_all(self, method_name: str, **kwargs) -> dict[str, float]:
        """
        Call `method_name` on every AF3Result in the ensemble and collect
        the results into a dict keyed by result id.

        Parameters
        ----------
        method_name : str
            Name of the method to call on each `AF3Result` (e.g. "get_ptm").
        **kwargs
            Forwarded to the underlying method call.

        Returns
        -------
        dict[str, float]
            One value per member of the ensemble.
        """
        return {
            id_: getattr(af3_result, method_name)(**kwargs)
            for id_, af3_result in self.af3_results.items()
        }

    def _get_ensemble(
        self,
        method_name: str,
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
        **kwargs,
    ) -> float:
        """
        Call `method_name` on every AF3Result in the ensemble and aggregate
        the resulting values with `ensemble_agg`.

        Parameters
        ----------
        method_name : str
            Name of the method to call on each `AF3Result` (e.g. "get_ptm").
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply across the ensemble of values.
        **kwargs
            Forwarded to the underlying method call.

        Returns
        -------
        float
            Ensemble-aggregated score.
        """
        values = np.array(list(self._get_all(method_name, **kwargs).values()))
        return ensemble_agg(values)

    # ------------------------------------------------------------------
    # ptm
    # ------------------------------------------------------------------

    def get_all_ptm(self, chain: Optional[str] = None) -> dict[str, float]:
        """
        Return the ptm score.

        Parameters
        ----------
        chain : str, optional
            If provided, computes the ptm for this specific chain.
            If None, returns the global score across all chains in the structure.

        Returns
        -------
        dict[str, float]
            ptm scores for each AF3 Result.

        Raises
        ------
        ValueError
            If `chain` is invalid based on the data
        """
        return self._get_all("get_ptm", chain=chain)

    def get_ensemble_ptm(
        self,
        chain: Optional[str] = None,
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the ensemble aggregated ptm score.

        Parameters
        ----------
        chain : str, optional
            If provided, computes the ptm for this specific chain.
            If None, returns the global score across all chains in the structure.
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply to the ensemble of values.

        Returns
        -------
        float
            ensemble aggregated ptm score.

        Raises
        ------
        ValueError
            If `chain` is invalid based on the data
        """
        return self._get_ensemble("get_ptm", ensemble_agg=ensemble_agg, chain=chain)

    # ------------------------------------------------------------------
    # iptm
    # ------------------------------------------------------------------

    def get_all_iptm(
        self, chain1: Optional[str] = None, chain2: Optional[str] = None
    ) -> dict[str, float]:
        """
        Return the interchain ptm (iptm) score for all members of the ensemble.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the iptm for this specific pair of chains.
            Otherwise, returns the global iptm score for all pairs of chains

        Returns
        -------
        dict[str, float]
            iptm scores for each AF3 Result.

        Raises
        ------
        ValueError
            If `chain1` or `chain2` is invalid based on the data
        """
        return self._get_all("get_iptm", chain1=chain1, chain2=chain2)

    def get_ensemble_iptm(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the ensemble aggregated interchain ptm (iptm) score.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the iptm for this specific pair of chains.
            Otherwise, returns the global iptm score for all pairs of chains
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply to the ensemble of values.

        Returns
        -------
        float
            ensemble aggregated iptm score.

        Raises
        ------
        ValueError
            If `chain` is invalid based on the data
        """
        return self._get_ensemble(
            "get_iptm", ensemble_agg=ensemble_agg, chain1=chain1, chain2=chain2
        )

    # ------------------------------------------------------------------
    # pae
    # ------------------------------------------------------------------

    def get_all_pae(
        self,
        chain: Optional[str] = None,
        tokens: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> dict[str, float]:
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
        dict[str, float]
            Full ensemble of pae scores, aggregated across PAE matrix.

        Raises
        ------
        ValueError
            If both `chain` and `tokens` are provided, or if either argument is invalid based on the data
        """
        return self._get_all("get_pae", chain=chain, tokens=tokens, agg=agg)

    def get_ensemble_pae(
        self,
        chain: Optional[str] = None,
        tokens: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the ensemble-aggregated pae (predicted alignment error) score.

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
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply to the ensemble of values.

        Returns
        -------
        float
            Ensemble aggregated pae score, aggregated across PAE matrix per
            structure and then aggregated across ensemble.

        Raises
        ------
        ValueError
            If both `chain` and `tokens` are provided, or if either argument is invalid based on the data
        """
        return self._get_ensemble(
            "get_pae",
            ensemble_agg=ensemble_agg,
            chain=chain,
            tokens=tokens,
            agg=agg,
        )

    # ------------------------------------------------------------------
    # ipae
    # ------------------------------------------------------------------

    def get_all_ipae(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        tokens1: Optional[list[int]] = None,
        tokens2: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> dict[str, float]:
        """
        Return the interface pae (predicted alignment error) score.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the ipae for this specific pair of chains.
            Only optional if `tokens1` and `tokens2` are provided
        tokens1, tokens2 : list[int], optional
            If provided, computes the ipae for this specific pair of token lists.
            Only optional if `chain1` and `chain2` are provided
        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.

        Returns
        -------
        dict[str, float]
            Aggregated ipae score for each structure in the ensemble.

        Raises
        ------
        ValueError
            If neither `chain` and `tokens` are provided, both are provided, or if either argument is invalid based on the data
        """
        return self._get_all(
            "get_ipae",
            chain1=chain1,
            chain2=chain2,
            tokens1=tokens1,
            tokens2=tokens2,
            agg=agg,
        )

    def get_ensemble_ipae(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        tokens1: Optional[list[int]] = None,
        tokens2: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the ensemble aggregated interface pae (predicted alignment error) score.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the ipae for this specific pair of chains.
            Only optional if `tokens1` and `tokens2` are provided
        tokens1, tokens2 : list[int], optional
            If provided, computes the ipae for this specific pair of token lists.
            Only optional if `chain1` and `chain2` are provided
        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply to the ensemble of values.

        Returns
        -------
        float
            Ensemble aggregated ipae score.

        Raises
        ------
        ValueError
            If neither `chain` and `tokens` are provided, both are provided, or if either argument is invalid based on the data
        """
        return self._get_ensemble(
            "get_ipae",
            ensemble_agg=ensemble_agg,
            chain1=chain1,
            chain2=chain2,
            tokens1=tokens1,
            tokens2=tokens2,
            agg=agg,
        )

    # ------------------------------------------------------------------
    # ipsae
    # ------------------------------------------------------------------

    def get_all_ipsae(
        self, chain1: str, chain2: str, pae_cutoff: float = 10, d0="d0res"
    ) -> dict[str, float]:
        """
        Return the ipase score for all members of the ensemble.

        Parameters
        ----------
        chain1, chain2 : str
            Computes the ipsae for this specific pair of chains.
        pae_cutoff : float, default=10
            PAE cutoff (in Angstroms) passed through to the ipSAE calculation.
        d0: str, default=d0res
            d0 used for calculation. Options are:
            d0res: d0 determined by number of residues in 2nd chain with PAE<cutoff
            d0chn: sum of chain lengths
            d0dom: d0 = total number of residues in both chains with any interchain PAE<cutoff
        Returns
        -------
        dict[str, float]
            ipsae scores for each AF3 Result.

        Raises
        ------
        ValueError
            If `chain1` or `chain2` is invalid based on the data
        """
        return self._get_all(
            "get_ipsae", chain1=chain1, chain2=chain2, pae_cutoff=pae_cutoff, d0=d0
        )

    def get_ensemble_ipsae(
        self,
        chain1: str,
        chain2: str,
        pae_cutoff: float = 10,
        d0="d0res",
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the ensemble aggregated ipsae score.

        Parameters
        ----------
        chain1, chain2 : str
            Computes the ipsae for this specific pair of chains.
        pae_cutoff : float, default=10
            PAE cutoff (in Angstroms) passed through to the ipSAE calculation.
        d0: str, default=d0res
            d0 used for calculation. Options are:
            d0res: d0 determined by number of residues in 2nd chain with PAE<cutoff
            d0chn: sum of chain lengths
            d0dom: d0 = total number of residues in both chains with any interchain PAE<cutoff
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply to the ensemble of values.

        Returns
        -------
        float
            ensemble aggregated ipsae score.

        Raises
        ------
        ValueError
            If `chain` is invalid based on the data
        """
        return self._get_ensemble(
            "get_ipsae",
            ensemble_agg=ensemble_agg,
            chain1=chain1,
            chain2=chain2,
            pae_cutoff=pae_cutoff,
            d0=d0,
        )

    # ------------------------------------------------------------------
    # contact probabilities
    # ------------------------------------------------------------------

    def get_all_contact_probs(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        tokens1: Optional[list[int]] = None,
        tokens2: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> dict[str, float]:
        """
        Return the contact probs score for each structure in the ensemble.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the contact probs for this specific pair of chains.
            Only optional if `tokens1` and `tokens2` are provided
        tokens1, tokens2 : list[int], optional
            If provided, computes the contact probs for this specific pair of token lists.
            Only optional if `chain1` and `chain2` are provided
        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.

        Returns
        -------
        dict[str, float]
            Contact probs score for each member of the ensemble.

        Raises
        ------
        ValueError
            If neither `chain` and `tokens` are provided, both are provided, or if either argument is invalid based on the data
        """
        return self._get_all(
            "get_contact_probs",
            chain1=chain1,
            chain2=chain2,
            tokens1=tokens1,
            tokens2=tokens2,
            agg=agg,
        )

    def get_ensemble_contact_probs(
        self,
        chain1: Optional[str] = None,
        chain2: Optional[str] = None,
        tokens1: Optional[list[int]] = None,
        tokens2: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the contact probs score aggregated over the ensemble.

        Parameters
        ----------
        chain1, chain2 : str, optional
            If provided, computes the contact probs for this specific pair of chains.
            Only optional if `tokens1` and `tokens2` are provided
        tokens1, tokens2 : list[int], optional
            If provided, computes the contact probs for this specific pair of token lists.
            Only optional if `chain1` and `chain2` are provided
        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply to the ensemble of values.

        Returns
        -------
        float
            Contact probs score aggregated across the ensemble.

        Raises
        ------
        ValueError
            If neither `chain` and `tokens` are provided, both are provided, or if either argument is invalid based on the data
        """
        return self._get_ensemble(
            "get_contact_probs",
            ensemble_agg=ensemble_agg,
            chain1=chain1,
            chain2=chain2,
            tokens1=tokens1,
            tokens2=tokens2,
            agg=agg,
        )

    # ------------------------------------------------------------------
    # plddt
    # ------------------------------------------------------------------

    def get_all_plddt(
        self,
        chain: Optional[str] = None,
        atoms: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
    ) -> dict[str, float]:
        """
        Return the plddt score for all structures in the ensemble.

        Parameters
        ----------
        chain : str, optional
            If provided, computes the plddt for this specific chain.
            If None, returns the score across all atoms in the structure.
        atoms : list[int], optional
            If provided, computes the plddt for specific atoms.
            If None, returns the score across all atoms in the structure.
            Cannot be used in combination with "chain"
        agg : callable, default=np.mean
            Aggregation function to apply to the selected values.

        Returns
        -------
        dict[str, float]
            per structure plddt score.

        Raises
        ------
        ValueError
            If both `chain` and `atoms` are provided, or if either argument is invalid based on the data
        """
        return self._get_all("get_plddt", chain=chain, atoms=atoms, agg=agg)

    def get_ensemble_plddt(
        self,
        chain: Optional[str] = None,
        atoms: Optional[list[int]] = None,
        agg: Callable[[np.ndarray], float] = np.mean,
        ensemble_agg: Callable[[np.ndarray], float] = np.mean,
    ) -> float:
        """
        Return the ensemble-aggregated plddt score.

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
        ensemble_agg : callable, default=np.mean
            Aggregation function to apply to the ensemble of values.

        Returns
        -------
        float
            Aggregated plddt score.

        Raises
        ------
        ValueError
            If both `chain` and `atoms` are provided, or if either argument is invalid based on the data
        """
        return self._get_ensemble(
            "get_plddt", ensemble_agg=ensemble_agg, chain=chain, atoms=atoms, agg=agg
        )

    @staticmethod
    def load_af3_result(result_dir: str, id: Optional[str] = None) -> "AF3Ensemble":
        """
        Generate an AF3Ensemble for each of the samples from a run on the AF3 webserver.
        Parameters
        ----------
            result_dir: path to directory storing results.
            id: optional ID. If unset, will set to directory name
        Returns
        -------
        AF3Ensemble
            loaded AF3Ensemble

        """
        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(
                f"Result directory not found: {result_dir} does not exist or is not a directory"
            )
        new_ensemble = AF3Ensemble()
        id = id if id is not None else result_dir.name
        new_ensemble._build_from_af3_output(id, result_dir)
        return new_ensemble

    @staticmethod
    def load_compressed_result(
        result_dir: str, id: Optional[str] = None
    ) -> "AF3Ensemble":
        """
        Generate an AF3Ensemble for each of the samples stored via export.
        Parameters
        ----------
            result_dir: path to directory storing results.
            id: optional ID. If unset, will set to directory name
        Returns
        -------
        AF3Ensemble
            loaded AF3Ensemble
        """
        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(
                f"Result directory not found: {result_dir} does not exist or is not a directory"
            )
        new_ensemble = AF3Ensemble()
        id = id if id is not None else result_dir.name
        new_ensemble._build_from_af3_output(id, result_dir, compressed=True)
        return new_ensemble

    @staticmethod
    def load_webserver_result(
        result_dir: str, seed: Optional[int] = None, id_stem: Optional[str] = None
    ) -> "AF3Ensemble":
        """
        Generate an AF3Ensemble for each of the samples from a run on the AF3 webserver.

        Parameters
        ----------
            result_dir: path to directory storing results.
            seed: Model seed. If unset, will try to infer these from a request file in result_dir.
            id_stem: Job name. If unset, will try to infer these from a request file in result_dir.
        Returns
        -------
        AF3Ensemble
            loaded AF3Ensemble
        """

        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(
                f"Result directory not found: {result_dir} or is not a directory"
            )

        new_ensemble = AF3Ensemble()
        new_ensemble.top_level_directory = result_dir

        # load from webserver results
        summary_files = list(result_dir.glob("*summary_confidences*.json"))
        if seed is None or id_stem is None:
            request_files = list(result_dir.glob("*request*.json"))
            if len(request_files) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one request.json, found {len(request_files)}."
                )
            with open(request_files[0], "r") as request_json:
                request_json = json.load(request_json)
                if len(request_json) != 1:
                    raise FileNotFoundError(
                        f"Expected exactly one request inside the JSON, found {len(request_json)}."
                    )
                request = request_json[0]
                id_stem = request["name"] if id_stem is None else id_stem

                seed = request["modelSeeds"][0] if seed is None else seed

        num_samples = len(summary_files)  # usually this is 5

        new_ensemble.seeds = [seed]
        new_ensemble.samples = list(range(num_samples))
        new_ensemble.size = num_samples
        new_ensemble.id = id_stem

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

            if match := re.search(r"summary_confidences_(\d)\.json", summary_file.name):
                sample_num = int(match.group(1))
            else:
                raise ValueError(
                    f"Improperly formatted summary_confidences filename:\n{summary_file.name}"
                )

            sample_stem = f"{id_stem}_seed{seed}_sample{sample_num}"

            res = AF3Result()
            res._build_from_af3_output(
                id=sample_stem,
                summary_json_path=summary_file,
                full_json_path=full_file,
                cif_path=structure_file,
            )
            new_ensemble.af3_results[f"seed-{seed}_sample-{sample_num}"] = res

            with open(summary_file, "r") as j:
                summary = json.load(j)
            ranking = float(summary["ranking_score"])
            new_ensemble.af3_ranking_scores[f"seed-{seed}_sample-{sample_num}"] = (
                ranking
            )

        return new_ensemble
