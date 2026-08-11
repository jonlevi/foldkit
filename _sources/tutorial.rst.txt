Tutorial
========
 
This tutorial covers basic usage of FoldKit, including the **Python API** and
the **command-line interface (CLI)**.
 
Python API usage
-----------------
 
Single predicted structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
Let's say you have a directory that contains the results of an AlphaFold3
prediction for a protein complex. For example, a TCR with four chains
``["A", "B", "M", "P"]`` (TCRa, TCRb, MHCa, peptide). The AF3 run results are
stored in a directory (see ``tutorial_example`` in the repo):
``tutorial_example/single_result``.
 
.. code-block:: text
 
    $ ls tutorial_example/single_result
    confidences.json  model.cif  summary_confidences.json
 
Load the result:
 
.. code-block:: python
 
    import foldkit
 
    result_obj = foldkit.AF3Result.load_af3_result("tutorial_example/single_result")
 
This object has access to all of the confidence metadata, as well as the
ability to compute specific statistics from it:
 
.. code-block:: pycon
 
    >>> result_obj.chains
    [np.str_('A'), np.str_('B'), np.str_('M'), np.str_('P')]
 
    >>> result_obj.get_ptm()  # mean global PTM
    0.81
 
    >>> result_obj.get_ptm("A")  # mean PTM for chain A
    0.82
 
    >>> result_obj.get_ipae(chain1="B", chain2="P")  # ipae between chain B and chain P
    np.float64(6.245691056910569)
 
    >>> result_obj.get_ipsae(chain1="B", chain2="P")  # ipsae between chain B and chain P
    np.float64(0.292483968491584)
 
.. note::
 
   ``get_ipsae`` follows the definition from `IPSAE <https://github.com/DunbrackLab/IPSAE>`_.
 
By default, these methods compute the mean over the confidence metrics
aggregated per chain / chain-pair. You can pass a custom ``agg`` for a
different aggregation:
 
.. code-block:: pycon
 
    >>> result_obj.get_ipae(chain1="B", chain2="P", agg=max)
    np.float64(29.8)
 
Ensemble of predicted structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
Let's say you have a directory containing AlphaFold3 predictions for a
protein complex across several seeds and samples. Using the same TCR, the
results are stored as a directory of directories (see ``tutorial_example``):
``tutorial_example/ensemble_result``.
 
.. code-block:: text
 
    ensemble_result/
    ├── ranking_scores.csv
    │
    ├── seed-1_sample-0/
    ├── seed-1_sample-1/
    ├── ...
    ├── seed-5_sample-0/
    ├── seed-5_sample-1/
    ├── ...
 
Load the ensemble:
 
.. code-block:: pycon
 
    >>> ensemble_obj = foldkit.AF3Ensemble.load_af3_result("tutorial_example/ensemble_result")
    100%|████████████████████████████████████████| 20/20 [00:03<00:00,  5.05it/s]
 
    >>> ensemble_obj.size
    20
    >>> ensemble_obj.seeds
    [1, 2, 5, 10]
    >>> ensemble_obj.samples
    [0, 1, 2, 3, 4]
 
The individual results are stored in ``ensemble_obj.af3_results``. Access an
individual structure by its seed and sample:
 
.. code-block:: pycon
 
    >>> ensemble_obj.get_result_by_seed_and_sample(seed=5, sample=2)
    <foldkit.af3_result.AF3Result object at 0x7f59d8a344b0>
 
Ranking scores are stored in ``ensemble_obj.af3_ranking_scores``. Get the
top-ranked structure directly:
 
.. code-block:: pycon
 
    >>> ensemble_obj.get_top_ranked_result()
    <foldkit.af3_result.AF3Result object at 0x7f215e6497f0>
 
You can also get any confidence metric, either per-structure or aggregated
over the entire ensemble:
 
.. code-block:: pycon
 
    >>> ensemble_obj.get_all_plddt("M")
    {'seed-1_sample-3': np.float64(78.71985788561527), 'seed-2_sample-4': np.float64(77.89603812824957), ...}
 
    >>> ensemble_obj.get_ensemble_plddt("M")  # default: mean over ensemble
    np.float64(78.2713937608319)
 
    >>> ensemble_obj.get_ensemble_plddt("M", ensemble_agg=max)  # max over ensemble
    np.float64(78.91594800693241)
 
.. warning::
 
   Don't mix up the matrix aggregation (``agg``) with the ensemble
   aggregation (``ensemble_agg``):
 
   .. code-block:: pycon
 
       # max of the plddt matrix, mean over the ensemble
       >>> ensemble_obj.get_ensemble_plddt("M", agg=max, ensemble_agg=np.mean)
       np.float64(98.647)
 
       # mean of the plddt matrix, max over the ensemble
       >>> ensemble_obj.get_ensemble_plddt("M", agg=np.mean, ensemble_agg=max)
       np.float64(78.91594800693241)
 
Full list of included metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
- pLDDT
- pAE and ipAE
- pTM and ipTM
- Contact probabilities
- ipSAE (see `IPSAE <https://github.com/DunbrackLab/IPSAE>`_)
 
Loading from exported, compressed format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
If you previously exported an AF3 run from the CLI (see below), the result
directory will contain a ``.npz`` file instead of JSON files, but loads the
same way:
 
.. code-block:: pycon
 
    >>> foldkit.AF3Result.load_compressed_result("tutorial_example/single_result_export")
    <foldkit.af3_result.AF3Result object at 0x7fe0bba2a120>
 
Same for an exported ensemble:
 
.. code-block:: pycon
 
    >>> foldkit.AF3Ensemble.load_compressed_result("tutorial_example/ensemble_result_export")
    100%|████████████████████████████████████████| 20/20 [00:00<00:00, 22.16it/s]
    <foldkit.af3_ensemble.AF3Ensemble object at 0x7f3a374ea120>
 
Loading from AF3 Server format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
If you exported a result from the AlphaFold Server, e.g. from
``structures/tcr_pmhc_1/`` to ``structures_compressed/tcr_pmhc_1/``, load it
the same way with the ``from_npz=True`` flag:
 
.. code-block:: python
 
    result_obj = foldkit.AF3Result.load_result("structures/tcr_pmhc_1/", from_npz=True)
 
CLI tutorial
------------
 
.. code-block:: text
 
    $ foldkit -h
    usage: foldkit [-h] [--verbose] {export-single-result,export-ensemble-result,webserver-export,batch-export} ...
 
    Export AlphaFold3 result directories into compressed format. Converts
    confidences into npz format and copies over the rest of the data as-is
    (except _input_data.json, which is redundant).
 
    positional arguments:
      {export-single-result,export-ensemble-result,webserver-export,batch-export}
        export-single-result
                            Export a single AlphaFold3 result directory to compressed format
        export-ensemble-result
                            Export multiseed/multisample AlphaFold3 ensemble of results to compressed format
        webserver-export    Export AlphaFold3 server results to compressed format
        batch-export        Export multiple AlphaFold3 results to compressed format
 
    options:
      -h, --help            show this help message and exit
      --verbose, -v         Print detailed output
 
There are four entry points, depending on the data you're exporting:
 
1. A single prediction directory (one prediction for a single seed and sample).
2. An ensemble prediction directory (N × K predictions for the same input, across N seeds and K samples).
3. A directory of AF3 server results — one seed, multiple samples, in a slightly different layout than the local AF3 installation output.
4. A directory of ensemble directories (many "ensemble directories" as in #2).
 
After a successful export, it's safe to delete the source directory.
 
Export a single result
~~~~~~~~~~~~~~~~~~~~~~~
 
.. code-block:: text
 
    $ foldkit export-single-result -h
    usage: foldkit export-single-result [-h] input_directory output_directory
 
    positional arguments:
      input_directory   Path to AlphaFold3 result directory
      output_directory  Output directory path
 
    options:
      -h, --help        show this help message and exit
 
Example:
 
.. code-block:: console
 
    $ foldkit -v export-single-result tutorial_example/single_result tutorial_example/single_result_export
    ✅ Exported Data to : tutorial_example/single_result_export
 
Export an ensemble of results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
.. code-block:: text
 
    $ foldkit export-ensemble-result -h
    usage: foldkit export-ensemble-result [-h] input_directory output_directory
 
    positional arguments:
      input_directory   Path to parent directory containing subdirectories with AF3 results
      output_directory  Parent output directory path
 
    options:
      -h, --help        show this help message and exit
 
Example:
 
.. code-block:: console
 
    $ foldkit -v export-ensemble-result tutorial_example/ensemble_result tutorial_example/ensemble_result_export
    ✅ Exported Data to : tutorial_example/ensemble_result_export/seed-1_sample-3
    ✅ Exported Data to : tutorial_example/ensemble_result_export/seed-2_sample-4
    ✅ Exported Data to : tutorial_example/ensemble_result_export/seed-2_sample-1
    ...
 
Export a webserver result
~~~~~~~~~~~~~~~~~~~~~~~~~~
 
.. code-block:: console
 
    $ foldkit -v webserver-export tutorial_example/server tutorial_example/server_export
    ✅ Exported Data to : tutorial_example/server_export
 
Batch export many results
~~~~~~~~~~~~~~~~~~~~~~~~~~
 
.. code-block:: text
 
    $ foldkit batch-export -h
    usage: foldkit batch-export [-h] input_directory output_directory
 
    positional arguments:
      input_directory   Path to parent directory containing subdirectories with subdirectories of AF3 results
      output_directory  Parent output directory path
 
    options:
      -h, --help        show this help message and exit
 
Cleanup
~~~~~~~
 
After running any of the above exports successfully, you can safely delete
the source directory or directories.