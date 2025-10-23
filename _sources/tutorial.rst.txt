Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial covers basic usage of FoldKit, including the **Python API** and the **command-line interface (CLI)**.

Python API Usage
----------------

Suppose you have a directory containing the results of an AlphaFold3 prediction for a protein complex. For example, a TCR complex with chains: ["A", "B", "M", "P"] (TCRa, TCRb, MHCa, peptide). The results are stored in:

`structures/tcr_pmhc_1/`


We can load the results using the factory method:

.. code-block:: python

    import foldkit

    result_obj = foldkit.AF3Result.load_result("structures/tcr_pmhc_1/")

.. admonition:: Note

    This object provides access to all confidence metadata and methods to compute chain-specific or global statistics.

Querying Chains and Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check which chains are present:

.. code-block:: python

    >>> result_obj.chains
    [np.str_('A'), np.str_('B'), np.str_('M'), np.str_('P')]

Structure-wide PTM:

.. code-block:: python

    >>> result_obj.get_ptm()
    0.81

Chain-specific PTM (e.g., TCRa):

.. code-block:: python

    >>> result_obj.get_ptm("A")
    0.82

Average interaction PAE (ipae) between TCRb and the peptide:

.. code-block:: python

    >>> result_obj.get_ipae(chain1="B", chain2="P")
    np.float64(6.253699186991869)

Custom aggregation functions (e.g., minimum instead of mean):

.. code-block:: python

    >>> result_obj.get_ipae(chain1="B", chain2="P", agg=np.min)
    np.float64(1.3)

Advanced Metrics
~~~~~~~~~~~~~~~~

Other useful methods include:

- ``get_iptm(chain1=None, chain2=None)`` — Interchain TM-score
- ``get_pae(chain=None, tokens=None, agg=np.mean)`` — Predicted alignment error
- ``get_plddt(chain=None, atoms=None, agg=np.mean)`` — Per-atom confidence scores
- ``get_contact_probs(chain1=None, chain2=None, tokens1=None, tokens2=None, agg=np.max)`` — Predicted contact probabilities

.. admonition:: Tip

    You can pass specific chains, atom indices, or token indices to compute metrics for subsets of the structure.


Loading from .npz format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
From the CLI, let's say you had previously exported the result of a AF3 run, so that the result from before
at `"structures/tcr_pmhc_1/"` is now at `"structures_compressed/tcr_pmhc_1/"`. This second directory will have a .npz file in it instead of JSON files. You can load it in a very similar way by adding the `from_npz=True` flag

.. code-block:: python

    >>> result_obj = foldkit.AF3Result.load_result("structures/tcr_pmhc_1/", from_npz=True)


Command-Line Interface (CLI) Usage
----------------------------------

FoldKit provides a CLI to export AlphaFold3 results into compressed formats. You can convert confidences into NPZ files and copy other data as-is (except `_input_data.json`, which is redundant).

Basic usage:

.. code-block:: text

    usage: foldkit [-h] [--verbose] {export-single-result,export-multi-result,batch-export-multi-result} ...

    positional arguments:
      {export-single-result,export-multi-result,batch-export-multi-result}
        export-single-result
                            Export a single AlphaFold3 result directory to compressed format
        export-multi-result
                            Export multiseed/multisample AlphaFold3 results to compressed format.
        batch-export-multi-result
                            Export multiple AlphaFold3 results to compressed format.

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Print detailed output.

The CLI has three main entry points, depending on the data being exported:

1. **Export a single prediction directory**  
   One prediction corresponding to a single seed and sample.

   .. code-block:: bash

       foldkit export-result /path/to/specific_structure_directory /path/to/outdir

2. **Export a single result with multiple seeds and/or samples**  
   N × K predictions corresponding to the same input.

   .. code-block:: bash

       foldkit export-multi-result /path/to/specific_structure_parent_directory /path/to/outdir

3. **Batch export many results**  
   A directory containing many "prediction directories" as in (2).

   .. code-block:: bash

       foldkit -v batch-export-multi-result /path/to/directory_of_subdirectories/ /path/to/outdir

.. admonition:: Tip

    Use ``--verbose`` to print detailed output and track progress during batch exports.
