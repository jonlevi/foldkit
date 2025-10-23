API Reference
=============

FoldKit provides Python and CLI interfaces to work with AlphaFold3 results.

AF3Result Class
---------------

.. autoclass:: foldkit.af3_result.AF3Result
    :members:
    :undoc-members:
    :special-members: __init__
    :exclude-members: _build_from_af3_output, _extract_confidences

.. rubric:: Attributes

The following attributes are available on each **AF3Result** instance:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Attribute
     - Type
     - Description
   * - id
     - str
     - Unique identifier for this result
   * - summary_json_path
     - Path
     - Path to the summary confidences JSON file
   * - full_json_path
     - Path
     - Path to the full confidences JSON file
   * - chains
     - list[str]
     - List of chain IDs in the structure
   * - plddt
     - np.ndarray
     - Per-atom pLDDT confidence scores
   * - pae
     - np.ndarray
     - Predicted Alignment Error matrix
   * - contact_probs
     - np.ndarray
     - Predicted contact probabilities
   * - global_ptm
     - float
     - Global predicted TM-score
   * - global_iptm
     - float
     - Global inter-chain predicted TM-score
   * - chain_ptm
     - np.ndarray
     - Per-chain predicted TM scores
   * - chain_pair_iptm
     - np.ndarray
     - Per-chain pair iPTM scores

.. admonition:: Note

   Use the factory method ``AF3Result.load_result(result_dir)`` to create an instance from a directory of AlphaFold3 results. This ensures all confidence metrics and chain-specific scores are properly loaded.

.. rubric:: Methods

Methods are used to query specific metrics, either globally or per chain.

- ``get_ptm(chain=None)``
- ``get_iptm(chain1=None, chain2=None)``
- ``get_pae(chain=None, tokens=None, agg=np.mean)``
- ``get_ipae(chain1=None, chain2=None, tokens1=None, tokens2=None, agg=np.mean)``
- ``get_plddt(chain=None, atoms=None, agg=np.mean)``
- ``get_contact_probs(chain1=None, chain2=None, tokens1=None, tokens2=None, agg=np.max)``

Each method supports optional aggregation functions and selection by chain or atom indices.

CLI Module
----------

FoldKit provides a command-line interface for batch processing and exporting AlphaFold3 results.

.. code-block:: bash

    # Export a single AlphaFold3 result
    foldkit export-result path/to/af3_result_dir path/to/output_dir

    # Export a directory of results
    foldkit export-directory path/to/results_directory path/to/output_directory

.. autofunction:: foldkit.cli.export_single_result
.. autofunction:: foldkit.cli.export_multi_result
.. autofunction:: foldkit.cli.batch_export_multi_result
.. autofunction:: foldkit.cli.main
