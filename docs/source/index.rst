.. foldkit documentation master file, created by
   sphinx-quickstart on Tue Oct 21 13:42:36 2025.
 
FoldKit
=======
 
**FoldKit** is a Python toolkit for working with AlphaFold3 results — loading
predictions, computing per-chain and per-pair confidence metrics, and
compressing large result directories down to a fraction of their size.
 
.. grid:: 1 2 2 2
    :gutter: 3
    :class-container: sd-text-center
 
    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc
 
        Get FoldKit installed from PyPI in one command.
 
    .. grid-item-card:: Tutorial
        :link: tutorial
        :link-type: doc
 
        Walk through loading single structures, ensembles, and the CLI.
 
    .. grid-item-card:: API Reference
        :link: api
        :link-type: doc
 
        Full documentation for ``AF3Result``, ``AF3Ensemble``, and the CLI.
 
    .. grid-item-card:: GitHub
        :link: https://github.com/jonlevi/foldkit
 
        Source code, issues, and releases.
 
Why FoldKit?
------------
 
AlphaFold3 output directories are large and awkward to work with at scale.
FoldKit gives you a clean Python interface to the confidence metrics you
actually care about — pLDDT, pAE/ipAE, pTM/ipTM, contact probabilities, and
ipSAE — for both single predictions and multi-seed, multi-sample ensembles.
 
It also ships a compression format that shrinks result directories
dramatically without losing any of the underlying data:
 
.. list-table::
   :header-rows: 1
   :widths: 40 30 30
 
   * - Scale
     - Raw AF3 output
     - FoldKit exported
   * - One 4-chain complex
     - ~7.8 MB
     - ~1.9 MB
   * - 1,000 complexes × 4 seeds × 5 samples
     - ~157 GB
     - ~38 GB
 
Quick example
-------------
 
.. code-block:: python
 
    import foldkit
 
    result = foldkit.AF3Result.load_af3_result("tutorial_example/single_result")
    result.get_ptm()                          # mean global PTM
    result.get_ipae(chain1="B", chain2="P")    # interface PAE between two chains
 
See the :doc:`tutorial` for the full walkthrough, including ensembles and the
command-line interface.
 
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents
 
   installation
   tutorial
   api