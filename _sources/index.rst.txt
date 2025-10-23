.. foldkit documentation master file, created by
   sphinx-quickstart on Tue Oct 21 13:42:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FoldKit
==================

FoldKit is a Python package for working with AlphaFold3 results.

Features:
- Load AF3 results
- Access per-chain and global confidence metrics
- Export predictions to NPZ files

How Efficient?
==================
Early benchmarking shows that a single AF3 output directory for a four-chain protein is around 7.8M, while the foldkit exported version is 1.9M. This may seem like a small difference, but can scale over a large protein design of co-folding campaign over a large dataset. For example, a parent directory of ~1000 AF3 folded complexes, each with 4 seeds and 5 samples, the total space to store the results goes from 157G --> 38G.

Getting started
---------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorial
   api