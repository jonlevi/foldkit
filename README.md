# FoldKit

A Python toolkit for working with AlphaFold3 results and storing them efficiently.

## Installation
`pip install foldkit`

## Use Cases
There are two main use cases for this package
(1) Convenient python interface for accessing AlphaFold3 prediction confidence metrics. This is particularly useful for metrics across chains in protein complexes, as they have been shown to be predictive of binding and are useful criteria for protein design and specificity predictions

(2) Efficient Storage of AlphaFold3 confidence results. The default JSON formats for AF3 results are large, and can take up a lot of unnecessary space. Foldkit has a CLI for exporting the AF3 confidence JSONs to space-efficient .npz files. These .npz files can also be used as inputs to the python interface in (1)

## foldkit - Python Interface Tutorial
Let's say you have a directory that contains the results of an AlphaFold3 prediction for a protein complex. This protein complex is actually a TCR with the following chains: ["A", "B", "M", "P"] (which is the TCRa, TCRb, MHCa, peptide). These results are stored in a directory:
`"structures/tcr_pmhc_1/"`.`
We can can load the results:
```
import foldkit
result_obj = foldkit.AF3Result.load_result("structures/tcr_pmhc_1/")
```
This object has access to all of the confidence metadata, as well as the ability to compute specific statistics on the metadata.
```
>>> result_obj.chains
[np.str_('A'), np.str_('B'), np.str_('M'), np.str_('P')]
``` 
For example, the structure wide PTM:
```
>>> result_obj.get_ptm()
0.81
```
Or, just the average PTM for the TCRa chain:
```
>>> result_obj.get_ptm("A")
0.82
```
Here is the average interaction_pae (ipae) between the TCRb chain and the peptide:
```
>>> result_obj.get_ipae(chain1="B", chain2="P")
np.float64(6.253699186991869)
```

By default, these methods compute the average. But maybe you want a different aggregation function? You can pass in a custom `agg`:
```
>>> result_obj.get_ipae(chain1="B", chain2="P", agg=np.min)
np.float64(1.3)
```

## folkdkit - CLI Tutorial
```
usage: foldkit [-h] [--verbose] {export-single-result,export-multi-result,batch-export-multi-result} ...

Export AlphaFold3 result directories into compressed format.Converts confidences into npz format and copies over the rest of the data as is (except the _input_data.json which is not kept since it is redundant).

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
```
There are 3 main entry points, depending on the data you are exporting:
1) A single prediction directory (i.e. one prediction corresponding to a single seed and sample)
2) A prediction directory (i.e. N*K predictions corresponding to the same input with N seeds and K samples)
3) A directory of prediction directories (i.e. a directory containing many "prediction directories" like in (2).

  ### 1- Export a single result (i.e. one single structure from a single seed and sample)
  ```
  foldkit export-result /path/to/specific_structure_directory /path/to/outdir
  ```
  ### 2- Export a single result with multiple seeds and/or samples
  ```
foldkit export-multi-result /path/to/specific_structure_parent_directory /data1/greenbab/users/levinej4/af3/foldkit/tests/test_data/test-m1
  ```
  ### 3- Batch export many results
  ```
  foldkit -v batch-export-multi-result  /path/to/directory_of_subdirectories/ /path/to/outdir
  ```