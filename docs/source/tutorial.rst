
---

## **2️⃣ `tutorial.rst`**

Create `docs/source/tutorial.rst`:

```rst
Tutorial
========

This tutorial covers basic usage of FoldKit, including **Python API** and **command-line interface (CLI)**.

---

Python API Usage
----------------

## Python Interface Tutorial

### Single predicted structure
Let's say you have a directory that contains the results of an AlphaFold3 prediction for a protein complex. For example, let's look at a protein complex that is a TCR with the following 4 chains: ["A", "B", "M", "P"] (which is the TCRa, TCRb, MHCa, peptide). The AF3 run results are stored in a directory (which you can look at in the tutorial_example directory):
`"tutorial_example/single_result"`.

```
ls tutorial_example/single_result
confidences.json  model.cif  summary_confidences.json
```
We can can load the results:
```
import foldkit
result_obj = foldkit.AF3Result.load_af3_result("tutorial_example/single_result")
```
This object has access to all of the confidence metadata, as well as the ability to compute specific statistics on the metadata.
```
>>> result_obj.chains
[np.str_('A'), np.str_('B'), np.str_('M'), np.str_('P')]
``` 
For example:
```
>>> result_obj.get_ptm() # mean global PTM
0.81

>>> result_obj.get_ptm("A") # mean PTM for chain A:
0.82

# ipae between chain B and chain P
>>> result_obj.get_ipae(chain1="B", chain2="P") 
np.float64(6.245691056910569)

# ipsae between chain B and chain P
# see https://github.com/DunbrackLab/IPSAE
>>> result_obj.get_ipsae(chain1="B", chain2="P") 
np.float64(0.292483968491584)
```
By default, these methods compute the mean over the confidence metrics aggregated per chain/chain-pair. But maybe you want a different aggregation function? You can pass in a custom `agg`:
```
>>> result_obj.get_ipae(chain1="B", chain2="P", agg=max)
np.float64(29.8)
```

### Ensemble of predicted structures
Let's say you have a directory that contains the results of an AlphaFold3 prediction for a protein complex, with multiple predictions across several seeds and samples. For example, let's look at the same TCR, with AF3 run results stored in a directory of directories (which you can look at in the tutorial_example directory):
`"tutorial_example/ensemble_result"`.
```
ensemble_result/
├── ranking_scores.csv
│
├── seed-1_sample-0/
├── seed-1_sample-1/
├── ...

├── seed-5_sample-0/
├── seed-5_sample-1/
├── ...
...

```

First, load the Ensemble:
```
>>> ensemble_obj = foldkit.AF3Ensemble.load_af3_result("tutorial_example/ensemble_result")
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:03<00:00,  5.05it/s]

>>> ensemble_obj.size
20
>>> ensemble_obj.seeds
[1, 2, 5, 10]
>>> ensemble_obj.samples
[0, 1, 2, 3, 4]
```

The individual results are stored in `ensemble_obj.af3_results`. You get easy access to an individual structure in the ensemble by using its seed and sample:
```
>>> ensemble_obj.get_result_by_seed_and_sample(seed=5,sample=2)
<foldkit.af3_result.AF3Result object at 0x7f59d8a344b0>
```
The ranking scores are stored in `ensemble_obj.af3_ranking_scores`. Access to the top-ranked structure:
```
>>> ensemble_obj.get_top_ranked_result()
<foldkit.af3_result.AF3Result object at 0x7f215e6497f0>
```

You can also get any of the confidence metrics, either by structure, or aggregated over the entire ensemble:

```
>>> >>> ensemble_obj.get_all_plddt('M')
{'seed-1_sample-3': np.float64(78.71985788561527), 'seed-2_sample-4': np.float64(77.89603812824957), 'seed-1_sample-4': np.float64(78.20608318890814), ...}

>>> ensemble_obj.get_ensemble_plddt('M') # default is mean over ensemble
np.float64(78.2713937608319)

ensemble_obj.get_ensemble_plddt('M',ensemble_agg=max) # use max instead
np.float64(78.91594800693241)
```

Be careful not to mix up matrix agg with ensemble agg

```
# this is max of the plddt matrix, but mean over the ensemble
ensemble_obj.get_ensemble_plddt('M',agg=max, ensemble_agg=np.mean) 
np.float64(98.647)

# this is mean of the plddt matrix, but max over the ensemble
>>> ensemble_obj.get_ensemble_plddt('M',agg=numpy.mean, ensemble_agg=max)
np.float64(78.91594800693241)

```


### Full list of included metrics:
1) pLDDT
2) pAE and iPAE
4) pTM and ipTM
5) Contact Probs
6) ipSAE (see https://github.com/DunbrackLab/IPSAE)


### Loading from exported, compressed format
From the CLI, let's say you had previously exported the result of a AF3 run (see below), so that the result from before is now at `"tutorial_example/single_result_export"`. This second directory will have a .npz file in it instead of JSON files. You can load it in a very similar way:
```
foldkit.AF3Result.load_compressed_result('tutorial_example/single_result_export')
<foldkit.af3_result.AF3Result object at 0x7fe0bba2a120>
```
Similarly, for an ensemble that was exported in compressed format:
```
>>> foldkit.AF3Ensemble.load_compressed_result('tutorial_example/ensemble_result_export')
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 22.16it/s]
<foldkit.af3_ensemble.AF3Ensemble object at 0x7f3a374ea120>
```

### Loading from AF3 Server format
From the CLI, let's say you had previously exported the result of a AF3 run, so that the result from before
at `"structures/tcr_pmhc_1/"` is now at `"structures_compressed/tcr_pmhc_1/"`. This second directory will have a .npz file in it instead of JSON files. You can load it in a very similar way by adding the `from_npz=True` flag
```
result_obj = foldkit.AF3Result.load_result("structures/tcr_pmhc_1/", from_npz=True)
```

## folkdkit - CLI Tutorial
```
foldkit -h
usage: foldkit [-h] [--verbose] {export-single-result,export-ensemble-result,webserver-export,batch-export} ...

Export AlphaFold3 result directories into compressed format.Converts confidences into npz format and copies over the rest of the data as is (except the
_input_data.json which is redundant).

positional arguments:
  {export-single-result,export-ensemble-result,webserver-export,batch-export}
    export-single-result
                        Export a single AlphaFold3 result directory to compressed format
    export-ensemble-result
                        Export multiseed/multisample AlphaFold3 Ensemble of results to compressed format.
    webserver-export    Export AlphaFold3 server results to compressed format.
    batch-export        Export multiple AlphaFold3 results to compressed format.

options:
  -h, --help            show this help message and exit
  --verbose, -v         Print detailed output.
```
There are 4 main entry points, depending on the data you are exporting:
1) A single prediction directory (i.e. one prediction corresponding to a single seed and sample)
2) An ensemble prediction directory (i.e. N*K predictions corresponding to the same input with N seeds and K samples)
3) A directory of AF3 server results, one seed and multiple samples in a slightly diferent output format than the AF3 installation version.
4) A directory of ensemble directories (i.e. a directory containing many "ensemble directories" like in (2)).

After running the export successfully, you can safely delete the source directory.

  ### 1- Export a single result (i.e. one single structure from a single seed and sample)
  ```
foldkit export-single-result -h
usage: foldkit export-single-result [-h] input_directory output_directory

positional arguments:
  input_directory   Path to AlphaFold3 result directory
  output_directory  Output directory path

options:
  -h, --help        show this help message and exit
  ```
Example: 
```
foldkit -v export-single-result tutorial_example/single_result tutorial_example/single_result_export
✅ Exported Data to : tutorial_example/single_result_export
```

  ### 2- Export an ensemble of results with multiple seeds and/or samples
  ```
foldkit export-ensemble-result -h
usage: foldkit export-ensemble-result [-h] input_directory output_directory

positional arguments:
  input_directory   Path to parent directory containing subdirectories with AF3 results.
  output_directory  Parent output directory path

options:
  -h, --help        show this help message and exit
  ```
  Example:
  ```
  (foldkit) [levinej4@isca027 foldkit]$  foldkit -v export-ensemble-result tutorial_example/ensemble_result tutorial_example/ensemble_result_export
✅ Exported Data to : tutorial_example/ensemble_result_export/seed-1_sample-3
✅ Exported Data to : tutorial_example/ensemble_result_export/seed-2_sample-4
✅ Exported Data to : tutorial_example/ensemble_result_export/seed-2_sample-1
...
  ```

### 4- Export webserver result
Example:
```
foldkit -v webserver-export tutorial_example/server tutorial_example/server_export
✅ Exported Data to : tutorial_example/server_export
```


### 5- Batch export many results
  ```
foldkit batch-export -h
usage: foldkit batch-export [-h] input_directory output_directory

positional arguments:
  input_directory   Path to parent directory containing subdirectories with subdirectories of AF3 results.
  output_directory  Parent output directory path

options:
  -h, --help        show this help message and exit
  ```

### Cleanup
  After running any of the above exports successfully, you can safely delete the source directory/directories.
