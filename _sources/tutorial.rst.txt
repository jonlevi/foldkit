
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
