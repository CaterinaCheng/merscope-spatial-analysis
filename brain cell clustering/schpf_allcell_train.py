"""
184_schpf_allcell_train.py

Train scHPF on ALL QC'd brain cells (not just the T/NK subset that script
135 used).  This all-cell factor model is the engine for the cross-lineage
decontamination in script 185: each factor is a gene program, and a
perivascular T cell that leaks vascular transcripts (PECAM1 etc.) will load
on BOTH a T-cell factor and a vascular factor.  Script 185 strips the
off-lineage factor share to remove the contamination.

Input  : merged_analysis/cellmap/merged_qc_brain_remapped.h5ad
         (layers/counts, int32, 617,399 cells x 550 genes)
Outputs (merged_analysis/scHPF/):
  allcell_schpf_K20_best.joblib    trained model
  allcell_schpf_cell_scores.csv    per-cell factor scores  (Theta, cells x 20)
  allcell_schpf_gene_scores.csv    per-gene factor scores  (beta,  genes x 20)
  allcell_schpf_factor_top25.csv   top-25 genes per factor (annotation aid)

Mirrors 135_train_scHPF.py.  K=20 (was 10 for the T/NK-only model): the
all-cell matrix needs more factors to separate ~13 lineages + immune-state
programs + ambient/contamination programs.  Run in the background.
"""
from pathlib import Path
import time
import numpy as np, pandas as pd
import h5py
from scipy.sparse import csr_matrix
from schpf import run_trials, save_model

CMAP = Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap")
ROOT = Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
H5   = CMAP / "merged_qc_brain_remapped.h5ad"

NFAC, NTRIALS, MAX_IT = 20, 3, 500

# ---- Load raw counts (cells x genes) directly from the master h5ad ----
print(f"[{time.strftime('%H:%M:%S')}] Loading counts from {H5.name} ...")
with h5py.File(H5, "r") as f:
    og = f["obs"]; vg = f["var"]
    cell_ids = [s.decode() if isinstance(s, bytes) else s
                for s in og[og.attrs.get("_index", "_index")][:]]
    gene_ids = [s.decode() if isinstance(s, bytes) else s
                for s in vg[vg.attrs.get("_index", "_index")][:]]
    g = f["layers/counts"]
    X = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                   shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.int32)
print(f"  cells x genes: {X.shape}, nnz: {X.nnz:,}")

# scHPF wants cells x genes COO of integer counts
X = X.tocoo()

t0 = time.time()
print(f"\n[{time.strftime('%H:%M:%S')}] Training scHPF: "
      f"K={NFAC}, trials={NTRIALS}, max_iter={MAX_IT} ...")
best = run_trials(X=X, nfactors=NFAC, ntrials=NTRIALS,
                  max_iter=MAX_IT, verbose=True)
print(f"\n[{time.strftime('%H:%M:%S')}] Training done in "
      f"{(time.time()-t0)/60:.1f} min")

save_model(best, str(ROOT / f"allcell_schpf_K{NFAC}_best.joblib"))

fcols = [f"F{k+1}" for k in range(NFAC)]
cell_df = pd.DataFrame(best.cell_score(), index=cell_ids, columns=fcols)
gene_df = pd.DataFrame(best.gene_score(), index=gene_ids, columns=fcols)
cell_df.to_csv(ROOT / "allcell_schpf_cell_scores.csv")
gene_df.to_csv(ROOT / "allcell_schpf_gene_scores.csv")

top25 = pd.DataFrame({fc: gene_df[fc].nlargest(25).index.tolist()
                      for fc in fcols})
top25.to_csv(ROOT / "allcell_schpf_factor_top25.csv", index=False)

print("\nTop-12 genes per factor:")
print(top25.head(12).to_string(index=False))
print(f"\nSaved model + scores under {ROOT}")
print(f"[{time.strftime('%H:%M:%S')}] ALL DONE  (total "
      f"{(time.time()-t0)/60:.1f} min)")
