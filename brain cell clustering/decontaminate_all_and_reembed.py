"""
188_decontaminate_all_and_reembed.py

Scripts 185-187 only decontaminated the 4,508 rescued lymphoid cells, so every
other cell type kept its raw (contaminated) counts and stayed poorly separated
(Mono/Mac kNN-purity 20%, Microglia 19%, Endothelial 3%, Pericytes 5%).  This
script applies the SAME scHPF factor-attribution denoising to EVERY cell:

  each cell keeps only the scHPF factors of its own lineage (+ neutral state
  factors) and the off-lineage programs are stripped.

Because Mono/Mac (factor F3) and Microglia (F8/F19) are separate lineages, each
loses the other's program and the two myeloid populations pull apart; likewise
Endothelial (F20) separates from mural/SMC-Per (F15), astro (F16) from oligo
(F4/F5) from OPC (F1), etc.  The lymphoid rescue + relabel from script 185 is
preserved.  Then PCA->Harmony->UMAP is recomputed on the fully denoised counts.

Inputs:
  cellmap/merged_qc_brain_remapped.h5ad      (raw counts, cell_type_v2, run)
  scHPF/allcell_schpf_cell_scores.csv        (Theta)
  scHPF/allcell_schpf_gene_scores.csv        (beta)
  cellmap/allcell_contamination_flags.csv    (lymphoid hard-rule rescue, script 185)
Outputs:
  cellmap/merged_qc_decontaminated_v2.h5ad
  Green2024/umap_decontam_all_coords.csv
  Green2024/umap_decontam_all_12cat.png      <- deliverable
"""
import time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import h5py
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

CMAP = Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap")
ROOT = Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
OUT  = Path(r"D:\Caterina\MERSCOPE\merged_analysis\Green2024")
H5   = CMAP / "merged_qc_brain_remapped.h5ad"

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Loading master h5ad ...")
with h5py.File(H5, "r") as f:
    og = f["obs"]
    obs_idx = [s.decode() if isinstance(s, bytes) else s
               for s in og[og.attrs.get("_index", "_index")][:]]
    def ocol(name):
        node = og[name]
        if isinstance(node, h5py.Group) and "categories" in node:
            cats = [s.decode() if isinstance(s, bytes) else s for s in node["categories"][:]]
            return pd.Categorical.from_codes(node["codes"][:], categories=cats).astype(str)
        arr = node[:]
        if arr.dtype.kind in ("O", "S"):
            arr = np.array([s.decode() if isinstance(s, bytes) else s for s in arr])
        return arr
    v2  = np.asarray(ocol("cell_type_v2")).astype(str)
    run = np.asarray(ocol("run")).astype(str)
    vg = f["var"]
    var_idx = [s.decode() if isinstance(s, bytes) else s
               for s in vg[vg.attrs.get("_index", "_index")][:]]
    g = f["layers/counts"]
    Xraw = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                      shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
n_obs, n_var = Xraw.shape

cell_df = pd.read_csv(ROOT / "allcell_schpf_cell_scores.csv", index_col=0).reindex(obs_idx)
gene_df = pd.read_csv(ROOT / "allcell_schpf_gene_scores.csv", index_col=0)
fcols = list(cell_df.columns); K = len(fcols)
Theta = cell_df.values.astype(np.float64)
Beta  = gene_df.reindex(var_idx)[fcols].values.astype(np.float64)

flags = pd.read_csv(CMAP / "allcell_contamination_flags.csv").set_index("cell_id").reindex(obs_idx)
hr_lin   = flags["hardrule_lineage"].astype(str).values
is_resc  = flags["is_contaminated"].fillna(False).values.astype(bool)

# ----------------------------------------------------------------------
# Refined factor -> lineage (clean the noisy mixed factors to 'neutral'
# so they are kept by everyone and cannot wrongly strip real genes).
# F-index is 0-based here (F1==index0).
# ----------------------------------------------------------------------
FACTOR_LIN = {
    "F1":"OPC","F2":"Neuron","F3":"MonoMac","F4":"Oli","F5":"Oli",
    "F6":"neutral","F7":"Lymphoid","F8":"Mic","F9":"neutral","F10":"neutral",
    "F11":"neutral","F12":"neutral","F13":"neutral","F14":"neutral","F15":"Mural",
    "F16":"Ast","F17":"neutral","F18":"Lymphoid","F19":"Mic","F20":"Endo",
}
flin = np.array([FACTOR_LIN[fc] for fc in fcols])
# keep-set per lineage = own-lineage factors + all neutral factors
LINEAGES = ["Lymphoid","MonoMac","Mic","Ast","Oli","OPC","Neuron","Endo","Mural"]
keepvec = {lin: ((flin == lin) | (flin == "neutral")) for lin in LINEAGES}

# ----------------------------------------------------------------------
# Per-cell lineage assignment:
#   1) start from prior cell_type_v2 mapped to a lineage
#   2) data-driven Mono-vs-Mic split for myeloid-labelled cells (dominant
#      myeloid factor F3 vs F8+F19)
#   3) override with the script-185 lymphoid hard-rule rescue
# ----------------------------------------------------------------------
V2_TO_LIN = {"Ast":"Ast","B":"Lymphoid","End":"Endo","Exc":"Neuron","Inh":"Neuron",
             "Mic":"Mic","Mono/Mac":"MonoMac","OPC":"OPC","Oli":"Oli","Per":"Mural",
             "SMC":"Mural","T/NK":"Lymphoid","Amb":"none"}
cell_lin = np.array([V2_TO_LIN.get(x, "none") for x in v2], dtype=object)

iF3 = fcols.index("F3"); iF8 = fcols.index("F8"); iF19 = fcols.index("F19")
myelo = np.isin(v2, ["Mic", "Mono/Mac"])
mono_score = Theta[:, iF3]; mic_score = Theta[:, iF8] + Theta[:, iF19]
cell_lin[myelo & (mono_score >= mic_score)] = "MonoMac"
cell_lin[myelo & (mono_score <  mic_score)] = "Mic"

resc_map = {"Tcell":"Lymphoid","NK":"Lymphoid","Bcell":"Lymphoid"}
for i in np.where(is_resc)[0]:
    cell_lin[i] = resc_map.get(hr_lin[i], cell_lin[i])

print("\nPer-cell lineage assignment:")
print(pd.Series(cell_lin).value_counts())

# ----------------------------------------------------------------------
# Global factor-attribution strip (vectorised, chunked over cells)
# ----------------------------------------------------------------------
print(f"\n[{time.strftime('%H:%M:%S')}] Stripping off-lineage programs from all cells ...")
keepmask = np.ones((n_obs, K), dtype=bool)        # 'none' -> keep all
for lin in LINEAGES:
    rows = np.where(cell_lin == lin)[0]
    keepmask[rows] = keepvec[lin]

Xdec = Xraw.copy()
indptr, indices = Xdec.indptr, Xdec.indices
CH = 50000
for c0 in range(0, n_obs, CH):
    c1 = min(c0 + CH, n_obs)
    seg0, seg1 = indptr[c0], indptr[c1]
    cols = indices[seg0:seg1]
    rowlens = np.diff(indptr[c0:c1 + 1])
    localrows = np.repeat(np.arange(c1 - c0), rowlens)
    Th = Theta[c0:c1]
    full = Th @ Beta.T
    keep = (Th * keepmask[c0:c1]) @ Beta.T
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(full > 0, keep / full, 1.0)
    np.clip(frac, 0.0, 1.0, out=frac)
    Xdec.data[seg0:seg1] = np.rint(Xdec.data[seg0:seg1] * frac[localrows, cols])
Xdec.eliminate_zeros()
print(f"  done.  nnz {Xraw.nnz:,} -> {Xdec.nnz:,}  "
      f"(kept {Xdec.nnz/Xraw.nnz*100:.1f}% of nonzeros)")

# sanity: vascular genes in myeloid/neuron cells should drop; lineage genes kept
def mean_in(gene, lin):
    j = var_idx.index(gene); m = cell_lin == lin
    return float(np.asarray(Xraw[m, j].todense()).mean()), float(np.asarray(Xdec[m, j].todense()).mean())
for gene, lin in [("PECAM1","MonoMac"),("CX3CR1","MonoMac"),("CD163","MonoMac"),
                  ("CD163","Mic"),("CX3CR1","Mic"),("MOG","Ast"),("PECAM1","Mural")]:
    b, a = mean_in(gene, lin)
    print(f"  {lin:8} mean {gene:7} {b:6.3f} -> {a:6.3f}")

# ----------------------------------------------------------------------
# Corrected 12-cat category
# ----------------------------------------------------------------------
LIN_TO_CAT = {"Lymphoid":"Immune","MonoMac":"Mono/Mac","Mic":"Microglia",
              "Ast":"Astrocytes","Oli":"Oligodendrocytes","OPC":"OPC",
              "Endo":"Endothelial","Mural":None,  # split back to Per/SMC by v2
              "Neuron":None, "none":"Ambiguous"}
V2_TO_CAT = {"Exc":"Excitatory neurons","Inh":"Inhibitory neurons","Per":"Pericytes",
             "SMC":"SMC","Amb":"Ambiguous"}
cat = np.empty(n_obs, dtype=object)
for i in range(n_obs):
    c = LIN_TO_CAT.get(cell_lin[i], None)
    cat[i] = c if c is not None else V2_TO_CAT.get(v2[i], "Ambiguous")
print("\ncategory distribution:")
print(pd.Series(cat).value_counts())

# ----------------------------------------------------------------------
# Write decontaminated v2 + re-embed (mirror script 69 / 187)
# ----------------------------------------------------------------------
obs = pd.DataFrame(index=obs_idx)
obs["run"] = run; obs["cell_type_v2"] = v2
obs["cell_lineage"] = cell_lin; obs["category_corrected"] = cat
obs["is_contaminated"] = is_resc
a = ad.AnnData(X=Xdec.copy(), obs=obs, var=pd.DataFrame(index=var_idx))
a.layers["counts_decontam"] = Xdec
a.write(CMAP / "merged_qc_decontaminated_v2.h5ad", compression="gzip")
print(f"\n[{time.strftime('%H:%M:%S')}] Saved merged_qc_decontaminated_v2.h5ad")

sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=400, flavor="seurat")
sc.pp.scale(a, max_value=10)
sc.tl.pca(a, n_comps=40, use_highly_variable=False)
print(f"[{time.strftime('%H:%M:%S')}] Harmony ...")
sc.external.pp.harmony_integrate(a, key="run", basis="X_pca",
                                 adjusted_basis="X_pca_harmony", max_iter_harmony=20)
print(f"[{time.strftime('%H:%M:%S')}] neighbors + UMAP ...")
sc.pp.neighbors(a, use_rep="X_pca_harmony", n_neighbors=20)
sc.tl.umap(a, min_dist=0.3, random_state=1234)
emb = a.obsm["X_umap"]

coords = pd.DataFrame({"cell_id": obs_idx, "umap_x": emb[:,0], "umap_y": emb[:,1],
                       "category": cat, "is_contaminated": is_resc})
coords.to_csv(OUT / "umap_decontam_all_coords.csv", index=False)

# per-category kNN purity
tree = cKDTree(emb); nn = tree.query(emb, k=16)[1][:,1:]
print("\n=== per-category kNN purity (k=15) after global decontam ===")
for c in ["Mono/Mac","Microglia","Immune","Astrocytes","Endothelial",
          "Oligodendrocytes","Excitatory neurons","Inhibitory neurons",
          "SMC","Pericytes","OPC"]:
    m = np.where(cat == c)[0]
    if len(m) == 0: continue
    print(f"  {c:20} {float((cat[nn[m]]==c).mean())*100:5.1f}%   n={len(m):,}")

# plot
CT_ORDER = ["Excitatory neurons","Inhibitory neurons","Astrocytes","Microglia",
            "Oligodendrocytes","OPC","Endothelial","Pericytes","SMC",
            "Mono/Mac","Immune","Ambiguous"]
CT_CLR = {"Excitatory neurons":"#3F2DA6","Inhibitory neurons":"#4CAF50",
    "Astrocytes":"#FFC107","Microglia":"#00897B","Oligodendrocytes":"#E91E63",
    "OPC":"#FF6F00","Endothelial":"#03A9F4","Pericytes":"#AD1457",
    "SMC":"#BCAAA4","Mono/Mac":"#D7263D","Immune":"#1B0E55","Ambiguous":"#9E9E9E"}
fig, ax = plt.subplots(figsize=(13, 9))
present = [c for c in CT_ORDER if (cat == c).any()]
for ct in sorted(present, key=lambda c: -int((cat==c).sum())):
    m = cat == ct
    small = ct in ("Immune","Mono/Mac","Pericytes","Endothelial","SMC")
    ax.scatter(emb[m,0], emb[m,1], c=CT_CLR.get(ct,"#888"), alpha=0.9,
               s=5 if small else 3, edgecolors="none",
               label=f"{ct} (n={int(m.sum()):,})")
for ct in present:
    m = cat == ct
    if m.sum() < 200: continue
    ax.text(float(np.median(emb[m,0])), float(np.median(emb[m,1])), ct,
            color="white", fontsize=10, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#4A4A4A",
                      edgecolor="none", alpha=0.92), zorder=40)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("All QC'd cells — UMAP on globally decontaminated counts\n"
             "(scHPF factor attribution per lineage; Mono/Mac vs Microglia split)",
             fontsize=12, fontweight="bold")
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False,
          fontsize=9, markerscale=3, labelspacing=0.6)
plt.tight_layout()
plt.savefig(OUT / "umap_decontam_all_12cat.png", dpi=170, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT / 'umap_decontam_all_12cat.png'}")
print(f"Total: {time.time()-t0:.1f}s")
