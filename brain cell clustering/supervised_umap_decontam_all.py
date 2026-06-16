"""
189_supervised_umap_decontam_all.py

Polished deliverable: supervised UMAP on the GLOBALLY DECONTAMINATED counts
(script 188) using the corrected 12-category labels as a categorical target.

Why this is the right combination now:
  * Features are clean — script 188 stripped off-lineage scHPF programs from
    every cell, so a cell's PCs reflect its true lineage (this is what made the
    failed script 186 fail: there the features were still contaminated).
  * Labels are corrected — lymphoid rescued (185) + Mono/Mac vs Microglia split
    by dominant myeloid factor (188).
  * The categorical target then concentrates each corrected label into a tight
    island, cleanly separating all 12 types — including Mono/Mac from Microglia,
    and the panel-limited pairs (Exc/Inh, SMC/Per) that share scHPF factors and
    so cannot separate by features alone.

Input  : cellmap/merged_qc_decontaminated_v2.h5ad  (layers/counts_decontam,
          category_corrected, run)
Outputs (Green2024/):
  umap_decontam_all_supervised_coords.csv
  umap_decontam_all_supervised_12cat.png   <- main deliverable
"""
import time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import umap
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

CMAP = Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap")
OUT  = Path(r"<MERSCOPE_ROOT>\merged_analysis\Green2024")
H5   = CMAP / "merged_qc_decontaminated_v2.h5ad"

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Loading {H5.name} ...")
A = ad.read_h5ad(H5)
a = ad.AnnData(X=A.layers["counts_decontam"].copy(), obs=A.obs.copy(),
               var=A.var.copy())

# Standard recipe -> Harmony PCs on the decontaminated counts (mirror 69/188)
sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=400, flavor="seurat")
sc.pp.scale(a, max_value=10)
sc.tl.pca(a, n_comps=40, use_highly_variable=False)
print(f"[{time.strftime('%H:%M:%S')}] Harmony ...")
sc.external.pp.harmony_integrate(a, key="run", basis="X_pca",
                                 adjusted_basis="X_pca_harmony", max_iter_harmony=20)
Xph = a.obsm["X_pca_harmony"]

cat = a.obs["category_corrected"].astype(str).values
is_contam = a.obs["is_contaminated"].values.astype(bool)
cats = pd.Categorical(cat); y = cats.codes.astype(np.int32)

print(f"[{time.strftime('%H:%M:%S')}] Supervised UMAP (target_weight=0.5) ...")
emb = umap.UMAP(n_neighbors=30, min_dist=0.3, target_metric="categorical",
                target_weight=0.5, random_state=1234, verbose=True,
                n_jobs=-1).fit_transform(Xph, y=y)

coords = pd.DataFrame({"cell_id": a.obs_names, "umap_x": emb[:,0],
                       "umap_y": emb[:,1], "category": cat,
                       "is_contaminated": is_contam})
coords.to_csv(OUT / "umap_decontam_all_supervised_coords.csv", index=False)

# purity
tree = cKDTree(emb); nn = tree.query(emb, k=16)[1][:,1:]
print("\n=== per-category kNN purity (supervised, decontaminated) ===")
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
ax.set_title("All QC'd cells — decontaminated + supervised UMAP\n"
             "(scHPF factor attribution per lineage + corrected labels)",
             fontsize=12, fontweight="bold")
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.02,0.5), frameon=False,
          fontsize=9, markerscale=3, labelspacing=0.6)
plt.tight_layout()
plt.savefig(OUT / "umap_decontam_all_supervised_12cat.png", dpi=170, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT / 'umap_decontam_all_supervised_12cat.png'}")
print(f"Total: {time.time()-t0:.1f}s")
