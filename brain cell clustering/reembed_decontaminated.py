"""
187_reembed_decontaminated.py

The supervised-UMAP-on-existing-Harmony-PCs approach (script 186) could NOT
relocate the rescued cells: their 40 Harmony PCs were computed from the
CONTAMINATED counts, so they still encode vascular/glial identity and the
categorical target (even at weight 0.8) only moved 25% of them into the Immune
cluster.

This script does the principled fix: recompute the whole feature space
(PCA -> Harmony) from the DECONTAMINATED counts, mirroring the established
Harmony recipe in 69_harmony_celltype.py (normalize 1e4 -> log1p -> HVG 400 ->
scale -> PCA 40 -> harmony key='run').  Now a rescued T cell — whose cleaned
profile is CD3+ with the vascular genes removed — genuinely neighbours the other
immune cells, so even an UNSUPERVISED UMAP places it in the adaptive-immune
cluster.  The cells move because their biology moved.

Input  : cellmap/merged_qc_decontaminated.h5ad   (layers/counts_decontam)
Outputs (Green2024/):
  umap_decontam_reembed_coords.csv
  umap_decontam_reembed_12cat.png     <- main deliverable
"""
import time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

CMAP = Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap")
OUT  = Path(r"D:\Caterina\MERSCOPE\merged_analysis\Green2024")
H5   = CMAP / "merged_qc_decontaminated.h5ad"

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Loading {H5.name} ...")
A = ad.read_h5ad(H5)
print(f"  {A.n_obs:,} cells x {A.n_vars} genes")

# Rebuild AnnData on the DECONTAMINATED counts
a = ad.AnnData(X=A.layers["counts_decontam"].copy(),
               obs=A.obs.copy(), var=A.var.copy())

# --- Established Harmony recipe (mirror 69_harmony_celltype.py) ---
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
sc.pp.highly_variable_genes(a, n_top_genes=400, flavor="seurat")
sc.pp.scale(a, max_value=10)
sc.tl.pca(a, n_comps=40, use_highly_variable=False)
print(f"[{time.strftime('%H:%M:%S')}] Harmony (key='run') ...")
sc.external.pp.harmony_integrate(a, key="run", basis="X_pca",
                                 adjusted_basis="X_pca_harmony",
                                 max_iter_harmony=20)
print(f"[{time.strftime('%H:%M:%S')}] neighbors + UMAP ...")
sc.pp.neighbors(a, use_rep="X_pca_harmony", n_neighbors=20)
sc.tl.umap(a, min_dist=0.3, random_state=1234)
emb = a.obsm["X_umap"]
print(f"  embedding done ({time.time()-t0:.1f}s)")

cat = a.obs["category_corrected"].astype(str).values
is_contam = a.obs["is_contaminated"].values.astype(bool)

coords = pd.DataFrame({"cell_id": a.obs_names, "umap_x": emb[:,0],
                       "umap_y": emb[:,1], "category": cat,
                       "is_contaminated": is_contam})
coords.to_csv(OUT / "umap_decontam_reembed_coords.csv", index=False)
print(f"Saved: {OUT / 'umap_decontam_reembed_coords.csv'}")

# ---- Verify rescued cells now sit with the immune cells ----
cents = coords.groupby("category")[["umap_x","umap_y"]].median()
cn, C = cents.index.values, cents.values
P = coords.loc[is_contam, ["umap_x","umap_y"]].values
nearest = cn[np.linalg.norm(P[:,None,:]-C[None,:,:], axis=2).argmin(1)]
tree = cKDTree(coords[["umap_x","umap_y"]].values)
isimm = (cat == "Immune")
nn = tree.query(P, k=31)[1][:,1:]
frac_imm = isimm[nn].mean(1)
print(f"\n=== Rescued-cell placement after re-embed ===")
print(f"  nearest centroid == Immune: {np.mean(nearest=='Immune')*100:.1f}%")
print(f"  mean fraction of 30 NN immune-labelled: {frac_imm.mean()*100:.1f}%")
print(f"  rescued with >50% immune neighbours: {np.mean(frac_imm>0.5)*100:.1f}%")

# ---- Plot (style from script 186) ----
CT_ORDER = ["Excitatory neurons","Inhibitory neurons","Astrocytes","Microglia",
            "Oligodendrocytes","OPC","Endothelial","Pericytes","SMC",
            "Mono/Mac","Immune","Ambiguous"]
CT_CLR = {"Excitatory neurons":"#3F2DA6","Inhibitory neurons":"#4CAF50",
    "Astrocytes":"#FFC107","Microglia":"#00897B","Oligodendrocytes":"#E91E63",
    "OPC":"#FF6F00","Endothelial":"#03A9F4","Pericytes":"#AD1457",
    "SMC":"#BCAAA4","Mono/Mac":"#D7263D","Immune":"#1B0E55","Ambiguous":"#9E9E9E"}

fig, axes = plt.subplots(1, 2, figsize=(24, 9))
ax = axes[0]
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
            color="white", fontsize=10, fontweight="bold", ha="center",
            va="center", bbox=dict(boxstyle="round,pad=0.32",
            facecolor="#4A4A4A", edgecolor="none", alpha=0.92), zorder=40)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("All QC'd cells — UMAP re-embedded on DECONTAMINATED counts\n"
             "(scHPF factor attribution + hard-rule relabelling)",
             fontsize=12, fontweight="bold")
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01,0.5), frameon=False,
          fontsize=9, markerscale=3, labelspacing=0.6)

ax = axes[1]
ax.scatter(emb[:,0], emb[:,1], c="#DDDDDD", s=2, edgecolors="none", alpha=0.6)
ax.scatter(emb[is_contam,0], emb[is_contam,1], c="#D7263D", s=10,
           edgecolors="black", linewidths=0.2, alpha=0.95,
           label=f"Decontaminated & rescued (n={int(is_contam.sum()):,})")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Rescued perivascular/parenchymal immune cells\n"
             "now embed in the adaptive-immune cluster",
             fontsize=12, fontweight="bold")
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=10, markerscale=2)

plt.tight_layout()
out = OUT / "umap_decontam_reembed_12cat.png"
plt.savefig(out, dpi=170, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
print(f"Total: {time.time()-t0:.1f}s")
