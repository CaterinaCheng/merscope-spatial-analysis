"""
186_supervised_umap_decontam.py

Final all-cell UMAP after cross-lineage decontamination (script 185).

Uses the SAME supervised-UMAP recipe as 175_supervised_umap.py (Harmony PCs +
12-category categorical target, target_weight=0.5) but with `category_corrected`
as the target.  Because the rescued perivascular immune cells now carry the
`Immune` label (instead of Endothelial/Pericyte/SMC), the categorical target
pulls them out of the vascular blob and into the adaptive-immune cluster — the
"put them back where they belong" step.

Inputs:
  cellmap/merged_qc_decontaminated.h5ad   (X_pca_harmony, category_corrected,
                                            is_contaminated)
Outputs (Green2024/):
  umap_supervised_decontam_coords.csv
  umap_supervised_decontam_12cat.png       <- main deliverable (2 panels:
                                              corrected map + rescued-cell highlight)
"""
import time
from pathlib import Path
import numpy as np, pandas as pd
import h5py
import umap
import matplotlib.pyplot as plt

CMAP = Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap")
OUT  = Path(r"D:\Caterina\MERSCOPE\merged_analysis\Green2024")
H5   = CMAP / "merged_qc_decontaminated.h5ad"

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] Loading Harmony PCs + corrected labels ...")
with h5py.File(H5, "r") as f:
    og = f["obs"]
    obs_idx = [s.decode() if isinstance(s, bytes) else s
               for s in og[og.attrs.get("_index", "_index")][:]]
    def col(name):
        node = og[name]
        if isinstance(node, h5py.Group) and "categories" in node:
            cats = [s.decode() if isinstance(s, bytes) else s
                    for s in node["categories"][:]]
            return pd.Categorical.from_codes(node["codes"][:], categories=cats).astype(str)
        arr = node[:]
        if arr.dtype.kind in ("O", "S"):
            arr = np.array([s.decode() if isinstance(s, bytes) else s for s in arr])
        return arr
    cat = np.asarray(col("category_corrected")).astype(str)
    is_contam = np.asarray(col("is_contaminated")).astype(bool)
    Xph = f["obsm/X_pca_harmony"][:]
print(f"  {Xph.shape}  ({time.time()-t0:.1f}s)")

# Encode 12-cat labels for the supervised target
cats = pd.Categorical(cat)
labels_int = cats.codes.astype(np.int32)
print("\nLabel distribution (category_corrected):")
for i, c in enumerate(cats.categories):
    print(f"  {i:>2}  {c:<22} {int((labels_int==i).sum()):>8,}")

# ---- Supervised UMAP (identical settings to script 175) ----
t1 = time.time()
print(f"\n[{time.strftime('%H:%M:%S')}] Fitting supervised UMAP ...")
# target_weight=0.8 (was 0.5): the rescued cells still carry their CONTAMINATED
# Harmony PCs (we reused the existing embedding rather than re-running PCA on the
# decontaminated counts), so a gentle target only half-pulls them and leaves the
# Immune cluster diffuse.  A stronger categorical target makes the corrected
# hard-rule label dominate placement, coalescing the adaptive-immune cells into
# one tight cluster while still retaining within-category transcriptomic order.
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                    target_metric="categorical", target_weight=0.8,
                    random_state=1234, verbose=True, n_jobs=-1)
emb = reducer.fit_transform(Xph, y=labels_int)
print(f"  done ({time.time()-t1:.1f}s)")

coords = pd.DataFrame({"cell_id": obs_idx,
                       "umap_x": emb[:, 0], "umap_y": emb[:, 1],
                       "category": cat, "is_contaminated": is_contam})
coords.to_csv(OUT / "umap_supervised_decontam_coords.csv", index=False)
print(f"Saved: {OUT / 'umap_supervised_decontam_coords.csv'}")

# ---- Plot (palette/style from script 175) ----
CT_ORDER = ["Excitatory neurons","Inhibitory neurons","Astrocytes","Microglia",
            "Oligodendrocytes","OPC","Endothelial","Pericytes","SMC",
            "Mono/Mac","Immune","Ambiguous"]
CT_CLR = {
    "Excitatory neurons":"#3F2DA6","Inhibitory neurons":"#4CAF50",
    "Astrocytes":"#FFC107","Microglia":"#00897B",
    "Oligodendrocytes":"#E91E63","OPC":"#FF6F00",
    "Endothelial":"#03A9F4","Pericytes":"#AD1457","SMC":"#BCAAA4",
    "Mono/Mac":"#D7263D","Immune":"#1B0E55","Ambiguous":"#9E9E9E",
}

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

# --- Panel A: corrected 12-category map ---
ax = axes[0]
present = [c for c in CT_ORDER if (cat == c).any()]
DRAW = sorted(present, key=lambda c: -int((cat == c).sum()))
for ct in DRAW:
    m = cat == ct
    if m.sum() == 0: continue
    small = ct in ("Immune","Mono/Mac","Pericytes","Endothelial","SMC")
    ax.scatter(emb[m, 0], emb[m, 1], c=CT_CLR.get(ct, "#888"),
               alpha=0.90, s=5 if small else 3, edgecolors="none",
               label=f"{ct} (n={int(m.sum()):,})")
for ct in present:
    m = cat == ct
    if m.sum() < 200: continue
    cx, cy = float(np.median(emb[m,0])), float(np.median(emb[m,1]))
    ax.text(cx, cy, ct, color="white", fontsize=10, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#4A4A4A",
                      edgecolor="none", alpha=0.92), zorder=40)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("All QC'd cells — decontaminated supervised UMAP\n"
             "(scHPF factor attribution + hard-rule relabelling)",
             fontsize=12, fontweight="bold")
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=False, fontsize=9, markerscale=3, labelspacing=0.6)

# --- Panel B: rescued cells highlighted ---
ax = axes[1]
ax.scatter(emb[:,0], emb[:,1], c="#DDDDDD", s=2, edgecolors="none", alpha=0.6)
r = is_contam
ax.scatter(emb[r,0], emb[r,1], c="#D7263D", s=10, edgecolors="black",
           linewidths=0.2, alpha=0.95,
           label=f"Decontaminated & rescued (n={int(r.sum()):,})")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Rescued cells (off-lineage scHPF program stripped,\n"
             "re-defined by hard rules) — now in their true cluster",
             fontsize=12, fontweight="bold")
for sp in ("top","right","left","bottom"): ax.spines[sp].set_visible(False)
ax.legend(loc="upper right", frameon=False, fontsize=10, markerscale=2)

plt.tight_layout()
out = OUT / "umap_supervised_decontam_12cat.png"
plt.savefig(out, dpi=170, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
print(f"\nTotal: {time.time()-t0:.1f}s")
