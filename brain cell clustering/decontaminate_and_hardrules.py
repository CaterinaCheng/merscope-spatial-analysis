"""
185_decontaminate_and_hardrules.py

Cross-lineage decontamination of the all-cell cohort using the scHPF model
trained in script 184, following the user's 4-step recipe:

  (1) MOVE TOGETHER   - find cells whose dominant scHPF program belongs to a
                        different lineage compartment than their hard-rule
                        identity (e.g. a CD3+ T cell whose dominant factor is
                        the vascular/PECAM1 program -> a perivascular T cell
                        that leaked vascular transcripts).
  (2) REMOVE CONTAM   - scHPF factor attribution: split each gene's expected
                        counts across factors (mu_ijk = theta_ik * beta_jk) and
                        keep only the share from factors in the cell's own
                        lineage compartment (+ neutral/state factors).  The
                        off-compartment share (the contamination) is stripped.
  (3) HARD RULES ONLY - the flagged cells are (re)defined purely by canonical
                        lineage-defining panel markers, not by their (now
                        partly ambient) transcriptome.
  (4) PUT BACK        - write the corrected 12-category label so the supervised
                        UMAP in script 186 places them with the cluster they
                        biologically belong to.

Inputs:
  cellmap/merged_qc_brain_remapped.h5ad         (counts, obs labels, Harmony PCs)
  scHPF/allcell_schpf_cell_scores.csv           (Theta, cells x 20)
  scHPF/allcell_schpf_gene_scores.csv           (beta,  genes x 20)
  scHPF/allcell_schpf_factor_top25.csv          (annotation aid)

Outputs:
  scHPF/allcell_factor_lineage.csv              factor -> lineage / compartment
  cellmap/allcell_contamination_flags.csv       per-cell decision table
  cellmap/merged_qc_decontaminated.h5ad         counts_decontam + corrected obs
"""
from pathlib import Path
import time
import numpy as np, pandas as pd, anndata as ad
import h5py
from scipy.sparse import csr_matrix

CMAP = Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap")
ROOT = Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
H5   = CMAP / "merged_qc_brain_remapped.h5ad"

# ======================================================================
# 0.  Load master h5ad (obs labels + raw counts + Harmony PCs) via h5py
# ======================================================================
print(f"[{time.strftime('%H:%M:%S')}] Loading {H5.name} ...")
with h5py.File(H5, "r") as f:
    og = f["obs"]
    cols = list(og.attrs.get("column-order", []))
    obs_idx = [s.decode() if isinstance(s, bytes) else s
               for s in og[og.attrs.get("_index", "_index")][:]]
    obs = pd.DataFrame(index=obs_idx)
    for c in cols:
        node = og[c]
        if isinstance(node, h5py.Group) and "categories" in node:
            cats = [s.decode() if isinstance(s, bytes) else s
                    for s in node["categories"][:]]
            obs[c] = pd.Categorical.from_codes(node["codes"][:], categories=cats)
        else:
            arr = node[:]
            if arr.dtype.kind in ("O", "S"):
                arr = np.array([s.decode() if isinstance(s, bytes) else s
                                for s in arr])
            obs[c] = arr
    vg = f["var"]
    var_idx = [s.decode() if isinstance(s, bytes) else s
               for s in vg[vg.attrs.get("_index", "_index")][:]]
    g = f["layers/counts"]
    Xraw = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                      shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    Xph = f["obsm/X_pca_harmony"][:]
    Xum = f["obsm/X_umap"][:] if "obsm/X_umap" in f else None

n_obs, n_var = Xraw.shape
var_pos = {g_: i for i, g_ in enumerate(var_idx)}
print(f"  cells={n_obs:,}  genes={n_var}")

# ======================================================================
# 1.  Load scHPF scores and annotate each factor -> lineage / compartment
# ======================================================================
cell_df = pd.read_csv(ROOT / "allcell_schpf_cell_scores.csv", index_col=0)
gene_df = pd.read_csv(ROOT / "allcell_schpf_gene_scores.csv", index_col=0)
cell_df = cell_df.reindex(obs_idx)             # align to master order
fcols = list(cell_df.columns)
K = len(fcols)
Theta = cell_df[fcols].values.astype(np.float64)         # cells x K
Beta_full = gene_df.reindex(var_idx)[fcols].values.astype(np.float64)  # genes x K
print(f"  scHPF factors: K={K}")

# Lineage marker dictionaries (all verified present in the 550 panel)
LINEAGE_MARKERS = {
    "Tcell":   ["CD3D","CD3E","CD3G","CD2","CD8A","CD8B","CD4","IL7R","CCL5"],
    "NK":      ["KLRD1","KLRF1","NKG7","GNLY","KLRC1","FCGR3A","PRF1"],
    "Bcell":   ["MS4A1","CD79A","CD79B","CD19","EBF1","IGHM"],
    "MonoMac": ["CD68","AIF1","CD14","LYZ","MRC1","CD163"],
    "Mic":     ["CX3CR1","TMEM119","CSF1R","C1QA","C1QB","C1QC"],
    "Oli":     ["MOG","MAL"],
    "OPC":     ["PDGFRA","OLIG2"],
    "Ast":     ["AQP4","GJA1"],
    "Exc":     ["RBFOX3","RORB"],
    "Inh":     ["GAD1","PVALB"],
    "Endo":    ["PECAM1"],
    "Per":     ["PDGFRB"],
    "SMC":     ["ACTA2"],
}
# Broad compartments: decontamination keeps factors in the SAME compartment as
# the cell's hard-rule lineage (+ neutral), strips off-compartment factors.
COMPARTMENT = {
    "Tcell":"lymphoid","NK":"lymphoid","Bcell":"lymphoid",
    "MonoMac":"myeloid","Mic":"myeloid",
    "Ast":"glia","Oli":"glia","OPC":"glia",
    "Exc":"neuron","Inh":"neuron",
    "Endo":"vascular","Per":"vascular","SMC":"vascular",
    "neutral":"neutral",
}

# Annotate each factor by its top-25 beta genes: which lineage's markers are
# most enriched at the top of the factor.  Factors with no clear lineage hit
# are "neutral" (state/ambient programs) and are always KEPT (never stripped).
top25 = {fc: gene_df[fc].nlargest(25).index.tolist() for fc in fcols}
factor_lineage = {}
rows = []
for fc in fcols:
    tops = top25[fc]
    rank = {g_: 25 - i for i, g_ in enumerate(tops)}   # weight by rank (top=25)
    scores = {}
    for lin, mks in LINEAGE_MARKERS.items():
        scores[lin] = sum(rank.get(m, 0) for m in mks)
    best = max(scores, key=scores.get)
    best_score = scores[best]
    lin = best if best_score > 0 else "neutral"
    factor_lineage[fc] = lin
    rows.append({"factor": fc, "lineage": lin,
                 "compartment": COMPARTMENT[lin],
                 "score": best_score,
                 "top10": ", ".join(tops[:10])})
fl_df = pd.DataFrame(rows)
fl_df.to_csv(ROOT / "allcell_factor_lineage.csv", index=False)
print("\n=== Factor -> lineage annotation (review the top genes) ===")
print(fl_df[["factor","lineage","compartment","score","top10"]].to_string(index=False))

factor_comp = np.array([COMPARTMENT[factor_lineage[fc]] for fc in fcols])

# ======================================================================
# 2.  Hard-rule lineage per cell from canonical lineage-defining markers
# ======================================================================
def sig(genes):
    idx = [var_pos[g_] for g_ in genes if g_ in var_pos]
    if not idx:
        return np.zeros(n_obs, dtype=np.float32)
    return np.asarray(Xraw[:, idx].sum(axis=1)).ravel()

def ndistinct(genes):
    """# of distinct markers in the list with count >= 1 (robustness against
    a single stray/ambient transcript)."""
    idx = [var_pos[g_] for g_ in genes if g_ in var_pos]
    if not idx:
        return np.zeros(n_obs, dtype=np.int32)
    return np.asarray((Xraw[:, idx] >= 1).sum(axis=1)).ravel().astype(np.int32)

# --- Lineage-defining signals ---
cd3  = sig(["CD3D","CD3E","CD3G"]); cd2 = sig(["CD2"]); cd8b = sig(["CD8B"])
ms4a1 = sig(["MS4A1"]); cd79a = sig(["CD79A"]); cd79b = sig(["CD79B"])
klrd1 = sig(["KLRD1"]); nkg7 = sig(["NKG7"]); gnly = sig(["GNLY"]); klrc1 = sig(["KLRC1"])

# IMPORTANT — rescue is restricted to ADAPTIVE LYMPHOID immune cells (T, B, NK).
# Their markers (CD3, MS4A1+CD79, KLRD1/KLRF1) are SPECIFIC and rare in brain,
# so multi-marker calls are reliable.  Myeloid markers (CD68/AIF1/C1Q/LYZ) and
# single structural markers (PECAM1/ACTA2/AQP4) have heavy ambient expression
# in this dense tissue (236k microglia leak myeloid transcripts everywhere;
# EBF1 is expressed in vascular cells) — multi-marker calls there still fire on
# real neurons/glia.  Mono/Mac and microglia are abundant and already cluster
# correctly, so they are intentionally NOT rescued.  This precisely targets the
# perivascular T/B/NK cells dragged into the vascular cluster.
hr_lineage = np.array(["none"] * n_obs, dtype=object)
hr_conf    = np.array(["none"] * n_obs, dtype=object)

def assign(mask, lin):
    take = mask & (hr_lineage == "none")
    hr_lineage[take] = lin; hr_conf[take] = "high"

# T cell: >=2 CD3 transcripts, OR CD3 + CD2 (guards against 1 ambient CD3)
assign((cd3 >= 2) | ((cd3 >= 1) & (cd2 >= 1)), "Tcell")
# B cell: CD79A-anchored (CD79A is the B-specific BCR chain) + a second B marker.
# KLRF1/EBF1/MS4A1-alone paths are dropped — KLRF1 loads on the macrophage
# factor, EBF1 on vascular cells, MS4A1 cross-reacts in myeloid.
assign((cd79a >= 1) & ((ms4a1 >= 1) | (cd79b >= 1)), "Bcell")
# NK: KLRD1 (CD94, pan-NK) AND a cytotoxic marker, no CD3, no CD8B.
# KLRF1 is EXCLUDED (non-specific here: positive in 87k cells, loads on the
# CD163/MRC1 macrophage factor F3).
assign((klrd1 >= 1) & ((nkg7 >= 1) | (gnly >= 1) | (klrc1 >= 1))
       & (cd3 < 1) & (cd8b < 1), "NK")

print("\n=== Robust hard-rule lineage (adaptive lymphoid, rescue-eligible) ===")
print(pd.Series(hr_lineage).value_counts())

# ======================================================================
# 3.  Flag cross-lineage contamination
#     A cell is contaminated when it has a CONFIDENT (high) hard-rule lineage
#     whose compartment differs from its dominant scHPF factor's compartment.
# ======================================================================
dom_idx  = np.argmax(Theta, axis=1)
dom_fac  = np.array(fcols)[dom_idx]
dom_comp = factor_comp[dom_idx]
hr_comp  = np.array([COMPARTMENT.get(l, "none") if l != "none" else "none"
                     for l in hr_lineage])

is_high = (hr_conf == "high")
is_contam = is_high & (hr_comp != "none") & (dom_comp != "neutral") \
            & (hr_comp != dom_comp)
print(f"\n[{time.strftime('%H:%M:%S')}] Contaminated (confident hard-rule, "
      f"off-compartment dominant program): {int(is_contam.sum()):,}")

# Headline subset: immune cells dragged into the vascular compartment
immune_lins = {"Tcell","NK","Bcell","MonoMac"}
head = is_contam & np.isin(hr_lineage, list(immune_lins)) & (dom_comp == "vascular")
print(f"  of which immune-into-vascular (the perivascular case): "
      f"{int(head.sum()):,}")
v2 = obs["cell_type_v2"].astype(str).values
print("  prior cell_type_v2 of all flagged cells:")
print(pd.Series(v2[is_contam]).value_counts().head(12))

flags = pd.DataFrame({
    "cell_id": obs_idx,
    "cluster_v2": v2,
    "hardrule_lineage": hr_lineage,
    "hardrule_confidence": hr_conf,
    "dominant_factor": dom_fac,
    "factor_lineage": np.array([factor_lineage[f] for f in dom_fac]),
    "dominant_compartment": dom_comp,
    "is_contaminated": is_contam,
})
flags.to_csv(CMAP / "allcell_contamination_flags.csv", index=False)
print(f"Saved: {CMAP / 'allcell_contamination_flags.csv'}")

# ======================================================================
# 4.  Decontaminate flagged cells via scHPF factor attribution
#     X_clean_ij = X_ij * (sum_keep mu_ijk) / (sum_all mu_ijk)
#     keep = factors in the cell's own compartment + neutral factors
# ======================================================================
print(f"\n[{time.strftime('%H:%M:%S')}] Decontaminating "
      f"{int(is_contam.sum()):,} flagged cells via factor attribution ...")
# Cleaning multiplies existing counts by a fraction in [0,1] -> it can only
# SHRINK or zero existing nonzeros, never create new ones.  So we edit the CSR
# .data array of the flagged rows in place (fast, no LIL/dense blow-up).
Xdec = Xraw.copy()
indptr, indices = Xdec.indptr, Xdec.indices
flag_pos = np.where(is_contam)[0]

# group flagged cells by their hard-rule compartment so the keep-mask is
# constant within a group (vectorised attribution per group)
hr_comp_flagged = hr_comp[flag_pos]
for comp in np.unique(hr_comp_flagged):
    grp = flag_pos[hr_comp_flagged == comp]
    if grp.size == 0:
        continue
    keep_cols = np.where((factor_comp == comp) | (factor_comp == "neutral"))[0]
    th = Theta[grp]                                   # g x K
    full = th @ Beta_full.T                           # g x genes  (expected)
    keep = th[:, keep_cols] @ Beta_full[:, keep_cols].T
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(full > 0, keep / full, 1.0)   # share to retain (g x genes)
    frac = np.clip(frac, 0.0, 1.0)
    for r, ci in enumerate(grp):
        a, b = indptr[ci], indptr[ci + 1]
        cols = indices[a:b]                           # genes this cell expresses
        Xdec.data[a:b] = np.rint(Xdec.data[a:b] * frac[r, cols])
    print(f"  compartment={comp:<9} cells={grp.size:>7,} "
          f"keep_factors={list(np.array(fcols)[keep_cols])}")

Xdec.eliminate_zeros()

# quick before/after on the headline vascular contamination genes
for gname in ["PECAM1","ACTA2","PDGFRB","CD3D","CD3E"]:
    if gname in var_pos:
        j = var_pos[gname]
        before = np.asarray(Xraw[flag_pos, j].todense()).ravel().mean() if flag_pos.size else 0
        after  = np.asarray(Xdec[flag_pos, j].todense()).ravel().mean() if flag_pos.size else 0
        print(f"  flagged-cell mean {gname:<7} before={before:6.3f}  after={after:6.3f}")

# ======================================================================
# 5.  Corrected labels (12-category) — base from existing labels, then
#     override flagged cells with their hard-rule identity.
# ======================================================================
V2_TO_CAT = {
    "Exc":"Excitatory neurons","Inh":"Inhibitory neurons","Ast":"Astrocytes",
    "Mic":"Microglia","Oli":"Oligodendrocytes","OPC":"OPC","End":"Endothelial",
    "Per":"Pericytes","SMC":"SMC","Mono/Mac":"Mono/Mac","B":"Immune",
    "T/NK":"Immune","Amb":"Ambiguous",
}
HR_TO_CAT = {
    "Tcell":"Immune","NK":"Immune","Bcell":"Immune","MonoMac":"Mono/Mac",
    "Mic":"Microglia","Oli":"Oligodendrocytes","OPC":"OPC","Ast":"Astrocytes",
    "Exc":"Excitatory neurons","Inh":"Inhibitory neurons","Endo":"Endothelial",
    "Per":"Pericytes","SMC":"SMC",
}
base_cat = obs["cell_type_v2"].astype(str).map(V2_TO_CAT).fillna("Ambiguous").values
cat_corr = base_cat.copy()
cat_corr[is_contam] = np.array([HR_TO_CAT.get(l, "Ambiguous")
                                for l in hr_lineage[is_contam]])

# fine label: keep existing cell_type_final_v2 except overwrite flagged cells
fine_corr = obs["cell_type_final_v2"].astype(str).values.copy()
fine_corr[is_contam] = hr_lineage[is_contam]

print("\n=== category_corrected distribution ===")
print(pd.Series(cat_corr).value_counts())
moved = base_cat[is_contam]
print("\nRescued cells moved FROM (base category) -> Immune/own lineage:")
print(pd.Series(moved).value_counts())

# ======================================================================
# 6.  Write decontaminated h5ad
# ======================================================================
obs_out = obs.copy()
obs_out["hardrule_lineage"]   = hr_lineage
obs_out["hardrule_confidence"] = hr_conf
obs_out["dominant_factor"]    = dom_fac
obs_out["factor_lineage"]     = [factor_lineage[f] for f in dom_fac]
obs_out["is_contaminated"]    = is_contam
obs_out["category_corrected"] = cat_corr
obs_out["cell_type_corrected"] = fine_corr
for fc in fcols:
    obs_out[fc] = cell_df[fc].values

A = ad.AnnData(X=Xdec, obs=obs_out, var=pd.DataFrame(index=var_idx))
A.layers["counts"]         = Xraw
A.layers["counts_decontam"] = Xdec
A.obsm["X_pca_harmony"] = Xph
if Xum is not None:
    A.obsm["X_umap_old"] = Xum
out = CMAP / "merged_qc_decontaminated.h5ad"
A.write(out, compression="gzip")
print(f"\n[{time.strftime('%H:%M:%S')}] Saved: {out}")
print(f"  total flagged/rescued: {int(is_contam.sum()):,}")
