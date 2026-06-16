"""
298_update_TNK_umap.py
Recolor the T/NK UMAP (abl5197 factor space) by HARD-RULE class (T=CD3+, NK=CD3-CD8B-,
ambiguous) instead of the over-claimed transferred fine-subtype labels.
Panel A: hard-rule T/NK/ambiguous.  Panel B: validated CD3+ phenotypes (NK/ambiguous greyed).
Uses saved coords (umap_TNK_coords.csv) + TNK_hardrule_classification.csv.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
co=pd.read_csv(NEW/"umap_TNK_coords.csv",index_col=0)
hr=pd.read_csv(NEW/"TNK_hardrule_classification.csv").set_index("cell_id")
d=co.join(hr,how="inner")
print("cells:",len(d),"| hard-rule:",d.hardrule.value_counts().to_dict())

# panel B label: validated phenotype, else NK/ambiguous/unphenotyped-T
realph={"CD8 TRM","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"}
def blab(r):
    if r.existing_phenotype in realph: return r.existing_phenotype
    if r.hardrule=="NK": return "NK (CD3- CD8B-)"
    if r.hardrule=="ambiguous": return "ambiguous"
    return "T (CD3+, unphenotyped)"
d["panelB"]=d.apply(blab,axis=1)

fig,axes=plt.subplots(1,2,figsize=(15,6))
# A: hard-rule
hp={"T":"#2166AC","NK":"#B2182B","ambiguous":"#BDBDBD"}
for k,c in hp.items():
    m=d.hardrule==k
    axes[0].scatter(d.u1[m],d.u2[m],s=14,c=c,label=f"{k} ({m.sum()})",linewidths=0,alpha=0.8)
axes[0].set_title("T/NK UMAP — HARD RULES\nT=CD3+ · NK=CD3- CD8B- NKmarker+ · ambiguous",fontsize=10,fontweight="bold")
axes[0].legend(markerscale=2,fontsize=9,loc="best")
# B: validated phenotype
bp={"CD8 TRM":"#C44E52","CD8 TEMRA":"#E67E22","CD4 Th":"#4C72B0","CD4 CTL":"#8172B3","CD4 Tcm/mem":"#55A868",
    "CD4 Treg":"#000000","T (CD3+, unphenotyped)":"#9ecae1","NK (CD3- CD8B-)":"#B2182B","ambiguous":"#E0E0E0"}
for k,c in bp.items():
    m=d.panelB==k
    if m.sum(): axes[1].scatter(d.u1[m],d.u2[m],s=14,c=c,label=f"{k} ({m.sum()})",linewidths=0,alpha=0.85)
axes[1].set_title("same UMAP — validated CD3+ phenotypes\n(NK / ambiguous shown separately; no over-claimed subtypes)",fontsize=10,fontweight="bold")
axes[1].legend(markerscale=2,fontsize=8,loc="best")
for ax in axes: ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); [ax.spines[s].set_visible(False) for s in ("top","right")]
fig.suptitle(f"T/NK compartment (n={len(d)}) — marker-gated, not label-transfer subtypes",fontsize=11,fontweight="bold",y=1.02)
plt.tight_layout(); fig.savefig(NEW/"umap_TNK_hardrule.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: umap_TNK_hardrule.png")
