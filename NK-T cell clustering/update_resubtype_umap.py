"""
301_update_resubtype_umap.py
Recolor the CD8/CD4 resubtype UMAPs with ROBUST annotations only — deleting the
spillover/panel artifacts (Mono-mac, DC, MAIT, CD16+ NK miscalls) that CellTypist produced.
Robust label = marker-validated existing phenotype; for unphenotyped cells: CD8 -> the robust
CellTypist cytotoxic call (Tem/Trm), CD4 -> generic 'CD4 T (unresolved)' (panel can't subtype).
UMAP recomputed from saved intrinsic (spillover-removed) scHPF factor scores.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
INTRINSIC={"CD8":["F0","F1","F2","F3","F4","F5","F6"],"CD4":["F0","F2","F3","F5","F6","F7"]}  # spillover factors dropped (from 300)
REALCD8={"CD8 TRM","CD8 TEMRA"}; REALCD4={"CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"}

def robust_label(lin,row):
    ph=row.existing_phenotype; ct=str(row.celltypist)
    if lin=="CD8":
        if ph in REALCD8: return ph
        # unphenotyped CD8: trust only the robust cytotoxic calls; artifacts -> generic CD8
        if "Temra" in ct: return "CD8 TEMRA-like"
        return "CD8 cytotoxic (Tem/Trm)"
    else:
        if ph in REALCD4: return ph
        return "CD4 T (unresolved)"

pal={"CD8 TRM":"#C44E52","CD8 TEMRA":"#E67E22","CD8 TEMRA-like":"#F0B27A","CD8 cytotoxic (Tem/Trm)":"#7FB3D5",
     "CD4 Th":"#4C72B0","CD4 CTL":"#8172B3","CD4 Tcm/mem":"#55A868","CD4 Treg":"#000000","CD4 T (unresolved)":"#BDC3C7"}
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,lin in zip(axes,["CD8","CD4"]):
    sc_=pd.read_csv(NEW/f"Tcell_{lin}_schpf_scores.csv",index_col=0)
    sub=pd.read_csv(NEW/f"Tcell_{lin}_subtype.csv").set_index("cell_id")
    d=sc_.join(sub,how="inner")
    A=ad.AnnData(X=np.zeros((len(d),1)),obs=d.copy())
    A.obsm["X_f"]=d[INTRINSIC[lin]].values
    sc.pp.neighbors(A,use_rep="X_f",n_neighbors=15); sc.tl.umap(A,min_dist=0.4,random_state=0)
    U=A.obsm["X_umap"]
    d["robust"]=[robust_label(lin,r) for _,r in d.iterrows()]
    for lab in [l for l in pal if l in set(d.robust)]:
        m=(d.robust==lab).values
        ax.scatter(U[m,0],U[m,1],s=18,c=pal[lab],label=f"{lab} ({m.sum()})",linewidths=0,alpha=0.85)
    ax.set_title(f"{lin} T cells — robust annotation\n(spillover/panel artifacts removed)",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8,markerscale=2,loc="best"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    for s in ("top","right"): ax.spines[s].set_visible(False)
    d[["robust"]].to_csv(NEW/f"Tcell_{lin}_robust_label.csv")
    print(f"{lin} robust labels:",d.robust.value_counts().to_dict())
fig.suptitle("CD8 / CD4 T cells — neighborhood-scHPF (spillover-removed) + robust annotation",fontsize=11,fontweight="bold",y=1.02)
plt.tight_layout(); fig.savefig(NEW/"Tcell_resubtype_umap.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: Tcell_resubtype_umap.png (overwritten, robust labels) + Tcell_{CD8,CD4}_robust_label.csv")
