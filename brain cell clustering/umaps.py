"""
296_umaps.py  — clean current UMAPs
A. All brain cells: master X_umap colored by validated cell_type_v2.
B. T/NK cells: UMAP on abl5197 projected factor scores, colored by (i) existing phenotype,
   (ii) transferred reference celltype.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
import umap
plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); LAB=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")

# ---- A: all-cell UMAP ----
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    um=f["obsm"]["X_umap"][:]
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
pal={"Exc":"#4C72B0","Inh":"#55A868","Ast":"#8172B3","Oli":"#CCB974","OPC":"#64B5CD","Mic":"#C44E52",
     "Mono/Mac":"#E67E22","T/NK":"#000000","B":"#E377C2","End":"#7F7F7F","Per":"#BCBD22","SMC":"#17BECF","Amb":"#DDDDDD"}
fig,ax=plt.subplots(figsize=(8.5,7))
for ct in [c for c in pal if c in set(v2)]:
    m=v2==ct; s=14 if ct in ("T/NK","Mic","Mono/Mac","B") else 2
    ax.scatter(um[m,0],um[m,1],s=s,c=pal[ct],label=f"{ct} ({m.sum()})",linewidths=0,rasterized=True,alpha=0.6 if s==2 else 0.9)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title(f"All brain cells (n={len(idx)}) — validated cell_type_v2",fontsize=11,fontweight="bold")
ax.legend(markerscale=3,fontsize=7,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"umap_allcells_celltype.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: umap_allcells_celltype.png")

# ---- B: T/NK UMAP on projected reference factors ----
fac=pd.read_csv(NEW/"merscope_Tcell_projected_factors.csv",index_col=0)
F=[c for c in fac.columns if c.startswith("F")]
tr=pd.read_csv(NEW/"merscope_Tcell_transferred_celltype.csv").set_index("cell_id")
ph=pd.concat([pd.read_csv(LAB/"schpf_CD8_final_labels.csv")[["cell_id","phenotype"]],
              pd.read_csv(LAB/"schpf_CD4_final_labels.csv")[["cell_id","phenotype"]]]).set_index("cell_id")["phenotype"]
emb=umap.UMAP(n_neighbors=20,min_dist=0.3,random_state=0).fit_transform(fac[F].values)
ud=pd.DataFrame(emb,index=fac.index,columns=["u1","u2"])
ud["phenotype"]=ud.index.map(ph).fillna("(unphenotyped T/NK)")
ud["ref"]=ud.index.map(tr["ref_celltype"])
fig,axes=plt.subplots(1,2,figsize=(15,6))
phpal={"CD8 TRM":"#C44E52","CD8 TEMRA":"#E67E22","CD4 Th":"#4C72B0","CD4 CTL":"#8172B3","CD4 Tcm/mem":"#55A868","CD4 Treg":"#000000","(unphenotyped T/NK)":"#DDDDDD"}
for ph_,c in phpal.items():
    m=ud.phenotype==ph_
    if m.sum(): axes[0].scatter(ud.u1[m],ud.u2[m],s=14,c=c,label=f"{ph_} ({m.sum()})",linewidths=0,alpha=0.8)
axes[0].set_title("T/NK UMAP (abl5197 factor space) — existing phenotype",fontsize=10,fontweight="bold"); axes[0].legend(markerscale=2,fontsize=7.5)
refs=ud.ref.value_counts().head(10).index
cmap=plt.cm.tab20(np.linspace(0,1,len(refs)))
for r,c in zip(refs,cmap):
    m=ud.ref==r; axes[1].scatter(ud.u1[m],ud.u2[m],s=14,color=c,label=f"{r} ({m.sum()})",linewidths=0,alpha=0.8)
axes[1].set_title("same UMAP — transferred abl5197 ref-celltype",fontsize=10,fontweight="bold"); axes[1].legend(markerscale=2,fontsize=7)
for ax in axes: ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top","right")]
plt.tight_layout(); fig.savefig(NEW/"umap_TNK_reference_factors.png",dpi=140,bbox_inches="tight"); plt.close()
ud.to_csv(NEW/"umap_TNK_coords.csv")
print("Saved: umap_TNK_reference_factors.png + umap_TNK_coords.csv")
