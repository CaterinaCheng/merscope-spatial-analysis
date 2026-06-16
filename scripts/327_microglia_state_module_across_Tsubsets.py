"""
327_microglia_state_module_across_Tsubsets.py
Targeted (NOT de-novo) comparison of a FIXED microglial-state gene module across microglia
neighbouring each T subset vs baseline microglia (no T/NK <=30um). Avoids underpowered de-novo
DEG on tiny subsets: every column reports the same pre-chosen genes, with n labeled and *padj<0.05.
Module = antigen-presentation/IFN (up in activation) + homeostatic (down in activation) + DAM.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0
COLS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 Tcm/mem","NK"]
ALLT=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
MODULE=[("antigen presentation / IFN",["CD74","HLA-DRA","HLA-DPA1","HLA-DQB1","HLA-DRB1","CIITA"]),
        ("DAM / activation",["GPNMB","ITGAX","LGALS3","APOE","CD68","C1QB","TIMP2"]),
        ("homeostatic (down on activation)",["KLF4","SPIB","NCR3","ZNF331","BCL6","BASP1","IL1B","CX3CR1","TMEM119","P2RY12"])]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}; labv=lab.reindex(idx).values; is_mic=(v2=="Mic")
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx)
nearsub={s:np.zeros(len(idx)) for s in ALLT}; anyT=np.zeros(len(idx))
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]
    if not len(ms): continue
    mxy=np.column_stack([mx[ms],my[ms]])
    for s in ALLT:
        ss=np.where((labv==s)&(run==r)&hasxy)[0]
        if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy,k=1); nearsub[s][ms]=(d<=R).astype(int)
    for s in ALLT: anyT[ms]+=nearsub[s][ms]
base=is_mic&hasxy&(anyT==0)
genes=[g for _,gs in MODULE for g in gs if g in vp]
absent=[g for _,gs in MODULE for g in gs if g not in vp]
if absent: print("absent from panel (skipped):",absent)
def deg_module(nearm):
    sel=np.where(nearm|base)[0]; a=ad.AnnData(X=X[sel].copy(),var=pd.DataFrame(index=var)); a.obs["g"]=np.where(nearm[sel],"near","far")
    sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a); sc.tl.rank_genes_groups(a,"g",groups=["near"],reference="far",method="wilcoxon")
    r=sc.get.rank_genes_groups_df(a,group="near").set_index("names")
    return r.reindex(genes)["logfoldchanges"], r.reindex(genes)["pvals_adj"]
LFC=pd.DataFrame(index=genes); PAD=pd.DataFrame(index=genes); ncol={}
for s in COLS:
    near=is_mic&hasxy&(nearsub[s]>=1); ncol[s]=int(near.sum())
    LFC[s],PAD[s]=deg_module(near)
    print(f"microglia near {s}: n={ncol[s]}")
LFC.to_csv(NEW/"microglia_state_module_logFC.csv"); PAD.to_csv(NEW/"microglia_state_module_padj.csv")

fig,ax=plt.subplots(figsize=(7.5,0.34*len(genes)+1.8))
norm=TwoSlopeNorm(vmin=-1.2,vcenter=0,vmax=1.2)
im=ax.imshow(LFC.values,cmap="RdBu_r",norm=norm,aspect="auto")
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([f"{s}\n(n={ncol[s]})" for s in COLS],fontsize=8)
ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes,fontsize=8)
for i in range(len(genes)):
    for j in range(len(COLS)):
        if pd.notna(PAD.values[i,j]) and PAD.values[i,j]<0.05: ax.text(j,i,"*",ha="center",va="center",fontsize=11,fontweight="bold")
# group separators + labels
yb=0
for gname,gs in MODULE:
    gg=[g for g in gs if g in vp]
    if not gg: continue
    if yb>0: ax.axhline(yb-0.5,color="#333",lw=1.2)
    ax.text(len(COLS)-0.4,yb+len(gg)/2-0.5,gname,rotation=270,va="center",ha="left",fontsize=7.5,color="#444")
    yb+=len(gg)
fig.colorbar(im,ax=ax,shrink=0.45,label="log2FC vs baseline microglia")
ax.set_title("Microglial state module by neighbouring T subset\n(vs microglia with no T/NK <=30µm; * padj<0.05; small n = power-limited)",fontsize=9.5,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"microglia_state_module_across_Tsubsets.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: microglia_state_module_across_Tsubsets.png + logFC/padj csvs")
