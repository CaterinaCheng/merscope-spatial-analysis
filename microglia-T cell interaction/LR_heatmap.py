"""
315_LR_heatmap.py
Heatmap of microglia->T ligand-receptor spatial interaction strength per T/NK subset.
Cell = (L-R pair) x (subset); color = fold-enrichment of ligand+ microglia <->receptor+ subset
contacts vs permutation null; * = p<0.05; grey = too few receptor+ cells.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
plt.rcParams.update({"font.size":9}); rng=np.random.default_rng(0)
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0; NPERM=400; MINREC=3
PAIRS=[("CXCL16","CXCR6","CXCL16→CXCR6 (retention)"),("CCL2","CCR2","CCL2→CCR2 (chemotaxis)"),
       ("CD86","CD28","CD86→CD28 (costim)"),("CD86","CTLA4","CD86→CTLA4 (inhibitory)"),
       ("HLA-DRA","CD4","HLA-DRA→CD4 (MHC-II)")]
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}; is_Mic=(v2=="Mic")
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel()
labv=lab.reindex(idx).values
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan); run=np.array(["?"]*len(idx),dtype=object)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); run[i]=pre; mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); micmask=is_Mic&hasxy
def coloc(lig_gene,recv_mask):
    lig=E(lig_gene); obs=0; null=np.zeros(NPERM)
    for r in np.unique(run):
        s=np.where(micmask&(run==r))[0]; rv=np.where(recv_mask&(run==r)&hasxy)[0]
        if len(s)<5 or len(rv)<MINREC: continue
        nbr=cKDTree(np.column_stack([mx[s],my[s]])).query_ball_point(np.column_stack([mx[rv],my[rv]]),r=R)
        lh=(lig[s]>0).astype(int); obs+=sum(lh[h].sum() for h in nbr)
        for k in range(NPERM): null[k]+=sum(rng.permutation(lh)[h].sum() for h in nbr)
    mu=null.mean(); return (obs/(mu+1e-9) if mu>0 else np.nan),(np.sum(null>=obs)+1)/(NPERM+1)
FOLD=np.full((len(PAIRS),len(SUBS)),np.nan); P=np.full((len(PAIRS),len(SUBS)),np.nan); N=np.zeros((len(PAIRS),len(SUBS)),int)
for i,(lg,rc,desc) in enumerate(PAIRS):
    rec=E(rc)
    for j,s in enumerate(SUBS):
        rmask=(labv==s)&(rec>0); N[i,j]=int((rmask&hasxy).sum())
        if N[i,j]>=MINREC: FOLD[i,j],P[i,j]=coloc(lg,rmask)
    print(desc,"done")
pd.DataFrame(FOLD,index=[p[2] for p in PAIRS],columns=SUBS).to_csv(NEW/"Tmic_LR_heatmap_fold.csv")
# heatmap
fig,ax=plt.subplots(figsize=(10,4.6))
M=np.ma.masked_invalid(FOLD)
norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=np.nanmax([2.5,np.nanmax(FOLD)]))
im=ax.imshow(M,cmap="RdBu_r",norm=norm,aspect="auto")
ax.set_facecolor("#d9d9d9")
for i in range(len(PAIRS)):
    for j in range(len(SUBS)):
        if np.isnan(FOLD[i,j]): ax.text(j,i,"n<3",ha="center",va="center",fontsize=7,color="#666"); continue
        star="*" if P[i,j]<0.05 else ""
        ax.text(j,i,f"{FOLD[i,j]:.2f}{star}",ha="center",va="center",fontsize=8,
                color="white" if (FOLD[i,j]>1.8 or FOLD[i,j]<0.4) else "black",fontweight="bold" if star else "normal")
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels([f"{s}\n(n={int(max(N[:,j]))})" for j,s in enumerate(SUBS)],fontsize=8,rotation=30,ha="right")
ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([p[2] for p in PAIRS],fontsize=8.5)
fig.colorbar(im,ax=ax,shrink=0.7,label="fold-enrichment vs null")
ax.set_title("Microglia → T/NK ligand-receptor spatial interaction (sender = all microglia; * p<0.05)",fontsize=10.5,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"Tmic_LR_heatmap.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: Tmic_LR_heatmap.png + Tmic_LR_heatmap_fold.csv")
