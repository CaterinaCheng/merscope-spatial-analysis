"""
316_LR_heatmap_v2.py  (revised per critique)
 - MHC-II ligand = aggregate of all MHC-II genes (any HLA-DRA/DPA1/DQB1/DRB1 detected).
 - MHC-II->CD4 restricted to CD4 subsets only (CD4 co-receptor not used by CD8/NK -> N/A).
 - each cell annotated with receptor+ n so underpowering is explicit.
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
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
CD4SUBS={"CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"}
MHCII=["HLA-DRA","HLA-DPA1","HLA-DQB1","HLA-DRB1"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}; vs=set(var)
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel() if gn in vp else np.zeros(X.shape[0])
labv=lab.reindex(idx).values; is_Mic=(v2=="Mic")
# ligand-positive vectors
mhc_pos=np.zeros(X.shape[0]);
for gnm in MHCII: mhc_pos=mhc_pos+E(gnm)
LIG={"CXCL16":E("CXCL16")>0,"CCL2":E("CCL2")>0,"CD86":E("CD86")>0,"MHC-II":mhc_pos>0}
# (ligand-key, receptor, label, allowed-subsets)
PAIRS=[("CXCL16","CXCR6","CXCL16→CXCR6 (retention)",set(SUBS)),
       ("CCL2","CCR2","CCL2→CCR2 (chemotaxis)",set(SUBS)),
       ("CD86","CD28","CD86→CD28 (costim)",set(SUBS)),
       ("CD86","CTLA4","CD86→CTLA4 (inhibitory)",set(SUBS)),
       ("MHC-II","CD4","MHC-II→CD4 (Ag-pres, CD4 only)",CD4SUBS)]
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
def coloc(lig_pos,recv_mask):
    obs=0; null=np.zeros(NPERM); used=0
    for r in np.unique(run):
        s=np.where(micmask&(run==r))[0]; rv=np.where(recv_mask&(run==r)&hasxy)[0]
        if len(s)<5 or len(rv)<2: continue
        used+=len(rv); nbr=cKDTree(np.column_stack([mx[s],my[s]])).query_ball_point(np.column_stack([mx[rv],my[rv]]),r=R)
        lh=lig_pos[s].astype(int); obs+=sum(lh[h].sum() for h in nbr)
        for k in range(NPERM): null[k]+=sum(rng.permutation(lh)[h].sum() for h in nbr)
    mu=null.mean(); return (obs/(mu+1e-9) if mu>0 else np.nan),(np.sum(null>=obs)+1)/(NPERM+1)
FOLD=np.full((len(PAIRS),len(SUBS)),np.nan); P=np.full((len(PAIRS),len(SUBS)),np.nan); N=np.zeros((len(PAIRS),len(SUBS)),int); NA=np.zeros((len(PAIRS),len(SUBS)),bool)
for i,(lk,rc,desc,allowed) in enumerate(PAIRS):
    rec=E(rc)
    for j,s in enumerate(SUBS):
        if s not in allowed: NA[i,j]=True; continue
        rmask=(labv==s)&(rec>0); N[i,j]=int((rmask&hasxy).sum())
        if N[i,j]>=MINREC: FOLD[i,j],P[i,j]=coloc(LIG[lk],rmask)
    print(desc,"done")
fig,ax=plt.subplots(figsize=(11,4.8))
norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(2.5,np.nanmax(FOLD)))
ax.imshow(np.ma.masked_invalid(FOLD),cmap="RdBu_r",norm=norm,aspect="auto"); ax.set_facecolor("#d9d9d9")
import matplotlib.cm as cm
sm=cm.ScalarMappable(cmap="RdBu_r",norm=norm)
for i in range(len(PAIRS)):
    for j in range(len(SUBS)):
        if NA[i,j]: ax.text(j,i,"N/A",ha="center",va="center",fontsize=7,color="#999"); continue
        if np.isnan(FOLD[i,j]): ax.text(j,i,f"n={N[i,j]}",ha="center",va="center",fontsize=7,color="#666"); continue
        star="*" if P[i,j]<0.05 else ""
        ax.text(j,i,f"{FOLD[i,j]:.2f}{star}\n(n={N[i,j]})",ha="center",va="center",fontsize=7.5,
                color="white" if (FOLD[i,j]>1.8 or FOLD[i,j]<0.4) else "black",fontweight="bold" if star else "normal")
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels(SUBS,fontsize=8,rotation=30,ha="right")
ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([p[2] for p in PAIRS],fontsize=8.5)
fig.colorbar(sm,ax=ax,shrink=0.7,label="fold-enrichment vs null")
ax.set_title("Microglia → T/NK L-R spatial interaction (n=receptor+ cells; * p<0.05; grey=too few; N/A=co-receptor not used)",fontsize=9.5,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"Tmic_LR_heatmap.png",dpi=140,bbox_inches="tight"); plt.close()
pd.DataFrame(FOLD,index=[p[2] for p in PAIRS],columns=SUBS).to_csv(NEW/"Tmic_LR_heatmap_fold.csv")
print("Saved: Tmic_LR_heatmap.png + Tmic_LR_heatmap_fold.csv")
