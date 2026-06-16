"""
317_LR_dotplot.py
L-R interaction DOTPLOT: dot SIZE = % of subset expressing the receptor; dot COLOR =
spatial fold-enrichment of ligand+ microglia <-> receptor+ subset contacts vs null;
black ring = p<0.05. MHC-II aggregated; MHC-II->CD4 = CD4 subsets only.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
plt.rcParams.update({"font.size":9}); rng=np.random.default_rng(0)
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0; NPERM=400; MINREC=3
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
CD4SUBS={"CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"}; MHCII=["HLA-DRA","HLA-DPA1","HLA-DQB1","HLA-DRB1"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel() if gn in vp else np.zeros(X.shape[0])
labv=lab.reindex(idx).values; is_Mic=(v2=="Mic")
mhc=np.zeros(X.shape[0])
for gnm in MHCII: mhc=mhc+E(gnm)
LIG={"CXCL16":E("CXCL16")>0,"CCL2":E("CCL2")>0,"CD86":E("CD86")>0,"MHC-II":mhc>0}
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
    obs=0; null=np.zeros(NPERM)
    for r in np.unique(run):
        s=np.where(micmask&(run==r))[0]; rv=np.where(recv_mask&(run==r)&hasxy)[0]
        if len(s)<5 or len(rv)<2: continue
        nbr=cKDTree(np.column_stack([mx[s],my[s]])).query_ball_point(np.column_stack([mx[rv],my[rv]]),r=R)
        lh=lig_pos[s].astype(int); obs+=sum(lh[h].sum() for h in nbr)
        for k in range(NPERM): null[k]+=sum(rng.permutation(lh)[h].sum() for h in nbr)
    mu=null.mean(); return (obs/(mu+1e-9) if mu>0 else np.nan),(np.sum(null>=obs)+1)/(NPERM+1)
rec_pos={rc:(E(rc)>0) for _,rc,_,_ in PAIRS}
xs,ys,sizes,colors,edges=[],[],[],[],[]
recs=[]
for i,(lk,rc,desc,allowed) in enumerate(PAIRS):
    rp=rec_pos[rc]
    for j,s in enumerate(SUBS):
        if s not in allowed: continue
        sub=(labv==s); nsub=sub.sum(); pct=100*(rp[sub].mean()); nrec=int((sub&rp&hasxy).sum())
        fold,p=(np.nan,np.nan)
        if nrec>=MINREC: fold,p=coloc(LIG[lk],sub&rp)
        xs.append(j); ys.append(i); sizes.append(pct); colors.append(fold if not np.isnan(fold) else 1.0); edges.append(p)
        recs.append((desc,s,round(pct,0),nrec,round(fold,2) if not np.isnan(fold) else None,p))
    print(desc,"done")
pd.DataFrame(recs,columns=["pair","subset","pct_receptor","n_receptor","fold","p"]).to_csv(NEW/"Tmic_LR_dotplot.csv",index=False)
fig,ax=plt.subplots(figsize=(11,4.8))
norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(2.5,np.nanmax([c for c in colors])))
sc=ax.scatter(xs,ys,s=[max(p,1)*22 for p in sizes],c=colors,cmap="RdBu_r",norm=norm,edgecolors=["black" if (e is not None and e<0.05) else "#888" for e in edges],linewidths=[1.8 if (e is not None and e<0.05) else 0.5 for e in edges])
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels(SUBS,rotation=30,ha="right",fontsize=8)
ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([p[2] for p in PAIRS],fontsize=8.5); ax.invert_yaxis()
ax.set_xlim(-0.6,len(SUBS)-0.4); ax.set_ylim(len(PAIRS)-0.4,-0.6); ax.grid(True,color="#eee",lw=0.5)
fig.colorbar(cm.ScalarMappable(cmap="RdBu_r",norm=norm),ax=ax,shrink=0.7,label="spatial fold-enrichment vs null")
for pv in [5,25,50]: ax.scatter([],[],s=pv*22,c="#bbb",edgecolors="#888",label=f"{pv}%")
ax.legend(title="% expressing receptor",loc="center left",bbox_to_anchor=(1.18,0.5),fontsize=8,labelspacing=1.4,frameon=False)
ax.set_title("Microglia → T/NK L-R dotplot (size=% receptor+; color=spatial fold; black ring=p<0.05)",fontsize=9.5,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"Tmic_LR_dotplot.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: Tmic_LR_dotplot.png + Tmic_LR_dotplot.csv")
