"""
318_multiAPC_LR_dotplot.py
Multi-APC -> T/NK L-R dotplot. Senders = Microglia, Mono/Mac, B cells. For each
(sender, L-R pair, T subset): spatial colocalization (ligand+ sender <-> receptor+ subset, 30um)
vs permutation null. Dot size = % receptor+; color = fold; black ring = p<0.05.
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
R=30.0; NPERM=300; MINREC=3
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
CD4SUBS={"CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"}; MHCII=["HLA-DRA","HLA-DPA1","HLA-DQB1","HLA-DRB1"]
SENDERS={"Mic":"Mic","Mono/Mac":"Mono/Mac","B":"B"}
PAIRS=[("CCL2","CCR2","CCL2→CCR2",set(SUBS)),("CXCL16","CXCR6","CXCL16→CXCR6",set(SUBS)),
       ("CD86","CD28","CD86→CD28",set(SUBS)),("CD86","CTLA4","CD86→CTLA4",set(SUBS)),
       ("MHC-II","CD4","MHC-II→CD4",CD4SUBS)]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel() if gn in vp else np.zeros(X.shape[0])
labv=lab.reindex(idx).values
mhc=np.zeros(X.shape[0])
for gnm in MHCII: mhc=mhc+E(gnm)
LIGPOS={"CCL2":E("CCL2")>0,"CXCL16":E("CXCL16")>0,"CD86":E("CD86")>0,"MHC-II":mhc>0}
recpos={rc:(E(rc)>0) for _,rc,_,_ in PAIRS}
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan); run=np.array(["?"]*len(idx),dtype=object)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); run[i]=pre; mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); runs=np.unique(run)
# cache per-sender per-section trees
cache={}
for snm,sct in SENDERS.items():
    sm=(v2==sct)&hasxy; cache[snm]={}
    for r in runs:
        s=np.where(sm&(run==r))[0]
        if len(s)>=5: cache[snm][r]=(s,cKDTree(np.column_stack([mx[s],my[s]])))
def coloc(snm,ligpos,recv_mask):
    obs=0; null=np.zeros(NPERM)
    for r,(s,tree) in cache[snm].items():
        rv=np.where(recv_mask&(run==r)&hasxy)[0]
        if len(rv)<2: continue
        nbr=tree.query_ball_point(np.column_stack([mx[rv],my[rv]]),r=R)
        lh=ligpos[s].astype(int); obs+=sum(lh[h].sum() for h in nbr)
        for k in range(NPERM): null[k]+=sum(rng.permutation(lh)[h].sum() for h in nbr)
    mu=null.mean(); return (obs/(mu+1e-9) if mu>0 else np.nan),(np.sum(null>=obs)+1)/(NPERM+1)

rows=[]; xs,ys,sz,col,edg=[],[],[],[],[]; ylabels=[]
yi=0
for snm in SENDERS:
    for lk,rc,desc,allowed in PAIRS:
        ylabels.append(f"{snm}: {desc}")
        rp=recpos[rc]
        for j,s in enumerate(SUBS):
            if s not in allowed: continue
            sub=(labv==s); pct=100*rp[sub].mean(); nrec=int((sub&rp&hasxy).sum())
            fold,p=(np.nan,np.nan)
            if nrec>=MINREC: fold,p=coloc(snm,LIGPOS[lk],sub&rp)
            xs.append(j); ys.append(yi); sz.append(pct); col.append(fold if not np.isnan(fold) else 1.0); edg.append(p)
            rows.append(dict(sender=snm,pair=desc,subset=s,pct=round(pct,0),n=nrec,fold=round(fold,2) if not np.isnan(fold) else None,p=p))
        yi+=1
    print(snm,"done")
pd.DataFrame(rows).to_csv(NEW/"multiAPC_LR.csv",index=False)
fig,ax=plt.subplots(figsize=(11,9))
norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(2.5,np.nanmax([c for c in col])))
ax.scatter(xs,ys,s=[max(p,1)*20 for p in sz],c=col,cmap="RdBu_r",norm=norm,
           edgecolors=["black" if (e is not None and e<0.05) else "#999" for e in edg],
           linewidths=[1.8 if (e is not None and e<0.05) else 0.4 for e in edg])
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels(SUBS,rotation=30,ha="right",fontsize=8)
ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels,fontsize=8); ax.invert_yaxis()
ax.set_xlim(-0.6,len(SUBS)-0.4); ax.grid(True,color="#eee",lw=0.5)
for k in range(len(SENDERS)-1): ax.axhline(5*(k+1)-0.5,color="#bbb",lw=1)
fig.colorbar(cm.ScalarMappable(cmap="RdBu_r",norm=norm),ax=ax,shrink=0.5,label="spatial fold vs null")
for pv in [5,25,50]: ax.scatter([],[],s=pv*20,c="#bbb",edgecolors="#999",label=f"{pv}%")
ax.legend(title="% receptor+",loc="center left",bbox_to_anchor=(1.12,0.5),fontsize=8,labelspacing=1.6,frameon=False)
ax.set_title("APC (microglia / Mono-Mac / B) → T/NK L-R interaction (size=% receptor+; color=fold; ring=p<0.05)",fontsize=9.5,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"multiAPC_LR_dotplot.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: multiAPC_LR_dotplot.png + multiAPC_LR.csv")
