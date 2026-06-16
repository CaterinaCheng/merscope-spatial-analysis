"""
321_nn_LR_and_QC.py
1. % perivascular T cells at 10/20/30/50um thresholds.
3. QC: microglia<->T distance distribution (how far is the nearest microglion/macrophage?) to
   choose the L-R interaction definition (nearest cell vs 20um vs 30um).
2. NEAREST-NEIGHBOUR L-R dotplot: for each receptor+ T cell, is its NEAREST microglion/macrophage
   ligand+? (vs permutation null). Microglia + Mono/Mac senders.
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
VESSEL=["End","Per","SMC"]; NPERM=500; MINREC=3
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
CD4SUBS={"CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"}; MHCII=["HLA-DRA","HLA-DPA1","HLA-DQB1","HLA-DRB1"]
PAIRS=[("CCL2","CCR2","CCL2->CCR2",set(SUBS)),("CXCL16","CXCR6","CXCL16->CXCR6",set(SUBS)),
       ("CD86","CD28","CD86->CD28",set(SUBS)),("CD86","CTLA4","CD86->CTLA4",set(SUBS)),
       ("MHC-II","CD4","MHC-II->CD4",CD4SUBS)]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    nn=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in nn["categories"][:]]; v2=np.array([cats[c] for c in nn["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel() if gn in vp else np.zeros(X.shape[0])
labv=lab.reindex(idx).values; is_ves=np.isin(v2,VESSEL); is_mic=(v2=="Mic"); is_mac=(v2=="Mono/Mac")
is_T=lab.reindex(idx).isin(SUBS).values
mhc=np.zeros(X.shape[0])
for gnm in MHCII: mhc=mhc+E(gnm)
LIGPOS={"CCL2":E("CCL2")>0,"CXCL16":E("CXCL16")>0,"CD86":E("CD86")>0,"MHC-II":mhc>0}
recpos={rc:(E(rc)>0) for _,rc,_,_ in PAIRS}
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
dVes=np.full(len(idx),np.inf); dMic=np.full(len(idx),np.inf); dMac=np.full(len(idx),np.inf)
for r in np.unique(run):
    ts=np.where(is_T&(run==r)&hasxy)[0]; txy=np.column_stack([mx[ts],my[ts]])
    for sel,arr in [(is_ves,dVes),(is_mic,dMic),(is_mac,dMac)]:
        ss=np.where(sel&(run==r)&hasxy)[0]
        if len(ss) and len(ts): dd,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(txy,k=1); arr[ts]=dd

# ===== PART 1 =====
print("=== PART 1: % perivascular T cells at each threshold ===")
tm=is_T&hasxy&np.isfinite(dVes)
for thr in [10,20,30,50]: print(f"  <= {thr}um: {100*(dVes[tm]<=thr).mean():.1f}% of T cells")

# ===== PART 3: microglia<->T distance QC =====
print("\n=== PART 3: T-cell distance to nearest myeloid (QC for interaction radius) ===")
for nm,arr in [("nearest MICROGLION",dMic),("nearest MACROPHAGE",dMac)]:
    a=arr[tm&np.isfinite(arr)]; print(f"  {nm}: median={np.median(a):.0f}um  %<=10={100*(a<=10).mean():.0f}  %<=15={100*(a<=15).mean():.0f}  %<=20={100*(a<=20).mean():.0f}  %<=30={100*(a<=30).mean():.0f}")
# reverse: microglion to nearest T
micm=is_mic&hasxy; dT=np.full(len(idx),np.inf)
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]; ts=np.where(is_T&(run==r)&hasxy)[0]
    if len(ms) and len(ts): dd,_=cKDTree(np.column_stack([mx[ts],my[ts]])).query(np.column_stack([mx[ms],my[ms]]),k=1); dT[ms]=dd
print(f"  (reverse) microglion->nearest T: median={np.median(dT[micm&np.isfinite(dT)]):.0f}um  %<=30={100*(dT[micm&np.isfinite(dT)]<=30).mean():.0f}")

# ===== PART 2: nearest-neighbour L-R dotplot =====
def coloc_nn(sender_sel,ligpos,recv_mask):
    obs=0; null=np.zeros(NPERM)
    for r in np.unique(run):
        ms=np.where(sender_sel&(run==r)&hasxy)[0]; rv=np.where(recv_mask&(run==r)&hasxy)[0]
        if len(ms)<5 or len(rv)<2: continue
        _,nidx=cKDTree(np.column_stack([mx[ms],my[ms]])).query(np.column_stack([mx[rv],my[rv]]),k=1)
        lig_sec=ligpos[ms].astype(int); obs+=lig_sec[nidx].sum()
        for k in range(NPERM): null[k]+=rng.permutation(lig_sec)[nidx].sum()
    mu=null.mean(); return (obs/(mu+1e-9) if mu>0 else np.nan),(np.sum(null>=obs)+1)/(NPERM+1)
SENDERS={"Mic":is_mic,"Mono/Mac":is_mac}
xs,ys,sz,col,edg=[],[],[],[],[]; ylabels=[]; rows=[]; yi=0
for snm,ssel in SENDERS.items():
    for lk,rc,desc,allowed in PAIRS:
        ylabels.append(f"{snm}: {desc}"); rp=recpos[rc]
        for j,s in enumerate(SUBS):
            if s not in allowed: continue
            sub=(labv==s); pct=100*rp[sub].mean(); nrec=int((sub&rp&hasxy).sum())
            fold,p=(np.nan,np.nan)
            if nrec>=MINREC: fold,p=coloc_nn(ssel,LIGPOS[lk],sub&rp)
            xs.append(j); ys.append(yi); sz.append(pct); col.append(fold if not np.isnan(fold) else 1.0); edg.append(p)
            rows.append(dict(sender=snm,pair=desc,subset=s,pct=round(pct,0),n=nrec,fold=round(fold,2) if not np.isnan(fold) else None,p=p))
        yi+=1
    print(snm,"done")
pd.DataFrame(rows).to_csv(NEW/"nn_LR.csv",index=False)
fig,ax=plt.subplots(figsize=(10,7))
norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=max(2.5,np.nanmax(col)))
ax.scatter(xs,ys,s=[max(p,1)*20 for p in sz],c=col,cmap="RdBu_r",norm=norm,
           edgecolors=["black" if (e is not None and e<0.05) else "#999" for e in edg],linewidths=[1.8 if (e is not None and e<0.05) else 0.4 for e in edg])
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels(SUBS,rotation=30,ha="right",fontsize=8)
ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels,fontsize=8); ax.invert_yaxis()
ax.set_xlim(-0.6,len(SUBS)-0.4); ax.axhline(4.5,color="#bbb",lw=1); ax.grid(True,color="#eee",lw=0.5)
fig.colorbar(cm.ScalarMappable(cmap="RdBu_r",norm=norm),ax=ax,shrink=0.5,label="fold vs null (nearest-cell)")
for pv in [5,25,50]: ax.scatter([],[],s=pv*20,c="#bbb",edgecolors="#999",label=f"{pv}%")
ax.legend(title="% receptor+",loc="center left",bbox_to_anchor=(1.12,0.5),fontsize=8,labelspacing=1.6,frameon=False)
ax.set_title("NEAREST-cell L-R: is the nearest microglion/macrophage ligand+? (size=% receptor+; ring=p<0.05)",fontsize=9.5,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"nn_LR_dotplot.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: nn_LR_dotplot.png + nn_LR.csv")
