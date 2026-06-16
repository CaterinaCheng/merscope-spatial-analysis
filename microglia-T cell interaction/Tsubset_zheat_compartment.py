"""
365_Tsubset_zheat_compartment.py
(A) Microglia state near each T subset (incl CD4 subsets) — Z-SCORED Green signature scores
    (full microglia set for power); * near-vs-baseline BH-FDR<0.05 (n>=10).
(B) vascular compartment ratio within each microglia state (clean labels).
(C) overall compartment ratio across all (clean) microglia.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new"); QC=Path(r"<MERSCOPE_ROOT>\QC data")
DEC=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
comps=["perivascular","vessel-adjacent","parenchymal"]
CCOL={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
g6=pd.read_csv(NEW/"green_mic_state_signatures.csv")
def colg(key): return [x for x in g6[[c for c in g6.columns if key in c][0]].dropna()]
GSIG={"Homeostatic":colg("Mic.2"),"MHC-II/APC":colg("Mic.9"),"DAM":sorted(set(colg("Mic.12"))|set(colg("Mic.13"))),"Phagocytic":colg("Mic.7"),"Inflammatory/IEG":colg("Mic.15")}
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_tnk=(v2=="T/NK"); micidx=np.where(v2=="Mic")[0]; mid=idx[micidx]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=mid),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,gl in GSIG.items(): sc.tl.score_genes(A,[x for x in gl if x in A.var_names],score_name=k,ctrl_size=50)
for st in STATEORD:
    v=A.obs[st].values.astype(float); A.obs[st]=(v-v.mean())/(v.std()+1e-9)   # z across all microglia
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); mi=micidx
near={s:np.zeros(len(mi),bool) for s in SUBS}; anyT=np.zeros(len(mi),bool)
for r in np.unique(run[mi]):
    sel=np.where(run[mi]==r)[0]; mxy=np.column_stack([mx[mi[sel]],my[mi[sel]]]); ok=np.isfinite(mxy[:,0])
    if ok.sum()==0: continue
    for s in SUBS:
        ss=np.where((labv==s)&(run==r)&hasxy)[0]
        if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy[ok],k=1); near[s][sel[ok]]=d<=30
    alls=np.where(is_tnk&(run==r)&hasxy)[0]
    if len(alls): d3,_=cKDTree(np.column_stack([mx[alls],my[alls]])).query(mxy[ok],k=1); anyT[sel[ok]]=d3<=30
base=~anyT; Zc={st:A.obs[st].values for st in STATEORD}
L=np.full((len(SUBS),len(STATEORD)),np.nan); star=np.zeros_like(L,bool); Ncnt=[]; pcollect=[]; keys=[]
for i,s in enumerate(SUBS):
    m=near[s]; nn=int(m.sum()); Ncnt.append(nn)
    for j,st in enumerate(STATEORD):
        L[i,j]=Zc[st][m].mean() if nn>0 else np.nan
        if nn>=10: pcollect.append(mannwhitneyu(Zc[st][m],Zc[st][base]).pvalue); keys.append((i,j))
pv=np.array(pcollect); o=np.argsort(pv); rk=np.empty(len(pv),int); rk[o]=np.arange(1,len(pv)+1); padj=np.minimum(pv*len(pv)/rk,1)
for (i,j),pa in zip(keys,padj): star[i,j]=pa<0.05
pd.DataFrame(L,index=SUBS,columns=STATEORD).assign(n=Ncnt).to_csv(NEW/"Tsubset_state_zscore.csv")
print("n near each subset:",{SUBS[i]:Ncnt[i] for i in range(len(SUBS))})
# compartment ratios (clean labels)
co=pd.read_csv(NEW/"microglia_final_coords.csv",index_col=0); co=co[~co.cluster_flag]
spc=pd.read_csv(NEW/"clean_microglia_spatial.csv",index_col=0)[["comp"]]; co=co.join(spc,how="inner"); co=co[co.comp!="n/a"]
RAT=pd.DataFrame({st:[100*((co.state==st)&(co.comp==c)).sum()/max((co.state==st).sum(),1) for c in comps] for st in STATEORD},index=comps).T
overall=[100*(co.comp==c).mean() for c in comps]
# ================= FIGURE =================
fig=plt.figure(figsize=(19,6)); gs=fig.add_gridspec(1,3,width_ratios=[1.25,1.15,0.32],wspace=0.3)
# A z-score heatmap
axA=fig.add_subplot(gs[0,0]); vm=np.nanmax(np.abs(L))
im=axA.imshow(L,cmap="RdBu_r",norm=TwoSlopeNorm(0,-vm,vm),aspect="auto")
axA.set_xticks(range(len(STATEORD))); axA.set_xticklabels(STATEORD,rotation=25,ha="right")
axA.set_yticks(range(len(SUBS))); axA.set_yticklabels([f"{s} (n={Ncnt[i]})" for i,s in enumerate(SUBS)])
for i in range(len(SUBS)):
    for j in range(len(STATEORD)):
        if np.isnan(L[i,j]): axA.text(j,i,"n/a",ha="center",va="center",fontsize=7,color="#999"); continue
        axA.text(j,i,f"{L[i,j]:+.2f}"+("*" if star[i,j] else ""),ha="center",va="center",fontsize=8,color="white" if abs(L[i,j])>vm*0.6 else "#222")
axA.set_title("Microglia state near each T subset (Z-SCORED, full microglia set)\n* near vs baseline BH-FDR<0.05",fontsize=10,fontweight="bold")
fig.colorbar(im,ax=axA,shrink=0.7,label="mean z-score")
# B compartment ratio per state
axB=fig.add_subplot(gs[0,1]); x=np.arange(len(STATEORD)); bottom=np.zeros(len(STATEORD))
for c in comps:
    vals=RAT[c].values; axB.bar(x,vals,bottom=bottom,color=CCOL[c],label=c)
    for xi,(vv,b) in enumerate(zip(vals,bottom)):
        if vv>=4: axB.text(xi,b+vv/2,f"{vv:.0f}%",ha="center",va="center",fontsize=8,color="white",fontweight="bold")
    bottom+=vals
axB.set_xticks(x); axB.set_xticklabels([f"{s}\n(n={int((co.state==s).sum())})" for s in STATEORD],fontsize=8); axB.set_ylim(0,100); axB.set_ylabel("% of state's microglia")
axB.set_title("Compartment ratio within each microglia state",fontsize=10,fontweight="bold"); axB.legend(fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=3)
for sp in axB.spines.values(): sp.set_visible(False)
# C overall compartment ratio
axC=fig.add_subplot(gs[0,2]); bottom=0
for c,v in zip(comps,overall):
    axC.bar(0,v,bottom=bottom,color=CCOL[c]); axC.text(0,bottom+v/2,f"{c.split('-')[0]}\n{v:.0f}%",ha="center",va="center",fontsize=8,color="white",fontweight="bold"); bottom+=v
axC.set_xticks([0]); axC.set_xticklabels([f"all microglia\n(n={len(co)})"],fontsize=8); axC.set_ylim(0,100); axC.set_yticks([])
axC.set_title("Overall",fontsize=10,fontweight="bold")
for sp in axC.spines.values(): sp.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"Tsubset_zheat_compartment.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: Tsubset_zheat_compartment.png + Tsubset_state_zscore.csv")
print("overall compartment %:",{c:round(v,1) for c,v in zip(comps,overall)})
