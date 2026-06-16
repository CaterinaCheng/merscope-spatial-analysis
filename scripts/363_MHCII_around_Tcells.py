"""
363_MHCII_around_Tcells.py
Is MHC-II/APC really ~0 around T cells? Test with RAW scores/genes (not discrete state),
on the FULL microglia set for power (MHC-II = microglia-specific, not cross-lineage ambient).
For each T group (each CD8 subset, pooled CD8, pooled CD4, NK, any T/NK) vs baseline (no T<=30um):
 - raw 5-state Green scores  (shows APC is NON-zero everywhere)
 - MHC-II machinery genes HLA-DRA/CD74/HLA-DRB1/HLA-DPA1/HLA-DQA1/CIITA mean expression
 Mann-Whitney near-vs-baseline, BH FDR.
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
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
MHCII=["HLA-DRA","CD74","HLA-DRB1","HLA-DPA1","HLA-DQA1","CIITA","HLA-DRB5","HLA-DPB1"]
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
GROUPS={"CD8 TRM 1":["CD8 TRM 1"],"CD8 TRM 2":["CD8 TRM 2"],"CD8 TEMRA":["CD8 TEMRA"],
        "any CD8":["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA"],"any CD4":["CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"],
        "NK":["NK"],"any T/NK":SUBS}
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
# z-score each state signature across all microglia (left panel display)
for st in STATEORD:
    v=A.obs[st].values.astype(float); A.obs[st]=(v-v.mean())/(v.std()+1e-9)
MHCII=[g for g in MHCII if g in A.var_names]
Eg=np.asarray(A[:,MHCII].X.todense())
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
hasxy=np.isfinite(mx); mpos={c:i for i,c in enumerate(idx)}; mi=micidx
nearsub={s:np.zeros(len(mi),bool) for s in SUBS}; anyT=np.zeros(len(mi),bool)
for r in np.unique(run[mi]):
    sel=np.where(run[mi]==r)[0]; mxy=np.column_stack([mx[mi[sel]],my[mi[sel]]]); ok=np.isfinite(mxy[:,0])
    if ok.sum()==0: continue
    for s in SUBS:
        ss=np.where((labv==s)&(run==r)&hasxy)[0]
        if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy[ok],k=1); nearsub[s][sel[ok]]=d<=30
    alls=np.where(is_tnk&(run==r)&hasxy)[0]
    if len(alls): d3,_=cKDTree(np.column_stack([mx[alls],my[alls]])).query(mxy[ok],k=1); anyT[sel[ok]]=d3<=30
base=~anyT; nbase=int(base.sum()); print(f"full microglia: {len(mi)}; baseline (no T/NK<=30): {nbase}")
def grpmask(g):
    m=np.zeros(len(mi),bool)
    for s in GROUPS[g]: m|=nearsub[s]
    return m
GN=list(GROUPS.keys())
# build matrices: groups x (states + MHCII genes), raw mean near; plus baseline row
feat=STATEORD+MHCII
rows=[]; pvals=[]; keys=[]
basevals={}
for f in STATEORD: basevals[f]=A.obs[f].values[base]
for gi,f in enumerate(MHCII): basevals[f]=Eg[base,gi]
Mn=np.zeros((len(GN)+1,len(feat))); star=np.zeros((len(GN)+1,len(feat)),bool); ncol=[]
# baseline row 0
for j,f in enumerate(feat): Mn[0,j]=basevals[f].mean()
ncol.append(nbase)
for ri,g in enumerate(GN,start=1):
    m=grpmask(g); nn=int(m.sum()); ncol.append(nn)
    for j,f in enumerate(feat):
        near=A.obs[f].values[m] if f in STATEORD else Eg[m,MHCII.index(f)]
        Mn[ri,j]=near.mean() if nn>0 else np.nan
        if nn>=8:
            p=mannwhitneyu(near,basevals[f]).pvalue; pvals.append(p); keys.append((ri,j))
pv=np.array(pvals); o=np.argsort(pv); rk=np.empty(len(pv),int); rk[o]=np.arange(1,len(pv)+1); padj=np.minimum(pv*len(pv)/rk,1)
for (ri,j),pa in zip(keys,padj): star[ri,j]=pa<0.05
labels=["baseline"]+GN
out=pd.DataFrame(Mn,index=labels,columns=feat); out.insert(0,"n",ncol); out.to_csv(NEW/"MHCII_around_Tcells.csv")
print("\nn near each group:",{labels[i]:ncol[i] for i in range(len(labels))})
print("\nMHC-II/APC raw score by group:"); print(out[["n","MHC-II/APC"]].round(3).to_string())
print("\nHLA-DRA / CD74 mean expr by group:"); print(out[["n","HLA-DRA","CD74"]].round(3).to_string())
# ================= FIGURE =================
fig,(axA,axB)=plt.subplots(1,2,figsize=(16,6),gridspec_kw={"width_ratios":[0.85,1.15]})
# A: states raw score (blue-red diverging, centered 0)
Sm=Mn[:,:len(STATEORD)]; vm=np.nanmax(np.abs(Sm))
im=axA.imshow(Sm,cmap="RdBu_r",norm=TwoSlopeNorm(vcenter=0,vmin=-vm,vmax=vm),aspect="auto")
axA.set_xticks(range(len(STATEORD))); axA.set_xticklabels(STATEORD,rotation=25,ha="right"); axA.set_yticks(range(len(labels))); axA.set_yticklabels([f"{l} (n={ncol[i]})" for i,l in enumerate(labels)])
for i in range(len(labels)):
    for j in range(len(STATEORD)):
        if np.isnan(Sm[i,j]): continue
        axA.text(j,i,f"{Sm[i,j]:.2f}"+("*" if star[i,j] else ""),ha="center",va="center",fontsize=8,color="white" if abs(Sm[i,j])>vm*0.6 else "#222")
axA.set_title("Green state score near each T group (Z-SCORED across microglia)\n(* near vs baseline BH-FDR<0.05); baseline≈0, color = SD from mean",fontsize=9.5,fontweight="bold")
fig.colorbar(im,ax=axA,shrink=0.7,label="mean z-score")
# B: MHC-II genes
Gm=Mn[:,len(STATEORD):]
im2=axB.imshow(Gm,cmap="Reds",aspect="auto")
axB.set_xticks(range(len(MHCII))); axB.set_xticklabels(MHCII,rotation=35,ha="right"); axB.set_yticks(range(len(labels))); axB.set_yticklabels([f"{l} (n={ncol[i]})" for i,l in enumerate(labels)])
for i in range(len(labels)):
    for j in range(len(MHCII)):
        if np.isnan(Gm[i,j]): continue
        axB.text(j,i,f"{Gm[i,j]:.2f}"+("*" if star[i,len(STATEORD)+j] else ""),ha="center",va="center",fontsize=7.5,color="white" if Gm[i,j]>np.nanmax(Gm)*0.6 else "#333")
axB.set_title("MHC-II machinery (mean log-expr) near each T group\n(* near vs baseline BH-FDR<0.05)",fontsize=9.5,fontweight="bold")
fig.colorbar(im2,ax=axB,shrink=0.7,label="mean log-expr")
fig.suptitle("Microglial antigen presentation around T cells (full microglia set, raw scores)",fontsize=12,fontweight="bold")
fig.text(0.5,0.005,"NOTE: MHC-I (HLA-A/B/C, B2M, TAP1/2, NLRC5) is NOT on the 550-gene panel — only MHC-II is measurable. "
         "Whether the CD8-adjacent microglia present via MHC-I (the CD8-appropriate pathway, IFN-γ co-induced with MHC-II) cannot be assessed here.",
         ha="center",fontsize=8,style="italic",color="#b00")
plt.tight_layout(rect=[0,0.02,1,1]); fig.savefig(NEW/"MHCII_around_Tcells.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: MHCII_around_Tcells.png + MHCII_around_Tcells.csv")
