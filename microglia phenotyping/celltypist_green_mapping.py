"""
337_celltypist_green_mapping.py
Reference label-transfer: map our (decontaminated) microglia onto Green2024 microglial CLUSTERS
using the Green CellTypist model (Green2024_state_panel.pkl). Per-cell prediction restricted to
Mic.X classes (we know these are microglia). Colors the existing microglia UMAP by the mapped
Green cluster. Compares decontam vs the old raw mapping (ambient rescue).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import pickle, warnings; warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
GREEN=Path(r"D:\Caterina\MERSCOPE\merged_analysis\Green2024")
# Green Mic cluster -> identity (from our marker analysis of green_mic_state_mean_expr)
MICNAME={"Mic.2":"Homeostatic","Mic.9":"MHC-II/APC","Mic.12":"Activated-DAM","Mic.13":"Lipid-DAM",
         "Mic.15":"Inflammatory/IEG","Mic.7":"Phagocytic","Mic.10":"Mic.10","Mic.6":"Mic.6","Mic.1":"Mic.1",
         "Mic.3":"Mic.3","Mic.4":"Mic.4","Mic.5":"Mic.5","Mic.8":"Mic.8"}
obj=pickle.load(open(GREEN/"Green2024_state_panel.pkl","rb")); M=obj["Model"]; SCL=obj["Scaler_"]; feats=list(M.features)
classes=list(M.classes_); micix=[i for i,c in enumerate(classes) if c.startswith("Mic.")]; micnames=[classes[i] for i in micix]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]; print("microglia:",len(micidx))
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)   # CellTypist standard
vp={gn:i for i,gn in enumerate(var)}; Xq=np.zeros((A.n_obs,len(feats)),dtype=np.float32)
present=[gn for gn in feats if gn in vp]; print(f"model features present on panel: {len(present)}/{len(feats)}")
Xfull=A.X.toarray()
for j,gn in enumerate(feats):
    if gn in vp: Xq[:,j]=Xfull[:,vp[gn]]
Xs=SCL.transform(Xq); proba=M.predict_proba(Xs)
# unrestricted top class (ambient QC)
top_un=np.array(classes)[proba.argmax(1)]; frac_mic=100*np.char.startswith(top_un.astype(str),"Mic.").mean()
print(f"unrestricted top-class is a Mic.X for {frac_mic:.1f}% of decontam microglia (raw mapping was 55.6%)")
# restrict to Mic classes
pm=proba[:,micix]; mapped=np.array(micnames)[pm.argmax(1)]; conf=pm.max(1)/ (pm.sum(1)+1e-9)
A.obs["green_mic"]=mapped; A.obs["green_name"]=[MICNAME.get(m,m) for m in mapped]; A.obs["conf"]=conf
vc=pd.Series(mapped).value_counts()
print("\nmapped Green Mic cluster distribution:");
for k,v in vc.items(): print(f"  {k:7} {MICNAME.get(k,k):16}: {v:6d} ({100*v/len(mapped):.1f}%)  median conf={np.median(conf[mapped==k]):.2f}")
out=pd.DataFrame({"green_mic":mapped,"green_name":A.obs["green_name"].values,"conf":conf},index=A.obs_names)
out.to_csv(NEW/"microglia_green_mapped_labels.csv")

# UMAP coords from 336
co=pd.read_csv(NEW/"microglia_umap_coords.csv",index_col=0)
U=co.reindex(A.obs_names)[["umap1","umap2"]].values
order=[c for c in ["Mic.2","Mic.9","Mic.12","Mic.13","Mic.15","Mic.7","Mic.10","Mic.6","Mic.1","Mic.3","Mic.4","Mic.5","Mic.8"] if c in set(mapped)]
import matplotlib.cm as cm
pal=cm.get_cmap("tab20",len(order))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17,8))
for i,c in enumerate(order):
    m=mapped==c
    ax1.scatter(U[m,0],U[m,1],s=3,c=[pal(i)],label=f"{c} {MICNAME.get(c,'')} ({100*m.mean():.0f}%)",linewidths=0,rasterized=True)
ax1.set_title("Microglia UMAP — mapped to Green2024 clusters (CellTypist label transfer, decontam)",fontsize=10.5,fontweight="bold")
ax1.set_xticks([]); ax1.set_yticks([]); ax1.legend(markerscale=3,fontsize=8,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False)
for sp in ax1.spines.values(): sp.set_visible(False)
sca=ax2.scatter(U[:,0],U[:,1],s=3,c=conf,cmap="viridis",vmin=0,vmax=1,linewidths=0,rasterized=True)
ax2.set_title("Mapping confidence (max Mic-class probability)",fontsize=10.5,fontweight="bold"); ax2.set_xticks([]); ax2.set_yticks([])
fig.colorbar(sca,ax=ax2,shrink=0.6,label="confidence")
for sp in ax2.spines.values(): sp.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_green_mapped.png",dpi=150,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_green_mapped.png + microglia_green_mapped_labels.csv")
