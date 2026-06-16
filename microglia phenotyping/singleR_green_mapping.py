"""
338_singleR_green_mapping.py
Cross-platform label transfer of our (decontam) microglia onto Green2024 microglial CLUSTERS
via SingleR-style Spearman nearest-centroid mapping (scale-invariant; robust to snRNA->imaging
domain shift, unlike CellTypist). Reference = Green per-Mic-cluster mean expression on shared
panel genes. Informative genes = variable across Green Mic centroids (cross-lineage genes blocked).
Outputs UMAP colored by mapped Green cluster + mapping margin (confidence).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import csr_matrix
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
NTOP=200
MICNAME={"Mic.2":"Homeostatic","Mic.9":"MHC-II/APC","Mic.12":"Activated-DAM","Mic.13":"Lipid-DAM",
         "Mic.15":"Inflammatory/IEG","Mic.7":"Phagocytic","Mic.10":"Mic.10","Mic.6":"Mic.6","Mic.1":"Mic.1",
         "Mic.3":"Mic.3","Mic.4":"Mic.4","Mic.5":"Mic.5","Mic.8":"Mic.8","Mic.11":"Mic.11","Mic.14":"Mic.14","Mic.16":"Mic.16"}
BLOCK=set(("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD28 CD247 IL7R CXCR6 CCL5 LIME1 SKAP1 IL32 LCK THEMIS GZMK GZMA NKG7 GNLY KLRD1 KLRB1 ICOS FOXP3 CD40LG TBX21 "
            "CD19 MS4A1 CD79A CD79B JCHAIN BANK1 IGHM "
            "AQP4 SLC1A3 SLC1A2 GJA1 GLUL GFAP "
            "MOG MAL PLP1 MOBP CNP MBP UGT8 GLDN "
            "RBFOX3 SYT1 SNAP25 GAD1 RORB FOXP2 NRGN MEG3 "
            "PECAM1 CLDN5 VWF ACTA2 PDGFRB RGS5 NOTCH3 COL1A1 COL3A1 DCN AHNAK RNASE1").split())
# ----- Green reference centroids -----
ae=pd.read_csv(NEW/"green_mic_state_mean_expr.csv",index_col=0)
ae.columns=[c.split(".",1)[-1] if c.startswith("SCT") else c for c in ae.columns]
RESOLVABLE=["Mic.2","Mic.7","Mic.9","Mic.12","Mic.13","Mic.15"]  # only states our immune panel can represent
micstates=[c for c in ae.columns if c in RESOLVABLE]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
panel=[gn for gn in var if gn in ae.index and gn not in BLOCK]
C=ae.loc[panel,micstates]  # genes x clusters (log mean)
# informative genes: most variable across Green Mic centroids
ginf=C.var(1).sort_values(ascending=False).head(NTOP).index.tolist()
print(f"reference: {len(micstates)} Mic clusters, {len(ginf)} informative genes (of {len(panel)} shared)")
# ----- query: decontam microglia, counts-per-median log1p -----
micidx=np.where(v2=="Mic")[0]; A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
vp={gn:i for i,gn in enumerate(var)}
Q=np.asarray(A[:,ginf].X.todense())   # cells x ginf
Cref=C.loc[ginf,micstates].values.T   # clusters x ginf
# ----- Spearman: rank within each row over genes, then Pearson on ranks -----
def rankrows(Mx): return Mx.argsort(1).argsort(1).astype(np.float32)
Qr=rankrows(Q); Cr=rankrows(Cref)
Qc=Qr-Qr.mean(1,keepdims=True); Cc=Cr-Cr.mean(1,keepdims=True)
Qn=Qc/(np.linalg.norm(Qc,axis=1,keepdims=True)+1e-9); Cn=Cc/(np.linalg.norm(Cc,axis=1,keepdims=True)+1e-9)
corr=Qn@Cn.T   # cells x clusters
srt=np.argsort(-corr,1); top=srt[:,0]; mapped=np.array(micstates)[top]
margin=corr[np.arange(len(corr)),srt[:,0]]-corr[np.arange(len(corr)),srt[:,1]]
A.obs["green_mic"]=mapped; A.obs["green_name"]=[MICNAME.get(m,m) for m in mapped]; A.obs["margin"]=margin; A.obs["topcorr"]=corr.max(1)
vc=pd.Series(mapped).value_counts()
print("\nmapped Green Mic cluster distribution (SingleR-style):")
for k,v in vc.items(): print(f"  {k:7} {MICNAME.get(k,k):16}: {v:6d} ({100*v/len(mapped):4.1f}%)  median topcorr={np.median(corr[mapped==k].max(1)):.2f}")
pd.DataFrame({"green_mic":mapped,"green_name":A.obs.green_name.values,"topcorr":corr.max(1),"margin":margin},index=A.obs_names).to_csv(NEW/"microglia_green_singleR_labels.csv")
# ----- UMAP from 336 -----
co=pd.read_csv(NEW/"microglia_umap_coords.csv",index_col=0); U=co.reindex(A.obs_names)[["umap1","umap2"]].values
order=[c for c in ["Mic.2","Mic.9","Mic.12","Mic.13","Mic.15","Mic.7","Mic.10","Mic.6","Mic.1","Mic.3","Mic.4","Mic.5","Mic.8"] if c in set(mapped)]
pal=cm.get_cmap("tab20",max(len(order),3))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17,8))
for i,c in enumerate(order):
    m=mapped==c
    ax1.scatter(U[m,0],U[m,1],s=3,c=[pal(i)],label=f"{c} {MICNAME.get(c,'')} ({100*m.mean():.0f}%)",linewidths=0,rasterized=True)
ax1.set_title("Microglia UMAP -> Green2024 clusters (SingleR-style correlation mapping, decontam)",fontsize=10,fontweight="bold")
ax1.set_xticks([]); ax1.set_yticks([]); ax1.legend(markerscale=3,fontsize=8,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False)
for sp in ax1.spines.values(): sp.set_visible(False)
sca=ax2.scatter(U[:,0],U[:,1],s=3,c=margin,cmap="magma",vmin=0,vmax=np.percentile(margin,95),linewidths=0,rasterized=True)
ax2.set_title("Mapping margin (top corr - 2nd corr; higher = more decisive)",fontsize=10,fontweight="bold"); ax2.set_xticks([]); ax2.set_yticks([])
fig.colorbar(sca,ax=ax2,shrink=0.6,label="margin")
for sp in ax2.spines.values(): sp.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_green_singleR.png",dpi=150,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_green_singleR.png + microglia_green_singleR_labels.csv")
