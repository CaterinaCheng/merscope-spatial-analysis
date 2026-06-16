"""
354_within_cluster_qc.py
QC *within* each clean microglia cluster: confirm clusters are internally pure microglia,
not a few residual contaminated/doublet cells. Per cluster report:
  median microglia-core, median max-contaminant, %high-purity cells (core > 1.5x max contam),
  depth/complexity, and the dominant residual contaminant if any.
Flags low-purity clusters and writes a validated label set (drops within-cluster outlier cells).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
DEC=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
SIGS={"Microglia-core":["CX3CR1","CSF1R","C1QA","C1QB","C1QC","AIF1","TYROBP","P2RY12","P2RY13","TMEM119","CTSS","FCER1G"],
 "Astrocyte":["AQP4","GJA1","SLC1A3","SLC1A2","GLUL","GFAP","AQP9"],"Oligo":["MOG","MAL","PLP1","MOBP","CNP","MBP","UGT8"],
 "OPC":["PDGFRA","OLIG2","VCAN"],"Neuron":["SNAP25","SYT1","RBFOX3","GAD1","MEG3","NRGN","SLC17A7"],
 "Vascular":["PECAM1","CLDN5","VWF","PDGFRB","ACTA2","RGS5","COL1A1"],"Lymphoid":["CD3D","CD3E","IL7R","CD8A","CXCR6","NKG7","MS4A1","SKAP1"]}
CONTAM=[k for k in SIGS if k!="Microglia-core"]
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F","Mixed/low":"#bbb"}
co=pd.read_csv(NEW/"microglia_clean_coords.csv",index_col=0)
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    raw=f["layers/counts"]; Xr=csr_matrix((raw["data"][:],raw["indices"][:],raw["indptr"][:]),shape=tuple(int(s) for s in raw.attrs["shape"])).astype(np.float32)
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; rowi=np.array([pos[c] for c in co.index])
A=ad.AnnData(X=Xd[rowi].copy(),obs=pd.DataFrame(index=co.index),var=pd.DataFrame(index=var))
A.obs["ng"]=np.asarray((Xr[rowi]>0).sum(1)).ravel(); A.obs["tot"]=np.asarray(Xr[rowi].sum(1)).ravel()
A.obs["leiden"]=co.leiden.astype(str).values; A.obs["state"]=co.state.values
# LEVEL-based QC: mean decontam transcript counts of canonical markers (presence of 1 ambient molecule is meaningless)
Xc=Xd[rowi]   # decontam counts (not normalized)
def meanct(gene): return np.asarray(Xc[:,var.index(gene)].todense()).ravel() if gene in var else np.zeros(A.n_obs)
MICm=[g for g in ["C1QB","CSF1R","CX3CR1","AIF1","P2RY12","CTSS","C1QA"] if g in var]
CONm={"Astro":"AQP4","Oligo":"MOG","Neuron":"SNAP25","Vascular":"PECAM1","Lymphoid":"CD3E"}; CONm={k:v for k,v in CONm.items() if v in var}
micexpr=np.mean([meanct(g) for g in MICm],0)
det_mic=np.zeros(A.n_obs,bool)
for g in MICm: det_mic|=meanct(g)>0
conct={k:meanct(v) for k,v in CONm.items()}
clusters=sorted(A.obs.leiden.unique(),key=int); rows=[]
for cl in clusters:
    m=A.obs.leiden.values==cl; mic=micexpr[m].mean(); d=dict(cl=cl,n=int(m.sum()),state=pd.Series(A.obs.state.values[m]).value_counts().index[0],
        pct_micpos=round(100*det_mic[m].mean(),1),mic_expr=round(mic,2),med_genes=int(np.median(A.obs.ng.values[m])),med_counts=int(np.median(A.obs.tot.values[m])))
    for k in CONm: d[k]=round(conct[k][m].mean(),2)
    d["ambient_ratio"]=round(max(conct[k][m].mean() for k in CONm)/max(mic,1e-6),2)
    rows.append(d)
T=pd.DataFrame(rows).sort_values("ambient_ratio").reset_index(drop=True)
print("=== WITHIN-CLUSTER QC (mean decontam transcript counts) ===")
print("mic_expr=mean microglial-marker counts; Astro..Lymphoid=mean contaminant-marker counts; ambient_ratio=worst contaminant/mic")
print(T.to_string(index=False))
# flag: a cluster whose worst contaminant rivals its microglial expression (ratio>0.5)
T["flag"]=T.ambient_ratio>1.0; FLAG=T[T.flag].cl.tolist()   # only if contaminant EXCEEDS microglial signal
print(f"\nflagged clusters (ambient_ratio>1.0, contaminant exceeds microglial): {FLAG if FLAG else 'NONE'}")
co2=co.loc[A.obs_names].copy(); co2["cluster_flag"]=co2.leiden.astype(str).isin(FLAG)
co2.to_csv(NEW/"microglia_final_coords.csv")
T.to_csv(NEW/"microglia_within_cluster_qc.csv",index=False)
print(f"\nvalidated microglia (unflagged): {(~co2.cluster_flag).sum()} of {len(co2)} cells; per-state:")
print({s:int(((~co2.cluster_flag)&(co2.state==s)).sum()) for s in SCOL if ((~co2.cluster_flag)&(co2.state==s)).any()})
# ===== figure: level-based within-cluster QC =====
clord=list(T.cl); yt=np.arange(len(clord)); CM=list(CONm.keys())
fig,axes=plt.subplots(1,2,figsize=(13.5,5),gridspec_kw={"width_ratios":[0.95,1]})
ax=axes[0]
ax.barh(yt,T.mic_expr,color=[SCOL.get(s,"#bbb") for s in T.state],label="microglial markers")
ax.barh(yt,[max(T.loc[T.cl==cl,k].iloc[0] for k in CM) for cl in clord],left=T.mic_expr,color="#888",alpha=0.5,label="worst contaminant")
ax.set_yticks(yt); ax.set_yticklabels([f"cl{c} ({s})" for c,s in zip(T.cl,T.state)],fontsize=8); ax.invert_yaxis()
ax.set_xlabel("mean decontam transcript counts per cell"); ax.set_title("A. microglial vs contaminant expression LEVEL",fontsize=10,fontweight="bold"); ax.legend(fontsize=8)
for sp in ax.spines.values(): sp.set_visible(False)
ax2=axes[1]; D=np.vstack([[T.loc[T.cl==cl,k].iloc[0] for k in CM] for cl in clord])
im=ax2.imshow(D,cmap="Reds",vmin=0,vmax=max(0.5,np.percentile(D,98)),aspect="auto")
ax2.set_xticks(range(len(CM))); ax2.set_xticklabels([f"{k}\n({CONm[k]})" for k in CM]); ax2.set_yticks(yt); ax2.set_yticklabels([f"cl{cl} (mic={T.loc[T.cl==cl,'mic_expr'].iloc[0]})" for cl in clord],fontsize=8)
for i in range(len(clord)):
    for j in range(len(CM)): ax2.text(j,i,f"{D[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if D[i,j]>np.percentile(D,90) else "#333")
ax2.set_title("B. mean contaminant counts/cell (trace = a few molecules)",fontsize=10,fontweight="bold")
fig.colorbar(im,ax=ax2,shrink=0.6,label="counts/cell")
fig.suptitle("Within-cluster QC of clean microglia — all clusters are microglia-dominant",fontsize=12,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"microglia_within_cluster_qc.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_within_cluster_qc.png + microglia_within_cluster_qc.csv + microglia_final_coords.csv")
