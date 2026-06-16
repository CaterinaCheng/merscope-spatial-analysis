"""
358_loose_microglia_schpf.py
LOOSENED gate (restore power for spatial/proximity): keep microglia with n_genes>=40 and
microglial identity dominant over any contaminant (core>max contam) -- DROP the strict core>0
rule, recovering genuine but low-depth microglia. Retrain scHPF, Green 5-state annotate,
within-cluster ambient flag (drop clusters where a contaminant's mean counts exceed microglial).
Outputs microglia_loose_coords.csv (leiden, state, cluster_flag) for the spatial scripts.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, schpf
from scipy.sparse import csr_matrix, coo_matrix
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
K=10; NTRIALS=3; MIN_CELLS=50; LEIDEN_RES=0.6; NGENE_MIN=40
SIGS={"Microglia-core":["CX3CR1","CSF1R","C1QA","C1QB","C1QC","AIF1","TYROBP","P2RY12","P2RY13","TMEM119","CTSS","FCER1G"],
 "Astrocyte":["AQP4","GJA1","SLC1A3","SLC1A2","GLUL","GFAP","AQP9"],"Oligo":["MOG","MAL","PLP1","MOBP","CNP","MBP","UGT8"],
 "OPC":["PDGFRA","OLIG2","VCAN"],"Neuron":["SNAP25","SYT1","RBFOX3","GAD1","MEG3","NRGN","SLC17A7"],
 "Vascular":["PECAM1","CLDN5","VWF","PDGFRB","ACTA2","RGS5","COL1A1"],"Lymphoid":["CD3D","CD3E","IL7R","CD8A","CXCR6","NKG7","MS4A1","SKAP1"]}
CONTAM=[k for k in SIGS if k!="Microglia-core"]
BLOCK=set(("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD28 CD247 IL7R CXCR6 CCL5 LIME1 SKAP1 IL32 LCK THEMIS GZMK GZMA NKG7 GNLY KLRD1 KLRF1 KLRB1 KLRG1 KLRC1 ICOS FOXP3 CD40LG TBX21 CD7 LAG3 TIGIT NCR3 TNFRSF18 TNFRSF25 SPIB ITK PRF1 TC2N ACAP1 EOMES KLRK1 BLK STAT4 RASGRP1 SAMD3 CD27 GATA2 "
            "CD19 MS4A1 CD79A CD79B JCHAIN BANK1 IGHM IGHE EBF1 TNFRSF13C CD22 "
            "AQP4 SLC1A3 SLC1A2 GJA1 GLUL GFAP AQP9 VCAN SERPING1 PLPP3 SPOCK2 FYN "
            "MOG MAL PLP1 MOBP CNP MBP UGT8 GLDN OLIG2 S1PR5 SORT1 ATP8A1 PMP22 RNASE1 GSN OSBPL1A FMNL2 "
            "RBFOX3 SYT1 SNAP25 GAD1 RORB FOXP2 NRGN MEG3 XIST CNR1 NELL2 BCL11B KCNMA1 COL19A1 ZNF831 GNG2 BASP1 TSPYL2 ACTN1 SYNE1 APBA2 SLC17A7 "
            "PECAM1 CLDN5 VWF ACTA2 PDGFRB RGS5 NOTCH3 COL1A1 COL3A1 COL4A3 COL9A3 DCN AHNAK PDGFRA FN1 COBLL1 ITGA1 THBS1 ITM2A").split())
g6=pd.read_csv(NEW/"green_mic_state_signatures.csv")
def colg(key): return [x for x in g6[[c for c in g6.columns if key in c][0]].dropna()]
GSIG={"Homeostatic":colg("Mic.2"),"MHC-II/APC":colg("Mic.9"),"DAM":sorted(set(colg("Mic.12"))|set(colg("Mic.13"))),"Phagocytic":colg("Mic.7"),"Inflammatory/IEG":colg("Mic.15")}
GORD=list(GSIG.keys())
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    raw=f["layers/counts"]; Xr=csr_matrix((raw["data"][:],raw["indices"][:],raw["indptr"][:]),shape=tuple(int(s) for s in raw.attrs["shape"])).astype(np.float32)
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]; mid=idx[micidx]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=mid),var=pd.DataFrame(index=var))
ng=np.asarray((Xr[micidx]>0).sum(1)).ravel()
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,gl in SIGS.items(): sc.tl.score_genes(A,[x for x in gl if x in A.var_names],score_name=k,ctrl_size=50)
core=A.obs["Microglia-core"].values; cmax=A.obs[CONTAM].max(1).values
# LOOSENED gate
keep=(ng>=NGENE_MIN)&(core>cmax)
print("=== LOOSENED gate ===")
print(f"  total Mic: {len(mid)}; FAIL n_genes<{NGENE_MIN}: {(ng<NGENE_MIN).sum()}; FAIL contaminant-dominant: {((ng>=NGENE_MIN)&(core<=cmax)).sum()}")
print(f"  KEPT: {keep.sum()} ({100*keep.mean():.1f}%)  [strict kept 14043]")
Ak=A[keep].copy()
Xm=Xr[micidx][keep]; Xint=csr_matrix((np.rint(Xm.data),Xm.indices,Xm.indptr),shape=Xm.shape); Xint.eliminate_zeros()
varr=np.array(var); det=np.asarray((Xint>0).sum(0)).ravel()
intrinsic=np.array([(gn not in BLOCK) and (not str(gn).startswith("Blank")) for gn in varr])
gk=(det>=MIN_CELLS)&intrinsic; genes=list(varr[gk]); print(f"intrinsic genes: {len(genes)}")
Xcoo=coo_matrix(Xint[:,gk].astype(np.float64))
print(f"training scHPF K={K} ntrials={NTRIALS} on {Xcoo.shape[0]} cells x {len(genes)} genes ...")
model=schpf.run_trials(Xcoo,nfactors=K,ntrials=NTRIALS,verbose=False)
theta=model.cell_score(); FAC=[f"F{i+1}" for i in range(K)]
Ak.obsm["X_schpf"]=theta
sc.pp.neighbors(Ak,n_neighbors=15,use_rep="X_schpf")
try: sc.tl.leiden(Ak,resolution=LEIDEN_RES,key_added="leiden",flavor="igraph",n_iterations=2,directed=False)
except Exception as e: print("leiden fallback",e); sc.tl.leiden(Ak,resolution=LEIDEN_RES,key_added="leiden")
sc.tl.umap(Ak); U=Ak.obsm["X_umap"]
for k,gl in GSIG.items(): sc.tl.score_genes(Ak,[x for x in gl if x in Ak.var_names],score_name=k,ctrl_size=50)
Z=(Ak.obs[GORD]-Ak.obs[GORD].mean())/Ak.obs[GORD].std()
clusters=sorted(Ak.obs.leiden.unique(),key=int); cl_state={}
for cl in clusters:
    m=Ak.obs.leiden.values==cl; mz=Z.loc[m].mean().sort_values(ascending=False); cl_state[cl]=mz.index[0] if mz.iloc[0]>0.05 else "Mixed/low"
Ak.obs["state"]=[cl_state[c] for c in Ak.obs.leiden.values]
# within-cluster ambient flag: mean decontam counts microglial vs worst contaminant
def mct(gene): return np.asarray(Xd[micidx][keep][:,var.index(gene)].todense()).ravel() if gene in var else np.zeros(Ak.n_obs)
MICm=[g for g in ["C1QB","CSF1R","CX3CR1","AIF1","P2RY12","CTSS","C1QA"] if g in var]
CONm={"Astro":"AQP4","Oligo":"MOG","Neuron":"SNAP25","Vascular":"PECAM1","Lymphoid":"CD3E"}; CONm={k:v for k,v in CONm.items() if v in var}
micexpr=np.mean([mct(g) for g in MICm],0); conct={k:mct(v) for k,v in CONm.items()}
flagcl=[]
for cl in clusters:
    m=Ak.obs.leiden.values==cl; mic=micexpr[m].mean(); worst=max(conct[k][m].mean() for k in CONm)
    if worst/max(mic,1e-6)>1.0: flagcl.append(cl)
Ak.obs["cluster_flag"]=Ak.obs.leiden.isin(flagcl).values
print(f"\nflagged clusters (ambient>microglial): {flagcl}")
print("Green 5-state composition (unflagged):",{s:round(100*((~Ak.obs.cluster_flag)&(Ak.obs.state==s)).mean(),1) for s in GORD+['Mixed/low'] if ((~Ak.obs.cluster_flag)&(Ak.obs.state==s)).any()})
pd.DataFrame({"umap1":U[:,0],"umap2":U[:,1],"leiden":Ak.obs.leiden.values,"state":Ak.obs.state.values,"cluster_flag":Ak.obs.cluster_flag.values},index=Ak.obs_names).to_csv(NEW/"microglia_loose_coords.csv")
print(f"\nvalidated loose microglia: {int((~Ak.obs.cluster_flag).sum())}")
print("Saved: microglia_loose_coords.csv")
