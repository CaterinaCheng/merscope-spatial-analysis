"""
353_clean_microglia_schpf_qc.py
1. SEPARATE bona-fide microglia from all brain cells: QC-gate the Mic cells to remove
   ambient-contaminated / doublet / low-complexity cells exposed by 352.
     keep if: n_genes>=40 AND microglia-core score>0 AND microglia-core > every cross-lineage score.
2. Re-train microglia scHPF (intrinsic genes) on the CLEAN set; cluster in factor space.
3. Annotate Green 5-state; show per-cluster QC (core identity + contamination + depth) to prove clusters are real.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, schpf, matplotlib.pyplot as plt
import matplotlib.patheffects as pe, matplotlib.lines as ml
from scipy.sparse import csr_matrix, coo_matrix
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
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
GORD=list(GSIG.keys()); SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    raw=f["layers/counts"]; Xr=csr_matrix((raw["data"][:],raw["indices"][:],raw["indptr"][:]),shape=tuple(int(s) for s in raw.attrs["shape"])).astype(np.float32)
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]; mid=idx[micidx]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=mid),var=pd.DataFrame(index=var))
ng=np.asarray((Xr[micidx]>0).sum(1)).ravel(); tot=np.asarray(Xr[micidx].sum(1)).ravel()
A.obs["ng"]=ng; A.obs["tot"]=tot
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,gl in SIGS.items(): sc.tl.score_genes(A,[x for x in gl if x in A.var_names],score_name=k,ctrl_size=50)
core=A.obs["Microglia-core"].values; cmax=A.obs[CONTAM].max(1).values; cwhich=A.obs[CONTAM].idxmax(1).values
# ---- QC gate ----
pass_ng=ng>=NGENE_MIN; pass_core=core>0; pass_dom=core>cmax
keep=pass_ng&pass_core&pass_dom
print("=== microglia separation / QC gate ===")
print(f"  total Mic cells: {len(mid)}")
print(f"  FAIL low complexity (n_genes<{NGENE_MIN}): {(~pass_ng).sum()}")
print(f"  FAIL no microglia identity (core<=0): {(pass_ng&~pass_core).sum()}")
rem=pass_ng&pass_core&~pass_dom
print(f"  FAIL contaminant-dominant (core<=max contam): {rem.sum()}")
print("     dominant contaminant among those:",{k:int((cwhich[rem]==k).sum()) for k in CONTAM})
print(f"  KEPT bona-fide microglia: {keep.sum()} ({100*keep.mean():.1f}%)")
Ak=A[keep].copy(); kidx=np.where(keep)[0]
# ---- scHPF on clean microglia, intrinsic genes ----
Xm=Xr[micidx][keep]   # raw counts for scHPF
Xint=csr_matrix((np.rint(Xm.data),Xm.indices,Xm.indptr),shape=Xm.shape); Xint.eliminate_zeros()
varr=np.array(var); det=np.asarray((Xint>0).sum(0)).ravel()
intrinsic=np.array([(gn not in BLOCK) and (not str(gn).startswith("Blank")) for gn in varr])
gk=(det>=MIN_CELLS)&intrinsic; genes=list(varr[gk]); print(f"\nintrinsic genes for scHPF: {len(genes)}")
Xcoo=coo_matrix(Xint[:,gk].astype(np.float64))
print(f"training scHPF K={K} ntrials={NTRIALS} on {Xcoo.shape[0]} clean cells x {len(genes)} genes ...")
model=schpf.run_trials(Xcoo,nfactors=K,ntrials=NTRIALS,verbose=False)
schpf.save_model(model,str(NEW/"microglia_clean_schpf_K10.joblib"))
theta=model.cell_score(); beta=model.gene_score(); FAC=[f"F{i+1}" for i in range(K)]
pd.DataFrame(beta,index=genes,columns=FAC).to_csv(NEW/"microglia_clean_schpf_gene_scores.csv")
gsd=pd.DataFrame(beta,index=genes,columns=FAC); print("top genes per clean factor:")
for fc in FAC: print(f"  {fc}: "+", ".join(gsd[fc].nlargest(10).index))
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
print("\nGreen 5-state composition (clean):",{s:round(100*(Ak.obs.state==s).mean(),1) for s in GORD+['Mixed/low'] if (Ak.obs.state==s).any()})
pd.DataFrame({"umap1":U[:,0],"umap2":U[:,1],"leiden":Ak.obs.leiden.values,"state":Ak.obs.state.values},index=Ak.obs_names).to_csv(NEW/"microglia_clean_coords.csv")
# ---- per-cluster QC ----
rows=[]
for cl in clusters:
    m=Ak.obs.leiden.values==cl; d=dict(cl=cl,n=int(m.sum()),med_counts=int(np.median(Ak.obs.tot.values[m])),med_genes=int(np.median(Ak.obs.ng.values[m])),state=cl_state[cl])
    for s in SIGS: d[s]=Ak.obs[s].values[m].mean()
    rows.append(d)
T=pd.DataFrame(rows); T=T.iloc[sorted(range(len(T)),key=lambda i:(GORD.index(T.state[i]) if T.state[i] in GORD else 9,-T["Microglia-core"][i]))].reset_index(drop=True)
T.to_csv(NEW/"microglia_clean_cluster_qc.csv",index=False)
sc.tl.rank_genes_groups(Ak,"leiden",method="wilcoxon",n_genes=10)
print("\n=== CLEAN per-cluster QC + top DE genes ===")
print(f"{'cl':>3} {'n':>5} {'cnts':>5} {'gns':>4} {'state':16} | top DE genes")
for cl in T.cl:
    names=[Ak.uns['rank_genes_groups']['names'][str(cl)][i] for i in range(8)]; r=T[T.cl==cl].iloc[0]
    print(f"{cl:>3} {r.n:>5} {r.med_counts:>5} {r.med_genes:>4} {r.state:16} | {', '.join(names)}")
# ================= QC FIGURE =================
SIGN=list(SIGS.keys()); ncl=len(T); yt=np.arange(ncl)
fig=plt.figure(figsize=(16,max(6,0.42*ncl))); gsf=fig.add_gridspec(1,4,width_ratios=[0.22,1.0,0.55,1.0],wspace=0.06)
axs=fig.add_subplot(gsf[0,0])
for i,st in enumerate(T.state): axs.add_patch(plt.Rectangle((0,i-0.5),1,1,color=SCOL.get(st,"#ccc")))
axs.set_xlim(0,1); axs.set_ylim(ncl-0.5,-0.5); axs.set_xticks([]); axs.set_yticks(yt); axs.set_yticklabels([f"cl{c}" for c in T.cl],fontsize=8); axs.set_title("state",fontsize=9)
for sp in axs.spines.values(): sp.set_visible(False)
axh=fig.add_subplot(gsf[0,1]); H=T[SIGN].values; Hz=(H-H.mean(0))/(H.std(0)+1e-9)
im=axh.imshow(Hz,cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
axh.set_xticks(range(len(SIGN))); axh.set_xticklabels(SIGN,rotation=35,ha="right"); axh.set_yticks(yt); axh.set_yticklabels([])
for i in range(ncl):
    for j in range(len(SIGN)): axh.text(j,i,f"{H[i,j]:.2f}",ha="center",va="center",fontsize=6.5,color="white" if abs(Hz[i,j])>1.3 else "#333")
axh.set_title("identity / contamination (number=raw score; color=z across clusters)",fontsize=9.5,fontweight="bold"); fig.colorbar(im,ax=axh,shrink=0.5,label="z")
axq=fig.add_subplot(gsf[0,2]); axq.barh(yt-0.2,T.med_counts,height=0.4,color="#555"); axq2=axq.twiny(); axq2.barh(yt+0.2,T.med_genes,height=0.4,color="#E67E22")
axq.set_ylim(ncl-0.5,-0.5); axq.set_yticks([]); axq.set_xlabel("median counts",fontsize=8); axq2.set_xlabel("median n_genes",color="#E67E22",fontsize=8); axq.set_title("depth/complexity",fontsize=9.5,fontweight="bold")
for sp in axq.spines.values(): sp.set_visible(False)
# UMAP by Green state
axU=fig.add_subplot(gsf[0,3])
for s in [x for x in GORD if (Ak.obs.state==x).any()]+([("Mixed/low")] if (Ak.obs.state=="Mixed/low").any() else []):
    m=Ak.obs.state.values==s; axU.scatter(U[m,0],U[m,1],s=3,c=SCOL.get(s,"#bbb"),linewidths=0,alpha=0.85,rasterized=True,label=f"{s} ({100*m.mean():.0f}%)")
axU.set_xticks([]); axU.set_yticks([]); axU.set_aspect("equal"); axU.set_title("clean microglia UMAP\n(scHPF + Green 5-state)",fontsize=9.5,fontweight="bold"); axU.legend(markerscale=3,fontsize=7.5,loc="center left",bbox_to_anchor=(1.02,0.5),frameon=False)
for sp in axU.spines.values(): sp.set_visible(False)
fig.suptitle(f"CLEAN microglia ({keep.sum()} of {len(mid)} cells passed QC) — scHPF clustering QC",fontsize=13,fontweight="bold",y=1.0)
plt.tight_layout(); fig.savefig(NEW/"microglia_clean_cluster_qc.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_clean_cluster_qc.png + microglia_clean_{coords,cluster_qc}.csv + model/scores")
