"""
346_microglia_schpf_green.py  (FINAL microglia clustering = scHPF, annotation = Green2024)
1. Subset decontam microglia, round to integer counts, gene-filter.
2. Train microglia-specific scHPF (De Jager/Marshe factor method) -> cell scores theta, gene scores beta.
3. Cluster in scHPF FACTOR SPACE (neighbors on theta -> Leiden) and embed (UMAP on theta).
4. Annotate each cluster with the dominant GREEN2024 microglial state (literature signatures, score_genes).
Outputs model + scores + UMAP(figure colored by Green state, clusters numbered) + Green-marker dotplot.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, schpf, joblib, matplotlib.pyplot as plt
import matplotlib.patheffects as pe, matplotlib.lines as ml
from scipy.sparse import csr_matrix, coo_matrix
import warnings; warnings.filterwarnings("ignore")
sc.settings.verbosity=0; plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
K=10; NTRIALS=3; MIN_CELLS=50; LEIDEN_RES=0.6; rng=np.random.RandomState(0)
# cross-lineage markers + blank probes to EXCLUDE so scHPF factors are microglia-intrinsic
BLOCK=set(("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD28 CD247 IL7R CXCR6 CCL5 LIME1 SKAP1 IL32 LCK THEMIS GZMK GZMA NKG7 GNLY KLRD1 KLRF1 KLRB1 KLRG1 KLRC1 ICOS FOXP3 CD40LG TBX21 CD7 LAG3 TIGIT NCR3 TNFRSF18 TNFRSF25 SPIB "
            "CD19 MS4A1 CD79A CD79B JCHAIN BANK1 IGHM IGHE EBF1 TNFRSF13C "
            "AQP4 SLC1A3 SLC1A2 GJA1 GLUL GFAP AQP9 VCAN SERPING1 PLPP3 "
            "MOG MAL PLP1 MOBP CNP MBP UGT8 GLDN OLIG2 S1PR5 CD22 "
            "RBFOX3 SYT1 SNAP25 GAD1 RORB FOXP2 NRGN MEG3 XIST CNR1 NELL2 BCL11B KCNMA1 COL19A1 ZNF831 GNG2 "
            "PECAM1 CLDN5 VWF ACTA2 PDGFRB RGS5 NOTCH3 COL1A1 COL3A1 COL4A3 COL9A3 DCN AHNAK RNASE1 PDGFRA FN1 COBLL1 ITGA1 GJA1").split())
SCOL={"Homeostatic (Mic.2)":"#3498DB","MHC-II/APC (Mic.9)":"#9B59B6","Activated-DAM (Mic.12)":"#E74C3C",
      "Lipid-DAM (Mic.13)":"#E67E22","Inflammatory/IEG (Mic.15)":"#F1C40F","Phagocytic-myeloid (Mic.7)":"#16A085","Mixed/low":"#BDC3C7"}
sig=pd.read_csv(NEW/"green_mic_state_signatures.csv"); SIG={c:[g for g in sig[c].dropna()] for c in sig.columns}
STATEORD=list(SIG.keys())
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]; print("microglia:",len(micidx))
Xm=Xd[micidx].copy()
# integer counts for scHPF; gene filter
Xint=csr_matrix((np.rint(Xm.data),Xm.indices,Xm.indptr),shape=Xm.shape); Xint.eliminate_zeros()
varr=np.array(var); det=np.asarray((Xint>0).sum(0)).ravel()
intrinsic=np.array([(gn not in BLOCK) and (not str(gn).startswith("Blank")) for gn in varr])
gkeep=(det>=MIN_CELLS)&intrinsic; genes=list(varr[gkeep])
Xg=Xint[:,gkeep]; print(f"microglia-intrinsic genes kept (>= {MIN_CELLS} cells, non-cross-lineage, non-Blank): {len(genes)} of {len(var)}")
Xcoo=coo_matrix(Xg.astype(np.float64))
# ---- train microglia scHPF ----
print(f"training scHPF K={K} ntrials={NTRIALS} on {Xg.shape[0]} cells x {Xg.shape[1]} genes ...")
model=schpf.run_trials(Xcoo,nfactors=K,ntrials=NTRIALS,verbose=False)
MODELPATH=NEW/f"microglia_schpf_K{K}.joblib"; schpf.save_model(model,str(MODELPATH))
theta=model.cell_score(); beta=model.gene_score()    # cells x K , genes x K
FAC=[f"F{i+1}" for i in range(K)]
pd.DataFrame(beta,index=genes,columns=FAC).to_csv(NEW/"microglia_schpf_gene_scores.csv")
pd.DataFrame(theta,index=idx[micidx],columns=FAC).to_csv(NEW/"microglia_schpf_cell_scores.csv")
print("top genes per scHPF factor:")
gsd=pd.DataFrame(beta,index=genes,columns=FAC)
for fcol in FAC: print(f"  {fcol}: "+", ".join(gsd[fcol].nlargest(10).index))
# ---- cluster in scHPF factor space ----
A=ad.AnnData(X=Xm.copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)   # for Green signature scoring
A.obsm["X_schpf"]=theta
sc.pp.neighbors(A,n_neighbors=15,use_rep="X_schpf");
try: sc.tl.leiden(A,resolution=LEIDEN_RES,key_added="leiden",flavor="igraph",n_iterations=2,directed=False)
except Exception as e: print("igraph leiden fallback:",e); sc.tl.leiden(A,resolution=LEIDEN_RES,key_added="leiden")
sc.tl.umap(A); U=A.obsm["X_umap"]
# ---- Green annotation ----
for k,gl in SIG.items(): sc.tl.score_genes(A,[g for g in gl if g in A.var_names],score_name=k,ctrl_size=50)
Z=(A.obs[STATEORD]-A.obs[STATEORD].mean())/A.obs[STATEORD].std()
clusters=sorted(A.obs.leiden.unique(),key=int); cl_state={}; rows=[]
for cl in clusters:
    m=A.obs.leiden.values==cl; mz=Z.loc[m].mean().sort_values(ascending=False)
    cl_state[cl]= mz.index[0] if mz.iloc[0]>0.05 else "Mixed/low"
    rows.append(dict(leiden=cl,n=int(m.sum()),green_state=cl_state[cl],topz=round(mz.iloc[0],3),second=mz.index[1],secz=round(mz.iloc[1],3)))
INFO=pd.DataFrame(rows); print("\nLeiden(scHPF) cluster -> Green state:"); print(INFO.to_string(index=False))
A.obs["green_state"]=[cl_state[c] for c in A.obs.leiden.values]
INFO.to_csv(NEW/"microglia_schpf_green_cluster_assignment.csv",index=False)
pd.DataFrame({"umap1":U[:,0],"umap2":U[:,1],"leiden":A.obs.leiden.values,"green_state":A.obs.green_state.values},index=A.obs_names).to_csv(NEW/"microglia_schpf_green_coords.csv")
present=[s for s in STATEORD if (A.obs.green_state==s).any()]+(["Mixed/low"] if (A.obs.green_state=="Mixed/low").any() else [])
print("\nGreen state composition:",{s:round(100*(A.obs.green_state==s).mean(),1) for s in present})
# ================= FIGURE =================
fig=plt.figure(figsize=(17,7.5)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.05])
axU=fig.add_subplot(gs[0,0])
for s in present:
    m=A.obs.green_state.values==s
    axU.scatter(U[m,0],U[m,1],s=4,c=SCOL[s],linewidths=0,alpha=0.85,rasterized=True,label=f"{s} ({100*m.mean():.0f}%)")
for cl in clusters:
    pts=U[A.obs.leiden.values==cl]
    if len(pts)<10: continue
    c=np.median(pts,0); d=np.linalg.norm(pts-c,axis=1); core=pts[d<=np.percentile(d,60)]; c=core.mean(0)
    axU.text(c[0],c[1],cl,fontsize=12,fontweight="bold",ha="center",va="center",color="white",path_effects=[pe.withStroke(linewidth=3.0,foreground="#2b2b2b")],zorder=10)
x0,x1=U[:,0].min(),U[:,0].max(); y0,y1=U[:,1].min(),U[:,1].max(); al=0.16*(x1-x0); ox=x0-0.02*(x1-x0); oy=y0-0.02*(y1-y0)
axU.annotate("",xy=(ox+al,oy),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
axU.annotate("",xy=(ox,oy+al),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
axU.text(ox+al*.5,oy-0.03*(y1-y0),"UMAP1",fontsize=8,ha="center",va="top"); axU.text(ox-0.03*(x1-x0),oy+al*.5,"UMAP2",fontsize=8,ha="right",va="center",rotation=90)
axU.set_xticks([]); axU.set_yticks([]); axU.set_aspect("equal")
for sp in axU.spines.values(): sp.set_visible(False)
axU.set_title(f"Microglia UMAP — scHPF (K={K}) clustering, Green2024 state annotation",fontsize=11,fontweight="bold")
axU.legend(markerscale=3,fontsize=8.5,loc="upper right",frameon=False)
# dotplot Green markers x clusters
axD=fig.add_subplot(gs[0,1]); MARK=[]; glab=[]; used=set()
for s in STATEORD:
    for gn in SIG[s][:5]:
        if gn in A.var_names and gn not in used: MARK.append((s,gn)); glab.append(gn); used.add(gn)
clord=list(clusters); gl=[g for _,g in MARK]; E=np.asarray(A[:,gl].X.todense())
emat=pd.DataFrame(E,columns=gl); clv=A.obs.leiden.values
mean_e=emat.groupby(clv).mean().reindex(clord); pct_e=(emat>0).groupby(clv).mean().reindex(clord)
mn=(mean_e-mean_e.min())/(mean_e.max()-mean_e.min()+1e-9)
for gi,(s,gn) in enumerate(MARK):
    for ci,cl in enumerate(clord):
        axD.scatter(ci,gi,s=10+float(pct_e.loc[cl,gn])*180,c=[plt.cm.Reds(float(mn.loc[cl,gn]))],edgecolors="#999",linewidths=0.3)
    axD.add_patch(plt.Rectangle((-1.4,gi-0.5),0.5,1,color=SCOL[s],clip_on=False))
axD.set_xticks(range(len(clord))); axD.set_xticklabels(clord,fontsize=8); axD.set_xlabel("Leiden cluster (scHPF space)")
axD.set_yticks(range(len(MARK))); axD.set_yticklabels(glab,fontsize=8)
axD.set_xlim(-1.5,len(clord)-0.5); axD.set_ylim(len(MARK)-0.5,-0.5)
axD.set_title("Green2024 state markers across clusters (size=%expr, color=scaled mean)",fontsize=10,fontweight="bold")
for sp in axD.spines.values(): sp.set_visible(False)
handles=[ml.Line2D([0],[0],marker="o",linestyle="",markersize=8,markerfacecolor=SCOL[s],markeredgecolor="#999",label=s) for s in STATEORD]
axD.legend(handles=handles,fontsize=7.5,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False,title="Green state")
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_schpf_green.png",dpi=200,bbox_inches="tight"); plt.close()
print(f"\nSaved: microglia_umap_schpf_green.png + microglia_schpf_K{K}.joblib + gene/cell scores + cluster assignment + coords")
