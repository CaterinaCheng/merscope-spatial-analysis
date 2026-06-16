"""
344_microglia_umap_5states.py
Re-do the microglia UMAP with OUR original unsupervised pipeline (decontam counts ->
normalize_total(None) -> log1p -> scale -> PCA30 -> neighbors15 -> Leiden -> UMAP),
then ANNOTATE each Leiden cluster with the 5 MAIN Tuddenham2024 microglia states:
  Homeostatic | Antigen-presenting | Motility | Proliferative | DAM/disease-assoc
State signatures = pooled up-genes (Suppl Table 2) of the defining Tuddenham clusters,
panel- and cross-lineage-filtered. Cluster -> state = argmax mean z signature.
Outputs: UMAP colored by state (clusters numbered) + marker dotplot to justify calls.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore")
sc.settings.verbosity=0; plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); REF=NEW/"reference"
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
TOPN=20; MINGENES=5; LEIDEN_RES=0.6
# 5 main states -> defining Tuddenham fine clusters (paper Fig2 / abstract)
STATES={"Homeostatic":[2,3,4],"Antigen-presenting":[10,13],"Motility":[5],"Proliferative":[12],"DAM/disease-assoc":[11]}
SORDER=list(STATES.keys())
SCOL={"Homeostatic":"#4C9F70","Antigen-presenting":"#4E79A7","Motility":"#76B7B2",
      "Proliferative":"#9467BD","DAM/disease-assoc":"#B07A3C","Unresolved":"#DDDDDD"}
BLOCK=set(("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD28 CD247 IL7R CXCR6 CCL5 LIME1 SKAP1 IL32 LCK THEMIS GZMK GZMA NKG7 GNLY KLRD1 KLRB1 KLRG1 ICOS FOXP3 CD40LG TBX21 CD7 "
            "CD19 MS4A1 CD79A CD79B JCHAIN BANK1 IGHM AQP4 SLC1A3 SLC1A2 GJA1 GLUL GFAP MOG MAL PLP1 MOBP CNP MBP UGT8 GLDN "
            "RBFOX3 SYT1 SNAP25 GAD1 RORB FOXP2 NRGN MEG3 XIST PECAM1 CLDN5 VWF ACTA2 PDGFRB RGS5 NOTCH3 COL1A1 COL3A1 DCN AHNAK RNASE1").split())
CYCLING=set("MKI67 TOP2A PCNA CDK1 CCNB1 CCNB2 CENPF MCM2 MCM6 BIRC5 UBE2C TYMS RRM2 NUSAP1 STMN1 HMGB2".split())
up=pd.read_excel(REF/"tuddenham/MOESM4.xlsx",sheet_name="Upregulated genes")
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vset=set(var)
# 1) raw pooled markers per state
RAW={}
for st,cls in STATES.items():
    sub=up[up.up_type.isin(cls)].sort_values("sum_logFC",ascending=False); genes=[]; seen=set()
    for gn in sub.gene:
        if gn in vset and gn not in BLOCK and gn not in seen: genes.append(gn); seen.add(gn)
        if len(genes)>=TOPN: break
    RAW[st]=genes
# 2) Proliferative needs CANONICAL cycling genes -> none on panel => unscoreable blind spot
prolif_hits=[g for g in CYCLING if g in vset]
RAW["Proliferative"]=prolif_hits   # empty -> dropped below
# 3) DISCRIMINATIVE filter: keep genes unique to ONE state (drop cross-state shared markers)
from collections import Counter
mult=Counter(g for gs in RAW.values() for g in set(gs))
SIG={}
for st,genes in RAW.items():
    disc=[g for g in genes if mult[g]==1]
    if len(disc)>=MINGENES: SIG[st]=disc
    else: print(f"  [drop] {st}: only {len(disc)} discriminative panel markers -> UNSCOREABLE")
print(f"scoreable main states: {len(SIG)} of 5 (Proliferative cycling genes on panel: {prolif_hits or 'NONE'})")
for k,v in SIG.items(): print(f"  {k:20}: {', '.join(v)}")
SCOREORD=list(SIG.keys())
# ---- OUR ORIGINAL UNSUPERVISED PIPELINE ----
micidx=np.where(v2=="Mic")[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A); A.raw=A
det=np.asarray((A.X>0).mean(0)).ravel(); keep=np.array(var)[det>=0.05]
Asc=A[:,keep].copy(); sc.pp.scale(Asc,max_value=10); sc.tl.pca(Asc,n_comps=30); sc.pp.neighbors(Asc,n_neighbors=15,n_pcs=30)
try: sc.tl.leiden(Asc,resolution=LEIDEN_RES,key_added="leiden",flavor="igraph",n_iterations=2,directed=False)
except Exception as e: print("igraph leiden fallback:",e); sc.tl.leiden(Asc,resolution=LEIDEN_RES,key_added="leiden")
sc.tl.umap(Asc); U=Asc.obsm["X_umap"]; A.obs["leiden"]=Asc.obs["leiden"].values
for k,genes in SIG.items(): sc.tl.score_genes(A,[g for g in genes if g in A.var_names],score_name=k,ctrl_size=50)
# ---- annotate each Leiden cluster by dominant state ----
SPEC=[s for s in SCOREORD if s!="Homeostatic"]   # specialized programs (deviations from baseline)
Z=(A.obs[SCOREORD]-A.obs[SCOREORD].mean())/A.obs[SCOREORD].std()
clusters=sorted(A.obs.leiden.unique(),key=int); cl_state={}; rows=[]
for cl in clusters:
    m=A.obs.leiden.values==cl; mz=Z.loc[m].mean().sort_values(ascending=False)
    msp=Z.loc[m,SPEC].mean().sort_values(ascending=False)   # best specialized program
    # default = Homeostatic baseline; assign a specialized state only if it is positively enriched
    state=msp.index[0] if msp.iloc[0]>0.25 else "Homeostatic"
    cl_state[cl]=state
    rows.append(dict(leiden=cl,n=int(m.sum()),state=state,best_spec=msp.index[0],best_spec_z=round(msp.iloc[0],3),homeo_z=round(Z.loc[m,"Homeostatic"].mean(),3)))
INFO=pd.DataFrame(rows); print("\nLeiden cluster -> main state:"); print(INFO.to_string(index=False))
A.obs["state"]=[cl_state[c] for c in A.obs.leiden.values]
INFO.to_csv(NEW/"microglia_5state_cluster_assignment.csv",index=False)
pd.DataFrame({"umap1":U[:,0],"umap2":U[:,1],"leiden":A.obs.leiden.values,"state":A.obs.state.values},index=A.obs_names).to_csv(NEW/"microglia_5state_coords.csv")
present=[s for s in SORDER if (A.obs.state==s).any()]
print("\nstate composition:",{s:round(100*(A.obs.state==s).mean(),1) for s in present})
unscore=[s for s in SORDER if s not in SIG]
if unscore: print("UNSCOREABLE on panel (no markers):",unscore)
# ================= FIGURE: UMAP (state) with cluster numbers + dotplot =================
fig=plt.figure(figsize=(17,7.5)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.05])
axU=fig.add_subplot(gs[0,0])
for s in present:
    m=A.obs.state.values==s
    axU.scatter(U[m,0],U[m,1],s=4,c=SCOL[s],linewidths=0,alpha=0.85,rasterized=True,label=f"{s} ({100*m.mean():.0f}%)")
for cl in clusters:
    pts=U[A.obs.leiden.values==cl]
    if len(pts)<10: continue
    c=np.median(pts,0); d=np.linalg.norm(pts-c,axis=1); core=pts[d<=np.percentile(d,60)]; c=core.mean(0)
    axU.text(c[0],c[1],cl,fontsize=12,fontweight="bold",ha="center",va="center",color="white",
             path_effects=[pe.withStroke(linewidth=3.0,foreground="#2b2b2b")],zorder=10)
x0,x1=U[:,0].min(),U[:,0].max(); y0,y1=U[:,1].min(),U[:,1].max(); al=0.16*(x1-x0); ox=x0-0.02*(x1-x0); oy=y0-0.02*(y1-y0)
axU.annotate("",xy=(ox+al,oy),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
axU.annotate("",xy=(ox,oy+al),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
axU.text(ox+al*.5,oy-0.03*(y1-y0),"UMAP1",fontsize=8,ha="center",va="top"); axU.text(ox-0.03*(x1-x0),oy+al*.5,"UMAP2",fontsize=8,ha="right",va="center",rotation=90)
axU.set_xticks([]); axU.set_yticks([]); axU.set_aspect("equal")
for sp in axU.spines.values(): sp.set_visible(False)
axU.set_title("Microglia UMAP (our Leiden clustering) — annotated to Tuddenham 5 main states",fontsize=11,fontweight="bold")
axU.legend(markerscale=3,fontsize=9,loc="upper right",frameon=False)
# dotplot: top markers per scoreable state x cluster (unique genes)
axD=fig.add_subplot(gs[0,1])
MARK=[]; glab=[]; used=set()
for s in SCOREORD:
    for gn in SIG[s][:5]:
        if gn not in used: MARK.append((s,gn)); glab.append(gn); used.add(gn)
clord=[c for c in clusters]; genes=[g for _,g in MARK]
E=np.asarray(A[:,genes].X.todense())
emat=pd.DataFrame(E,columns=genes); clv=A.obs.leiden.values
mean_e=emat.groupby(clv).mean().reindex(clord); pct_e=(emat>0).groupby(clv).mean().reindex(clord)
mn=(mean_e-mean_e.min())/(mean_e.max()-mean_e.min()+1e-9)   # per-gene 0-1 scale
for gi,(s,gn) in enumerate(MARK):
    for ci,cl in enumerate(clord):
        axD.scatter(ci,gi,s=10+float(pct_e.loc[cl,gn])*180,c=[plt.cm.Reds(float(mn.loc[cl,gn]))],edgecolors="#999",linewidths=0.3)
axD.set_xticks(range(len(clord))); axD.set_xticklabels(clord,fontsize=8); axD.set_xlabel("Leiden cluster")
axD.set_yticks(range(len(MARK))); axD.set_yticklabels(glab,fontsize=8)
# state bracket colors on y axis
for gi,(s,gn) in enumerate(MARK): axD.add_patch(plt.Rectangle((-1.4,gi-0.5),0.5,1,color=SCOL[s],clip_on=False))
axD.set_xlim(-1.5,len(clord)-0.5); axD.set_ylim(len(MARK)-0.5,-0.5)
axD.set_title("State marker genes across clusters (size=%expr, color=scaled mean)",fontsize=10.5,fontweight="bold")
for sp in axD.spines.values(): sp.set_visible(False)
import matplotlib.lines as ml
handles=[ml.Line2D([0],[0],marker="o",linestyle="",markersize=8,markerfacecolor=SCOL[s],markeredgecolor="#999",label=s) for s in SCOREORD]
axD.legend(handles=handles,fontsize=8,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False,title="state")
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_5states.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_5states.png + microglia_5state_cluster_assignment.csv + microglia_5state_coords.csv")
