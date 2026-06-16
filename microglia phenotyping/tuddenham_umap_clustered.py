"""
343_tuddenham_umap_clustered.py
Homogeneous (cluster-level) version of the paper-style microglia UMAP.
Instead of per-cell majority hexbins (noisy), we:
  1. Leiden-cluster the decontam microglia (graph-contiguous regions).
  2. Score the 8 merged Tuddenham meta-group signatures per cell.
  3. Assign each Leiden cluster ONE meta-group = argmax of its mean z-scored signature.
  4. DROP low-confidence clusters (weak top score or no clear winner) -> "Unassigned".
  5. Render hex-binned UMAP (Tuddenham Fig2a style); territories are now homogeneous.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore")
sc.settings.verbosity=0; plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); REF=NEW/"reference"
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
TOPN=18; MINGENES=6; LEIDEN_RES=1.2; MINZ=0.12; MINSEP=0.04   # confidence gates
GROUP={2:"Homeostatic",3:"Homeostatic",4:"Homeostatic",1:"Metabolic/Tx",6:"Metabolic/Tx",7:"Stress",
       9:"APOE/Lipid",5:"Motility",8:"Cytokine/IL",10:"APC (HLA/Compl)",13:"APC (HLA/Compl)",11:"DAM/GPNMB",12:"Proliferative"}
GORDER=["Homeostatic","Metabolic/Tx","Stress","APOE/Lipid","Motility","Cytokine/IL","APC (HLA/Compl)","DAM/GPNMB"]
BLOCK=set(("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD28 CD247 IL7R CXCR6 CCL5 LIME1 SKAP1 IL32 LCK THEMIS GZMK GZMA NKG7 GNLY KLRD1 KLRB1 KLRG1 ICOS FOXP3 CD40LG TBX21 CD7 "
            "CD19 MS4A1 CD79A CD79B JCHAIN BANK1 IGHM AQP4 SLC1A3 SLC1A2 GJA1 GLUL GFAP MOG MAL PLP1 MOBP CNP MBP UGT8 GLDN "
            "RBFOX3 SYT1 SNAP25 GAD1 RORB FOXP2 NRGN MEG3 XIST PECAM1 CLDN5 VWF ACTA2 PDGFRB RGS5 NOTCH3 COL1A1 COL3A1 DCN AHNAK RNASE1").split())
up=pd.read_excel(REF/"tuddenham/MOESM4.xlsx",sheet_name="Upregulated genes")
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vset=set(var)
SIG={}
for grp in GORDER:
    cls=[cl for cl,gg in GROUP.items() if gg==grp]; sub=up[up.up_type.isin(cls)].sort_values("sum_logFC",ascending=False)
    genes=[]; seen=set()
    for gn in sub.gene:
        if gn in vset and gn not in BLOCK and gn not in seen: genes.append(gn); seen.add(gn)
        if len(genes)>=TOPN: break
    if len(genes)>=MINGENES: SIG[grp]=genes
STATEORD=list(SIG.keys())
micidx=np.where(v2=="Mic")[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A); A.raw=A
for k,genes in SIG.items(): sc.tl.score_genes(A,[g for g in genes if g in A.var_names],score_name=k,ctrl_size=50)
# embedding + Leiden clustering
det=np.asarray((A.X>0).mean(0)).ravel(); keep=np.array(var)[det>=0.05]
Asc=A[:,keep].copy(); sc.pp.scale(Asc,max_value=10); sc.tl.pca(Asc,n_comps=30); sc.pp.neighbors(Asc,n_neighbors=15,n_pcs=30)
try:
    sc.tl.leiden(Asc,resolution=LEIDEN_RES,key_added="leiden",flavor="igraph",n_iterations=2,directed=False)
except Exception as e:
    print("igraph leiden failed, fallback:",e); sc.tl.leiden(Asc,resolution=LEIDEN_RES,key_added="leiden")
sc.tl.umap(Asc); U=Asc.obsm["X_umap"]; A.obs["leiden"]=Asc.obs["leiden"].values
# z-score signatures across cells, then per-cluster mean
Z=(A.obs[STATEORD]-A.obs[STATEORD].mean())/A.obs[STATEORD].std()
clusters=sorted(A.obs.leiden.unique(),key=int); cl_label={}; rows=[]
for cl in clusters:
    m=A.obs.leiden.values==cl; mz=Z.loc[m].mean().sort_values(ascending=False)
    top,sec=mz.index[0],mz.index[1]; topv,secv=mz.iloc[0],mz.iloc[1]
    conf=(topv>=MINZ) and (topv-secv>=MINSEP)
    cl_label[cl]= top if conf else "Unassigned"
    rows.append(dict(leiden=cl,n=int(m.sum()),assigned=cl_label[cl],top=top,topz=round(topv,3),second=sec,secz=round(secv,3),conf=conf))
INFO=pd.DataFrame(rows); print(INFO.to_string(index=False))
A.obs["subtype"]=[cl_label[c] for c in A.obs.leiden.values]
INFO.to_csv(NEW/"tuddenham_clustered_assignment.csv",index=False)
pd.DataFrame({"umap1":U[:,0],"umap2":U[:,1],"leiden":A.obs.leiden.values,"subtype":A.obs.subtype.values},index=A.obs_names).to_csv(NEW/"microglia_tuddenham_clustered_coords.csv")
present=[g for g in GORDER if (A.obs.subtype==g).any()]
print("\nterritory composition:",{k:round(100*(A.obs.subtype==k).mean(),1) for k in present+['Unassigned'] if (A.obs.subtype==k).any()})
# ===== paper-style hex-binned figure =====
PAL={"Homeostatic":"#4C9F70","Metabolic/Tx":"#8E6FB0","Stress":"#E15759","APOE/Lipid":"#F28E2B",
     "Motility":"#76B7B2","Cytokine/IL":"#E377C2","APC (HLA/Compl)":"#4E79A7","DAM/GPNMB":"#B07A3C","Unassigned":"#DDDDDD"}
ORDER=[g for g in GORDER if (A.obs.subtype==g).any()]+(["Unassigned"] if (A.obs.subtype=="Unassigned").any() else [])
NUM={g:(i+1) for i,g in enumerate([o for o in ORDER if o!="Unassigned"])}
CODE={g:i for i,g in enumerate(ORDER)}; cmap=ListedColormap([PAL[g] for g in ORDER]); norm=BoundaryNorm(np.arange(-0.5,len(ORDER)+0.5,1),len(ORDER))
codes=pd.Series(A.obs.subtype.values).map(CODE).values.astype(float)
gridsize=int(round(np.sqrt(len(A)/50.0)))
def majority(v): v=np.asarray(v).astype(int); return np.bincount(v,minlength=len(ORDER)).argmax()
fig,ax=plt.subplots(figsize=(8.6,8))
ax.hexbin(U[:,0],U[:,1],C=codes,reduce_C_function=majority,gridsize=gridsize,cmap=cmap,norm=norm,mincnt=1,linewidths=0.2,edgecolors="white")
for g,nidx in NUM.items():
    pts=U[A.obs.subtype.values==g]
    if len(pts)<10: continue
    c=np.median(pts,0); d=np.linalg.norm(pts-c,axis=1); core=pts[d<=np.percentile(d,60)]; c=core.mean(0)
    ax.text(c[0],c[1],str(nidx),fontsize=15,fontweight="bold",ha="center",va="center",color="white",path_effects=[pe.withStroke(linewidth=3.0,foreground="#2b2b2b")],zorder=10)
x0,x1=U[:,0].min(),U[:,0].max(); y0,y1=U[:,1].min(),U[:,1].max(); al=0.16*(x1-x0); ox=x0-0.02*(x1-x0); oy=y0-0.02*(y1-y0)
ax.annotate("",xy=(ox+al,oy),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.6))
ax.annotate("",xy=(ox,oy+al),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.6))
ax.text(ox+al*0.5,oy-0.03*(y1-y0),"UMAP1",fontsize=8,ha="center",va="top",color="#333")
ax.text(ox-0.03*(x1-x0),oy+al*0.5,"UMAP2",fontsize=8,ha="right",va="center",color="#333",rotation=90)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_aspect("equal"); ax.set_title("Human brain microglia — Tuddenham 2024 subset mapping",fontsize=12,fontweight="bold",pad=12)
handles=[Line2D([0],[0],marker="h",linestyle="",markersize=11,markerfacecolor=PAL[g],markeredgecolor="white",label=(f"{NUM[g]}  {g}" if g in NUM else f"–  {g}")) for g in ORDER]
leg=ax.legend(handles=handles,fontsize=9.5,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False,handletextpad=0.5,labelspacing=0.7,title="Microglia subset",title_fontsize=10); leg._legend_box.align="left"
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_tuddenham_clustered.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_tuddenham_clustered.png + tuddenham_clustered_assignment.csv + microglia_tuddenham_clustered_coords.csv")
