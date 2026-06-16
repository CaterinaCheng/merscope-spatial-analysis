"""
352_cluster_qc_biology.py
TRUE BIOLOGY + QC of the scHPF microglia clusters. Per cluster:
 - depth (raw total counts) & complexity (n_genes)
 - microglia-core identity score + cross-lineage CONTAMINATION signatures (astro/oligo/OPC/neuron/vascular/lymphoid)
 - top one-vs-rest DE genes (what the cluster really expresses)
Outputs heatmap(cluster x signature) + QC bars, and prints a per-cluster DE table.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
SIGS={
 "Microglia-core":["CX3CR1","CSF1R","C1QA","C1QB","C1QC","AIF1","TYROBP","P2RY12","P2RY13","TMEM119","CTSS","FCER1G"],
 "Astrocyte":["AQP4","GJA1","SLC1A3","SLC1A2","GLUL","GFAP","AQP9"],
 "Oligo":["MOG","MAL","PLP1","MOBP","CNP","MBP","UGT8"],
 "OPC":["PDGFRA","OLIG2","VCAN"],
 "Neuron":["SNAP25","SYT1","RBFOX3","GAD1","MEG3","NRGN","SLC17A7"],
 "Vascular":["PECAM1","CLDN5","VWF","PDGFRB","ACTA2","RGS5","COL1A1"],
 "Lymphoid":["CD3D","CD3E","IL7R","CD8A","CXCR6","NKG7","MS4A1","SKAP1"],
}
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
co=pd.read_csv(NEW/"microglia_schpf_5state_resolved_coords.csv",index_col=0)
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    raw=f["layers/counts"]; Xr=csr_matrix((raw["data"][:],raw["indices"][:],raw["indptr"][:]),shape=tuple(int(s) for s in raw.attrs["shape"])).astype(np.float32)
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]; mid=idx[micidx]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=mid),var=pd.DataFrame(index=var)); A=A[co.index].copy()
tot=np.asarray(Xr[micidx].sum(1)).ravel(); ng=np.asarray((Xr[micidx]>0).sum(1)).ravel()
QC=pd.DataFrame({"tot":tot,"ng":ng},index=mid).reindex(co.index)
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,gl in SIGS.items(): sc.tl.score_genes(A,[x for x in gl if x in A.var_names],score_name=k,ctrl_size=50)
A.obs["leiden"]=co.leiden.astype(str).values; A.obs["state"]=co.state_resolved.values
clusters=sorted(co.leiden.unique())
# per-cluster summary
SIGN=list(SIGS.keys()); rows=[]
for cl in clusters:
    m=A.obs.leiden.values==str(cl); st=pd.Series(A.obs.state.values[m]).value_counts().index[0]
    d=dict(cl=cl,n=int(m.sum()),med_counts=int(np.median(QC.tot.values[m])),med_genes=int(np.median(QC.ng.values[m])),state=st)
    for s in SIGN: d[s]=A.obs[s].values[m].mean()
    rows.append(d)
T=pd.DataFrame(rows)
order=sorted(range(len(T)),key=lambda i:(list(SCOL).index(T.state[i]) if T.state[i] in SCOL else 9, -T["Microglia-core"][i]))
T=T.iloc[order].reset_index(drop=True)
T.to_csv(NEW/"microglia_cluster_qc.csv",index=False)
# DE one-vs-rest
sc.tl.rank_genes_groups(A,"leiden",method="wilcoxon",n_genes=10)
print("=== per-cluster QC + true top DE genes (decontam, one-vs-rest Wilcoxon) ===")
print(f"{'cl':>3} {'n':>5} {'cnts':>5} {'gns':>4} {'state':18} | top DE genes")
deg={}
for cl in T.cl:
    names=[A.uns['rank_genes_groups']['names'][str(cl)][i] for i in range(10)]; deg[cl]=names
    r=T[T.cl==cl].iloc[0]
    print(f"{cl:>3} {r.n:>5} {r.med_counts:>5} {r.med_genes:>4} {r.state:18} | {', '.join(names)}")
# ================= FIGURE =================
fig=plt.figure(figsize=(15,10)); gsf=fig.add_gridspec(1,3,width_ratios=[0.25,1.0,0.6],wspace=0.05)
ncl=len(T); yt=np.arange(ncl)
# state color strip
axs=fig.add_subplot(gsf[0,0])
for i,st in enumerate(T.state): axs.add_patch(plt.Rectangle((0,i-0.5),1,1,color=SCOL.get(st,"#ccc")))
axs.set_xlim(0,1); axs.set_ylim(ncl-0.5,-0.5); axs.set_xticks([]); axs.set_yticks(yt); axs.set_yticklabels([f"cl{c}" for c in T.cl],fontsize=8)
axs.set_title("state",fontsize=9);
for sp in axs.spines.values(): sp.set_visible(False)
# signature heatmap (z across clusters)
axh=fig.add_subplot(gsf[0,1]); H=T[SIGN].values; Hz=(H-H.mean(0))/(H.std(0)+1e-9)
im=axh.imshow(Hz,cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
axh.set_xticks(range(len(SIGN))); axh.set_xticklabels(SIGN,rotation=35,ha="right"); axh.set_yticks(yt); axh.set_yticklabels([])
for i in range(ncl):
    for j in range(len(SIGN)): axh.text(j,i,f"{H[i,j]:.2f}",ha="center",va="center",fontsize=6.5,color="white" if abs(Hz[i,j])>1.3 else "#333")
axh.set_title("identity / contamination signature (number=raw score; color=z across clusters)",fontsize=9.5,fontweight="bold")
fig.colorbar(im,ax=axh,shrink=0.5,label="z")
# QC bars
axq=fig.add_subplot(gsf[0,2]); axq.barh(yt-0.2,T.med_counts,height=0.4,color="#555",label="median counts")
axq2=axq.twiny(); axq2.barh(yt+0.2,T.med_genes,height=0.4,color="#E67E22",label="median genes")
axq.set_ylim(ncl-0.5,-0.5); axq.set_yticks([]); axq.set_xlabel("median total counts",fontsize=8); axq2.set_xlabel("median n_genes",color="#E67E22",fontsize=8)
axq.set_title("depth / complexity",fontsize=9.5,fontweight="bold")
for sp in axq.spines.values(): sp.set_visible(False)
fig.suptitle("scHPF microglia clusters — QC & true biology",fontsize=13,fontweight="bold",y=0.995)
plt.tight_layout(); fig.savefig(NEW/"microglia_cluster_qc.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_cluster_qc.png + microglia_cluster_qc.csv")
