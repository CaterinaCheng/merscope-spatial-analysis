"""
349_microglia_schpf_5states.py
Re-annotate the scHPF microglia clusters (from 346) with a 5-STATE Green scheme:
  Homeostatic | MHC-II/APC | DAM (Activated+Lipid merged) | Phagocytic | Inflammatory/IEG
Reuses the saved scHPF UMAP + Leiden clustering (microglia_schpf_green_coords.csv); only
the state signatures and cluster->state assignment change. DAM = union of Mic.12 & Mic.13 markers.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import matplotlib.patheffects as pe, matplotlib.lines as ml
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
g6=pd.read_csv(NEW/"green_mic_state_signatures.csv")
def col(key): return [x for x in g6[[c for c in g6.columns if key in c][0]].dropna()]
SIG={"Homeostatic":col("Mic.2"),"MHC-II/APC":col("Mic.9"),
     "DAM":sorted(set(col("Mic.12"))|set(col("Mic.13"))),"Phagocytic":col("Mic.7"),"Inflammatory/IEG":col("Mic.15")}
STATEORD=list(SIG.keys())
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F","Mixed/low":"#BDC3C7"}
print("5-state signatures:"); [print(f"  {k:18}: {', '.join(v)}") for k,v in SIG.items()]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,gl in SIG.items(): sc.tl.score_genes(A,[x for x in gl if x in A.var_names],score_name=k,ctrl_size=50)
# reuse scHPF clustering + UMAP from 346
co=pd.read_csv(NEW/"microglia_schpf_green_coords.csv",index_col=0).reindex(A.obs_names)
A.obs["leiden"]=co["leiden"].astype(str).values; U=co[["umap1","umap2"]].values
Z=(A.obs[STATEORD]-A.obs[STATEORD].mean())/A.obs[STATEORD].std()
clusters=sorted(A.obs.leiden.unique(),key=int); cl_state={}; rows=[]
for cl in clusters:
    m=A.obs.leiden.values==cl; mz=Z.loc[m].mean().sort_values(ascending=False)
    cl_state[cl]= mz.index[0] if mz.iloc[0]>0.05 else "Mixed/low"
    rows.append(dict(leiden=cl,n=int(m.sum()),state=cl_state[cl],topz=round(mz.iloc[0],3),second=mz.index[1],secz=round(mz.iloc[1],3)))
INFO=pd.DataFrame(rows); print("\nscHPF cluster -> 5-state:"); print(INFO.to_string(index=False))
A.obs["state"]=[cl_state[c] for c in A.obs.leiden.values]
INFO.to_csv(NEW/"microglia_schpf_5state_cluster_assignment.csv",index=False)
pd.DataFrame({"umap1":U[:,0],"umap2":U[:,1],"leiden":A.obs.leiden.values,"state":A.obs.state.values},index=A.obs_names).to_csv(NEW/"microglia_schpf_5state_coords.csv")
present=[s for s in STATEORD if (A.obs.state==s).any()]+(["Mixed/low"] if (A.obs.state=="Mixed/low").any() else [])
print("\n5-state composition:",{s:round(100*(A.obs.state==s).mean(),1) for s in present})
# ===== FIGURE =====
fig=plt.figure(figsize=(17,7.5)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.05])
axU=fig.add_subplot(gs[0,0])
for s in present:
    m=A.obs.state.values==s
    axU.scatter(U[m,0],U[m,1],s=4,c=SCOL[s],linewidths=0,alpha=0.85,rasterized=True,label=f"{s} ({100*m.mean():.0f}%)")
for cl in clusters:
    pts=U[A.obs.leiden.values==cl]
    if len(pts)<10: continue
    c=np.median(pts,0); d=np.linalg.norm(pts-c,axis=1); core=pts[d<=np.percentile(d,60)]; c=core.mean(0)
    axU.text(c[0],c[1],cl,fontsize=11,fontweight="bold",ha="center",va="center",color="white",path_effects=[pe.withStroke(linewidth=3.0,foreground="#2b2b2b")],zorder=10)
x0,x1=U[:,0].min(),U[:,0].max(); y0,y1=U[:,1].min(),U[:,1].max(); al=0.16*(x1-x0); ox=x0-0.02*(x1-x0); oy=y0-0.02*(y1-y0)
axU.annotate("",xy=(ox+al,oy),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
axU.annotate("",xy=(ox,oy+al),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
axU.text(ox+al*.5,oy-0.03*(y1-y0),"UMAP1",fontsize=8,ha="center",va="top"); axU.text(ox-0.03*(x1-x0),oy+al*.5,"UMAP2",fontsize=8,ha="right",va="center",rotation=90)
axU.set_xticks([]); axU.set_yticks([]); axU.set_aspect("equal")
for sp in axU.spines.values(): sp.set_visible(False)
axU.set_title("Microglia UMAP — scHPF clustering, Green 5-state annotation",fontsize=11,fontweight="bold")
axU.legend(markerscale=3,fontsize=9,loc="upper right",frameon=False)
# dotplot
axD=fig.add_subplot(gs[0,1]); MARK=[]; glab=[]; used=set()
for s in STATEORD:
    for gn in SIG[s]:
        if gn in A.var_names and gn not in used: MARK.append((s,gn)); glab.append(gn); used.add(gn)
        if sum(1 for ss,_ in MARK if ss==s)>=5: break
clord=list(clusters); gl=[g for _,g in MARK]; E=np.asarray(A[:,gl].X.todense())
emat=pd.DataFrame(E,columns=gl); clv=A.obs.leiden.values
mean_e=emat.groupby(clv).mean().reindex(clord); pct_e=(emat>0).groupby(clv).mean().reindex(clord)
mn=(mean_e-mean_e.min())/(mean_e.max()-mean_e.min()+1e-9)
for gi,(s,gn) in enumerate(MARK):
    for ci,cl in enumerate(clord):
        axD.scatter(ci,gi,s=10+float(pct_e.loc[cl,gn])*170,c=[plt.cm.Reds(float(mn.loc[cl,gn]))],edgecolors="#999",linewidths=0.3)
    axD.add_patch(plt.Rectangle((-1.4,gi-0.5),0.5,1,color=SCOL[s],clip_on=False))
axD.set_xticks(range(len(clord))); axD.set_xticklabels(clord,fontsize=7.5); axD.set_xlabel("scHPF Leiden cluster")
axD.set_yticks(range(len(MARK))); axD.set_yticklabels(glab,fontsize=8)
axD.set_xlim(-1.5,len(clord)-0.5); axD.set_ylim(len(MARK)-0.5,-0.5)
axD.set_title("Green 5-state markers across clusters (size=%expr, color=scaled mean)",fontsize=10,fontweight="bold")
for sp in axD.spines.values(): sp.set_visible(False)
handles=[ml.Line2D([0],[0],marker="o",linestyle="",markersize=8,markerfacecolor=SCOL[s],markeredgecolor="#999",label=s) for s in STATEORD]
axD.legend(handles=handles,fontsize=8,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False,title="state")
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_schpf_5states.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_schpf_5states.png + microglia_schpf_5state_{cluster_assignment,coords}.csv")
