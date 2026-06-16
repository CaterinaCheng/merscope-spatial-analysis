"""
351_resolve_mixed.py
Resolve the 'Mixed/low' microglia by k-NN label propagation in scHPF FACTOR space:
each Mixed cell takes the majority 5-state label of its k nearest CONFIDENTLY-ASSIGNED
neighbors (theta = scHPF cell scores). Regenerates the 5-state scHPF UMAP with no Mixed.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import matplotlib.patheffects as pe, matplotlib.lines as ml
from scipy.sparse import csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
KNN=30
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
co=pd.read_csv(NEW/"microglia_schpf_5state_coords.csv",index_col=0)
theta=pd.read_csv(NEW/"microglia_schpf_cell_scores.csv",index_col=0).reindex(co.index)
# L1-normalize theta per cell (scHPF cell scores are rates) for a fair distance
Tn=normalize(theta.values,norm="l1",axis=1)
assigned=co.state.values!="Mixed/low"; mixed=~assigned
print(f"assigned={assigned.sum()}  mixed={mixed.sum()} ({100*mixed.mean():.1f}%)")
clf=KNeighborsClassifier(n_neighbors=KNN,weights="distance").fit(Tn[assigned],co.state.values[assigned])
pred=clf.predict(Tn[mixed])
new=co.state.values.copy(); new[mixed]=pred; co["state_resolved"]=new
print("\nMixed redistributed to:")
vc=pd.Series(pred).value_counts()
for k,v in vc.items(): print(f"  {k:18}: {v:6d} ({100*v/len(pred):.1f}% of mixed)")
print("\nFINAL 5-state composition (mixed resolved):")
for s in STATEORD: print(f"  {s:18}: {100*(co.state_resolved==s).mean():4.1f}%")
co.to_csv(NEW/"microglia_schpf_5state_resolved_coords.csv")
# per-cluster resolved majority (for labels)
cl_state={cl:pd.Series(co.state_resolved.values[co.leiden.values==cl]).value_counts().index[0] for cl in sorted(co.leiden.unique())}
# ---- figure: UMAP + Green dotplot (reload expression) ----
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
micidx=np.where(v2=="Mic")[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var)); A=A[co.index].copy()
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
g6=pd.read_csv(NEW/"green_mic_state_signatures.csv")
def colg(key): return [x for x in g6[[c for c in g6.columns if key in c][0]].dropna()]
SIG={"Homeostatic":colg("Mic.2"),"MHC-II/APC":colg("Mic.9"),"DAM":sorted(set(colg("Mic.12"))|set(colg("Mic.13"))),"Phagocytic":colg("Mic.7"),"Inflammatory/IEG":colg("Mic.15")}
U=co[["umap1","umap2"]].values; clusters=sorted(co.leiden.unique())
fig=plt.figure(figsize=(17,7.5)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.05])
axU=fig.add_subplot(gs[0,0])
for s in STATEORD:
    m=co.state_resolved.values==s
    axU.scatter(U[m,0],U[m,1],s=4,c=SCOL[s],linewidths=0,alpha=0.85,rasterized=True,label=f"{s} ({100*m.mean():.0f}%)")
for cl in clusters:
    pts=U[co.leiden.values==cl]
    if len(pts)<10: continue
    c=np.median(pts,0); d=np.linalg.norm(pts-c,axis=1); core=pts[d<=np.percentile(d,60)]; c=core.mean(0)
    axU.text(c[0],c[1],str(cl),fontsize=11,fontweight="bold",ha="center",va="center",color="white",path_effects=[pe.withStroke(linewidth=3.0,foreground="#2b2b2b")],zorder=10)
x0,x1=U[:,0].min(),U[:,0].max(); y0,y1=U[:,1].min(),U[:,1].max(); al=0.16*(x1-x0); ox=x0-0.02*(x1-x0); oy=y0-0.02*(y1-y0)
axU.annotate("",xy=(ox+al,oy),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5)); axU.annotate("",xy=(ox,oy+al),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.5))
axU.text(ox+al*.5,oy-0.03*(y1-y0),"UMAP1",fontsize=8,ha="center",va="top"); axU.text(ox-0.03*(x1-x0),oy+al*.5,"UMAP2",fontsize=8,ha="right",va="center",rotation=90)
axU.set_xticks([]); axU.set_yticks([]); axU.set_aspect("equal")
for sp in axU.spines.values(): sp.set_visible(False)
axU.set_title("Microglia UMAP — scHPF clustering, Green 5-state (Mixed resolved by kNN)",fontsize=11,fontweight="bold")
axU.legend(markerscale=3,fontsize=9,loc="upper right",frameon=False)
axD=fig.add_subplot(gs[0,1]); MARK=[]; glab=[]; used=set()
for s in STATEORD:
    cnt=0
    for gn in SIG[s]:
        if gn in A.var_names and gn not in used: MARK.append((s,gn)); glab.append(gn); used.add(gn); cnt+=1
        if cnt>=5: break
gl=[g for _,g in MARK]; E=np.asarray(A[:,gl].X.todense())
emat=pd.DataFrame(E,columns=gl); clv=co.leiden.values
mean_e=emat.groupby(clv).mean().reindex(clusters); pct_e=(emat>0).groupby(clv).mean().reindex(clusters)
mn=(mean_e-mean_e.min())/(mean_e.max()-mean_e.min()+1e-9)
for gi,(s,gn) in enumerate(MARK):
    for ci,cl in enumerate(clusters):
        axD.scatter(ci,gi,s=10+float(pct_e.loc[cl,gn])*170,c=[plt.cm.Reds(float(mn.loc[cl,gn]))],edgecolors="#999",linewidths=0.3)
    axD.add_patch(plt.Rectangle((-1.4,gi-0.5),0.5,1,color=SCOL[s],clip_on=False))
axD.set_xticks(range(len(clusters))); axD.set_xticklabels(clusters,fontsize=7.5); axD.set_xlabel("scHPF cluster")
axD.set_yticks(range(len(MARK))); axD.set_yticklabels(glab,fontsize=8); axD.set_xlim(-1.5,len(clusters)-0.5); axD.set_ylim(len(MARK)-0.5,-0.5)
axD.set_title("Green 5-state markers across clusters",fontsize=10,fontweight="bold")
for sp in axD.spines.values(): sp.set_visible(False)
handles=[ml.Line2D([0],[0],marker="o",linestyle="",markersize=8,markerfacecolor=SCOL[s],markeredgecolor="#999",label=s) for s in STATEORD]
axD.legend(handles=handles,fontsize=8,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False,title="state")
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_schpf_5states_resolved.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_schpf_5states_resolved.png + microglia_schpf_5state_resolved_coords.csv")
