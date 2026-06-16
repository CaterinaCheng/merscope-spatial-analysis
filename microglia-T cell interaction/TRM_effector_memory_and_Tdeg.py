"""
326_TRM_effector_memory_and_Tdeg.py
Q1: which CD8 TRM is effector-like vs memory-like? Marker comparison (memory/residency vs
    cytotoxic/effector vs exhaustion) across CD8 TRM 1 / TRM 2 / TEMRA.
Q2: IFN expression on T cells per subset; then T-cell DEG INSIDE vs OUTSIDE microglia
    neighbourhood (>=1 microglion <=30um) for CD8 TRM 1 and CD8 TRM 2 (microglial spillover flagged).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0
MEM=["ITGAE","CXCR6","CD69","IL7R","TCF7","SELL","CD27","CD28"]          # residency/memory/quiescence
EFF=["GZMK","GZMB","PRF1","GNLY","NKG7","IFNG","TBX21","KLRG1","FGFBP2","CCL5"]  # cytotoxic/effector
EXH=["PDCD1","CTLA4","TIGIT","LAG3","HAVCR2","TOX","ENTPD1"]              # exhaustion/activation
IFNRESP=["STAT1","IRF1","GBP1","GBP2","ISG15","IFI6","MX1","CXCL9","CXCL10"]
MIC_SPILL=set("C1QA C1QB C1QC CSF1R CX3CR1 TMEM119 AIF1 TYROBP FCER1G CTSS HEXB TREM2 GPNMB APOE ITGAX CD68 SELPLG SPI1 CD74 HLA-DRA HLA-DPA1 HLA-DQB1 HLA-DRB1 CIITA MRC1 CD163 C3 LAPTM5 MOG MAL PLP1 MOBP AQP4 GJA1 SLC1A2 SLC1A3 PLPP3 GSN AHNAK ACTA2 PECAM1 PDGFRB RNASE1".split())
CD8SUB=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}; labv=lab.reindex(idx).values
is_mic=(v2=="Mic")
def raw(g): return np.asarray(X[:,vp[g]].todense()).ravel() if g in vp else np.zeros(X.shape[0])
sf=np.asarray(X.sum(1)).ravel(); med=np.median(sf); sfn=sf/med; sfn[sfn==0]=1
def norm(g): return raw(g)/sfn

# ===== Q1: TRM marker comparison =====
print("=== Q1: CD8 TRM marker comparison (mean normalized expr; %pos in parens) ===")
allg=[g for g in MEM+EFF+EXH if g in vp]; miss=[g for g in MEM+EFF+EXH if g not in vp]
if miss: print("  (absent from panel:",miss,")")
rowsM=[]
for s in CD8SUB:
    m=(labv==s); row={"subset":f"{s} (n={int(m.sum())})"}
    for g in allg: row[g]=norm(g)[m].mean()
    rowsM.append(row)
T=pd.DataFrame(rowsM).set_index("subset"); print(T[allg].round(2).T.to_string())
# dotplot: rows=genes grouped, cols=subset; size=%pos, color=mean (col-zscored within gene)
groups=[("memory/residency",[g for g in MEM if g in vp]),("cytotoxic/effector",[g for g in EFF if g in vp]),("exhaustion/activation",[g for g in EXH if g in vp])]
order=[g for _,gs in groups for g in gs]
mean=np.array([[norm(g)[labv==s].mean() for s in CD8SUB] for g in order])
pct=np.array([[100*(raw(g)[labv==s]>0).mean() for s in CD8SUB] for g in order])
z=(mean-mean.mean(1,keepdims=True))/(mean.std(1,keepdims=True)+1e-9)
fig,ax=plt.subplots(figsize=(5.2,0.34*len(order)+2))
for i,g in enumerate(order):
    for j in range(len(CD8SUB)):
        ax.scatter(j,len(order)-1-i,s=max(pct[i,j],1)*7,c=[z[i,j]],cmap="RdBu_r",vmin=-1.3,vmax=1.3,edgecolors="#555",linewidths=0.3)
ax.set_xticks(range(len(CD8SUB))); ax.set_xticklabels(CD8SUB,rotation=20,ha="right",fontsize=8)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1],fontsize=8); ax.set_xlim(-0.6,len(CD8SUB)-0.4)
yb=len(order);
for gname,gs in groups:
    if gs:
        top=yb-1; bot=yb-len(gs); ax.text(len(CD8SUB)-0.3,(top+bot)/2,gname,rotation=270,va="center",ha="left",fontsize=7.5,color="#444"); yb-=len(gs)
fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r",norm=plt.Normalize(-1.3,1.3)),ax=ax,shrink=0.4,label="row z (mean norm expr)")
for pv in [10,40,80]: ax.scatter([],[],s=pv*7,c="#bbb",edgecolors="#555",label=f"{pv}%")
ax.legend(title="% positive",loc="upper left",bbox_to_anchor=(1.25,1.0),fontsize=7,labelspacing=1.1,frameon=False)
ax.set_title("CD8 TRM 1 vs TRM 2 vs TEMRA: memory vs effector markers",fontsize=9.5,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"CD8_TRM_effector_vs_memory.png",dpi=140,bbox_inches="tight"); plt.close()

# ===== Q2a: IFN on T cells =====
print("\n=== Q2a: IFN / IFN-response on T cells per subset (%pos | mean norm) ===")
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
ifg=["IFNG"]+[g for g in IFNRESP if g in vp]; print("  IFN genes on panel:",ifg)
for s in SUBS:
    m=(labv==s); print(f"  {s:13}: "+" | ".join(f"{g} {100*(raw(g)[m]>0).mean():.0f}%/{norm(g)[m].mean():.2f}" for g in ifg))

# ===== Q2b: T-cell DEG inside vs outside microglia neighbourhood, per CD8 TRM subset =====
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); nmic=np.zeros(len(idx))
isT=np.isin(labv,SUBS)
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]; ts=np.where(isT&(run==r)&hasxy)[0]
    if len(ms) and len(ts):
        tree=cKDTree(np.column_stack([mx[ms],my[ms]]))
        for k,h in enumerate(tree.query_ball_point(np.column_stack([mx[ts],my[ts]]),r=R)): nmic[ts[k]]=len(h)
def deg(nearm,farm):
    sel=np.where(nearm|farm)[0]; a=ad.AnnData(X=X[sel].copy(),var=pd.DataFrame(index=var)); a.obs["g"]=np.where(nearm[sel],"near","far")
    sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a); sc.tl.rank_genes_groups(a,"g",groups=["near"],reference="far",method="wilcoxon")
    r=sc.get.rank_genes_groups_df(a,group="near").rename(columns={"names":"gene","logfoldchanges":"log2FC","pvals":"pval","pvals_adj":"padj"})
    r=r[~r.gene.str.startswith("Blank")].copy(); r["spillover"]=r.gene.isin(MIC_SPILL); return r
TARG=["CD8 TRM 1","CD8 TRM 2"]
print("\n=== Q2b: T-cell DEG near vs far from microglia, per subset ===")
res={}
for s in TARG:
    m=(labv==s)&hasxy; near=m&(nmic>=1); far=m&(nmic==0)
    print(f"  {s}: near={int(near.sum())} far={int(far.sum())}")
    res[s]=deg(near,far); res[s].to_csv(NEW/("DEG_Tcell_%s_near_microglia.csv"%s.replace(" ","_")),index=False)
    sig=res[s][(~res[s].spillover)&(res[s].padj<0.05)]
    print(f"     intrinsic sig={len(sig)}: UP {sig[sig.log2FC>0].nsmallest(8,'pval').gene.tolist()}  DOWN {sig[sig.log2FC<0].nsmallest(8,'pval').gene.tolist()}")
fig,axes=plt.subplots(1,len(TARG),figsize=(7*len(TARG),6)); axes=np.atleast_1d(axes)
for ax,s in zip(axes,TARG):
    r=res[s]; intr=r[~r.spillover]
    up=intr[intr.log2FC>0].nsmallest(8,"pval"); dn=intr[intr.log2FC<0].nsmallest(8,"pval")
    d=pd.concat([dn,up]).drop_duplicates("gene").sort_values("log2FC"); y=np.arange(len(d))
    cols=[(0.78,0.24,0.20,1 if p<0.05 else .35) if v>0 else (0.12,0.47,0.71,1 if p<0.05 else .35) for v,p in zip(d.log2FC,d.padj)]
    ax.barh(y,d.log2FC,color=cols,edgecolor="#333",lw=0.3)
    for yi,(_,rr) in zip(y,d.iterrows()): ax.text(rr.log2FC+(0.04 if rr.log2FC>0 else -0.04),yi,rr.gene+(" *" if rr.padj<0.05 else ""),va="center",ha="left" if rr.log2FC>0 else "right",fontsize=8)
    nn=int(((labv==s)&hasxy&(nmic>=1)).sum()); ff=int(((labv==s)&hasxy&(nmic==0)).sum())
    nsig=int((intr.padj<0.05).sum())
    ax.axvline(0,color="#333",lw=0.7); ax.set_yticks([]); mm=max(abs(d.log2FC).max(),0.5); ax.set_xlim(-mm*1.9,mm*1.9)
    ax.set_xlabel("log2FC (near vs far microglia)"); ax.set_title(f"{s} T cells: near vs far from microglia\n(near={nn} far={ff}; {nsig} intrinsic padj<0.05)",fontsize=9.5,fontweight="bold")
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"Tsubset_DEG_near_microglia.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: CD8_TRM_effector_vs_memory.png + Tsubset_DEG_near_microglia.png + DEG csvs")
