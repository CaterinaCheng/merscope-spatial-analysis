"""
322_niche_DEG.py
DEGs of (1) T cells NEAR microglia (>=1 microglion <=30um) vs FAR, and
        (2) microglia in the T-cell neighborhood (>=1 T <=30um) vs not.
Wilcoxon on log-norm; BH across panel; cross-cell SPILLOVER genes flagged (microglial genes
leaking into T; T genes leaking into microglia). 30um = calibrated neighborhood radius.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]
MIC_SPILL=set("C1QA C1QB C1QC CSF1R CX3CR1 TMEM119 AIF1 TYROBP FCER1G CTSS HEXB TREM2 GPNMB APOE ITGAX CD68 SELPLG SPI1 CD74 HLA-DRA HLA-DPA1 HLA-DQB1 MRC1 CD163 C3 LAPTM5 MOG MAL PLP1 MOBP AQP4 GJA1 SLC1A2".split())
T_SPILL=set("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD5 CD6 CD7 CD27 CD28 CCL5 IL7R LCK ZAP70 THEMIS CD247 SKAP1 GZMK GZMA NKG7 CCL4 IL32 CXCR6 ITK LIME1 CD40LG TC2N".split())

lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
Tids=set(lab[lab.isin(SUBS)].index)
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_T=np.array([c in Tids for c in idx]); is_mic=(v2=="Mic")
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
hasxy=np.isfinite(mx)
nmicT=np.zeros(len(idx)); ntT=np.zeros(len(idx))
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]; ts=np.where(is_T&(run==r)&hasxy)[0]
    if len(ms) and len(ts):
        mt=cKDTree(np.column_stack([mx[ms],my[ms]])); tt=cKDTree(np.column_stack([mx[ts],my[ts]]))
        for k,h in enumerate(mt.query_ball_point(np.column_stack([mx[ts],my[ts]]),r=R)): nmicT[ts[k]]=len(h)
        for k,h in enumerate(tt.query_ball_point(np.column_stack([mx[ms],my[ms]]),r=R)): ntT[ms[k]]=len(h)
near_T=is_T&hasxy&(nmicT>=1); far_T=is_T&hasxy&(nmicT==0)
near_M=is_mic&hasxy&(ntT>=1); far_M=is_mic&hasxy&(ntT==0)
print(f"T cells: near microglia={near_T.sum()}  far={far_T.sum()}")
print(f"Microglia: in T-neighborhood={near_M.sum()}  not={far_M.sum()}")

def bh(p):
    p=np.asarray(p,float); o=np.argsort(p); rk=np.empty(len(p),int); rk[o]=np.arange(1,len(p)+1)
    q=np.minimum.accumulate((p*len(p)/rk)[o][::-1])[::-1]; out=np.empty(len(p)); out[o]=q; return np.minimum(out,1)
def deg(nearm,farm,spill):
    sel=np.where(nearm|farm)[0]; a=ad.AnnData(X=X[sel].copy(),var=pd.DataFrame(index=var)); a.obs["g"]=np.where(nearm[sel],"near","far")
    sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a); sc.tl.rank_genes_groups(a,"g",groups=["near"],reference="far",method="wilcoxon")
    r=sc.get.rank_genes_groups_df(a,group="near").rename(columns={"names":"gene","logfoldchanges":"log2FC","pvals":"pval","pvals_adj":"padj"})
    r=r[~r.gene.str.startswith("Blank")]; r["spillover"]=r.gene.isin(spill); return r
rt=deg(near_T,far_T,MIC_SPILL); rt.to_csv(NEW/"DEG_Tcells_near_microglia.csv",index=False)
rm=deg(near_M,far_M,T_SPILL); rm.to_csv(NEW/"DEG_microglia_near_Tcells.csv",index=False)
def report(r,nm,intr_label):
    intr=r[~r.spillover]
    print(f"\n=== {nm} (intrinsic genes, padj<0.05: {int((intr.padj<0.05).sum())}) ===")
    up=intr[(intr.padj<0.05)&(intr.log2FC>0)].nsmallest(10,'pval'); dn=intr[(intr.padj<0.05)&(intr.log2FC<0)].nsmallest(10,'pval')
    print("  UP  :",", ".join(f"{g}(+{fc:.2f})" for g,fc in zip(up.gene,up.log2FC)) or "none")
    print("  DOWN:",", ".join(f"{g}({fc:.2f})" for g,fc in zip(dn.gene,dn.log2FC)) or "none")
    print("  (spillover, excluded):",", ".join(r[r.spillover&(r.padj<0.05)&(r.log2FC>0)].nlargest(8,'log2FC').gene) or "none")
report(rt,"T cells NEAR vs FAR from microglia","T-intrinsic"); report(rm,"Microglia IN vs OUT of T-neighborhood","microglia-intrinsic")

fig,axes=plt.subplots(1,2,figsize=(15,6))
for ax,(r,title) in zip(axes,[(rt,"T cells: near vs far from microglia"),(rm,"Microglia: in vs out of T-neighborhood")]):
    intr=r[~r.spillover]; up=intr[intr.log2FC>0].nsmallest(8,'pval'); dn=intr[intr.log2FC<0].nsmallest(8,'pval')
    d=pd.concat([dn,up]).drop_duplicates('gene').sort_values('log2FC'); y=np.arange(len(d))
    cols=[(0.78,0.24,0.20,1 if p<0.05 else .4) if v>0 else (0.12,0.47,0.71,1 if p<0.05 else .4) for v,p in zip(d.log2FC,d.padj)]
    ax.barh(y,d.log2FC,color=cols,edgecolor="#333",lw=0.3)
    for yi,(_,rr) in zip(y,d.iterrows()): ax.text(rr.log2FC+(0.03 if rr.log2FC>0 else -0.03),yi,rr.gene+(" *" if rr.padj<0.05 else ""),va="center",ha="left" if rr.log2FC>0 else "right",fontsize=8)
    ax.axvline(0,color="#333",lw=0.7); ax.set_yticks([]); mm=max(abs(d.log2FC).max(),0.5); ax.set_xlim(-mm*1.8,mm*1.8)
    ax.set_xlabel("log2FC (near/in vs far/out)"); ax.set_title(f"{title}\n(intrinsic genes; spillover excluded; * padj<0.05)",fontsize=9.5,fontweight="bold")
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"niche_DEG_Tmic.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: niche_DEG_Tmic.png + DEG_Tcells_near_microglia.csv + DEG_microglia_near_Tcells.csv")
