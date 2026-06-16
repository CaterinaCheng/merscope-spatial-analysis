"""
323_microglia_DEG_by_Tlineage.py
Do microglia adopt DIFFERENT states depending on which T lineage they contact?
For each microglion, flag whether it has a CD8 / CD4 / NK cell within 30um.
DEG (Wilcoxon, log-norm) of microglia near each lineage vs BASELINE microglia (no T of any kind <=30um).
Also direct CD4-near vs CD8-near contrast. T-spillover genes flagged & excluded from intrinsic view.
Output: heatmap (intrinsic genes x {near CD8, near CD4, near NK}, color=log2FC vs baseline, *=padj<0.05).
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
CD8=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA"]; CD4=["CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]; NKL=["NK"]
LINS={"near CD8":CD8,"near CD4":CD4,"near NK":NKL}
T_SPILL=set("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD5 CD6 CD7 CD27 CD28 CCL5 IL7R LCK ZAP70 THEMIS CD247 SKAP1 GZMK GZMA NKG7 GNLY KLRD1 KLRF1 CCL4 IL32 CXCR6 ITK LIME1 CD40LG TC2N PRF1 GZMB".split())
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_mic=(v2=="Mic"); labv=lab.reindex(idx).values
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
# per microglion: count of each lineage within R
cnt={k:np.zeros(len(idx)) for k in LINS}; anyT=np.zeros(len(idx))
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]
    if not len(ms): continue
    mxy=np.column_stack([mx[ms],my[ms]])
    for k,subs in LINS.items():
        sel=np.isin(labv,subs)&(run==r)&hasxy; ss=np.where(sel)[0]
        if len(ss):
            tree=cKDTree(np.column_stack([mx[ss],my[ss]]))
            d,_=tree.query(mxy,k=1); cnt[k][ms]=(d<=R).astype(int)
    for k in LINS: anyT[ms]+=cnt[k][ms]
base=is_mic&hasxy&(anyT==0)  # microglia with no T of any lineage within R
print("baseline microglia (no T <=30um):",int(base.sum()))
for k in LINS: print(f"  microglia {k}: {int((is_mic&hasxy&(cnt[k]>=1)).sum())}")

def bh(p):
    p=np.asarray(p,float); o=np.argsort(p); rk=np.empty(len(p),int); rk[o]=np.arange(1,len(p)+1)
    q=np.minimum.accumulate((p*len(p)/rk)[o][::-1])[::-1]; out=np.empty(len(p)); out[o]=q; return np.minimum(out,1)
def deg(nearm,farm):
    sel=np.where(nearm|farm)[0]; a=ad.AnnData(X=X[sel].copy(),var=pd.DataFrame(index=var)); a.obs["g"]=np.where(nearm[sel],"near","far")
    sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a); sc.tl.rank_genes_groups(a,"g",groups=["near"],reference="far",method="wilcoxon")
    r=sc.get.rank_genes_groups_df(a,group="near").rename(columns={"names":"gene","logfoldchanges":"log2FC","pvals":"pval","pvals_adj":"padj"})
    r=r[~r.gene.str.startswith("Blank")].copy(); r["spillover"]=r.gene.isin(T_SPILL); return r.set_index("gene")
res={}
for k in LINS:
    near=is_mic&hasxy&(cnt[k]>=1)
    res[k]=deg(near,base); res[k].to_csv(NEW/f"DEG_microglia_{k.replace(' ','_')}.csv")
    sig=res[k][(~res[k].spillover)&(res[k].padj<0.05)]
    up=sig[sig.log2FC>0].nsmallest(8,"pval").index.tolist(); dn=sig[sig.log2FC<0].nsmallest(8,"pval").index.tolist()
    print(f"\n{k} vs baseline (intrinsic sig={len(sig)}): UP {up}  DOWN {dn}")
# direct CD4 vs CD8
near_cd4=is_mic&hasxy&(cnt["near CD4"]>=1); near_cd8=is_mic&hasxy&(cnt["near CD8"]>=1)
dc=deg(near_cd4,near_cd8); dc.to_csv(NEW/"DEG_microglia_CD4near_vs_CD8near.csv")
sig=dc[(~dc.spillover)&(dc.padj<0.05)]
print(f"\nmicroglia near-CD4 vs near-CD8 (intrinsic sig={len(sig)}): higher-near-CD4 {sig[sig.log2FC>0].nsmallest(8,'pval').index.tolist()}  higher-near-CD8 {sig[sig.log2FC<0].nsmallest(8,'pval').index.tolist()}")

# heatmap: union of top intrinsic genes across the 3 vs-baseline contrasts
genes=[]
for k in LINS:
    s=res[k][(~res[k].spillover)&(res[k].padj<0.05)].reindex(res[k].index)
    s=res[k][(~res[k].spillover)&(res[k].padj<0.05)]
    genes+=s.reindex(s.log2FC.abs().sort_values(ascending=False).index).index[:8].tolist()
genes=list(dict.fromkeys(genes))
M=pd.DataFrame({k:res[k].reindex(genes).log2FC for k in LINS})
P=pd.DataFrame({k:res[k].reindex(genes).padj for k in LINS})
M=M.reindex(M.mean(axis=1).sort_values().index); P=P.reindex(M.index)
fig,ax=plt.subplots(figsize=(5.5,max(4,0.32*len(M)+1)))
norm=TwoSlopeNorm(vmin=min(-0.5,M.min().min()),vcenter=0,vmax=max(0.5,M.max().max()))
im=ax.imshow(M.values,cmap="RdBu_r",norm=norm,aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(M.columns,rotation=20,ha="right"); ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index,fontsize=8)
for i in range(len(M)):
    for j in range(3):
        if P.values[i,j]<0.05: ax.text(j,i,"*",ha="center",va="center",fontsize=11,fontweight="bold")
fig.colorbar(im,ax=ax,shrink=0.5,label="log2FC vs baseline microglia")
ax.set_title("Microglial state by neighbouring T lineage\n(log2FC vs microglia with no T <=30µm; * padj<0.05; spillover excluded)",fontsize=9,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"microglia_DEG_by_Tlineage.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_DEG_by_Tlineage.png + DEG_microglia_near_{CD8,CD4,NK}.csv + CD4near_vs_CD8near.csv")
