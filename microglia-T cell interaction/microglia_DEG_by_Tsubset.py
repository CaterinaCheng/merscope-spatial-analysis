"""
325_microglia_DEG_by_Tsubset.py
Finer than 323: microglia near each individual T SUBSET vs baseline microglia (no T/NK <=30um).
Keep only subsets whose microglia-near group has >= MIN_MIC cells (exclude very few).
DEG = Wilcoxon log-norm; T-spillover excluded; bar panels (style of niche_DEG_Tmic.png).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
H5=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"<MERSCOPE_ROOT>\QC data")
R=30.0; MIN_MIC=40
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
ALLT=SUBS
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
nearsub={s:np.zeros(len(idx)) for s in SUBS}; anyT=np.zeros(len(idx))
for r in np.unique(run):
    ms=np.where(is_mic&(run==r)&hasxy)[0]
    if not len(ms): continue
    mxy=np.column_stack([mx[ms],my[ms]])
    for s in SUBS:
        ss=np.where((labv==s)&(run==r)&hasxy)[0]
        if len(ss):
            d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy,k=1); nearsub[s][ms]=(d<=R).astype(int)
    for s in SUBS: anyT[ms]+=nearsub[s][ms]
base=is_mic&hasxy&(anyT==0)
print(f"baseline microglia (no T/NK <=30um): {int(base.sum())}")
keep=[]
for s in SUBS:
    nmic=int((is_mic&hasxy&(nearsub[s]>=1)).sum())
    tag="KEEP" if nmic>=MIN_MIC else "drop (<%d)"%MIN_MIC
    print(f"  microglia near {s:14}: {nmic:4d}  {tag}")
    if nmic>=MIN_MIC: keep.append(s)

def deg(nearm,farm):
    sel=np.where(nearm|farm)[0]; a=ad.AnnData(X=X[sel].copy(),var=pd.DataFrame(index=var)); a.obs["g"]=np.where(nearm[sel],"near","far")
    sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a); sc.tl.rank_genes_groups(a,"g",groups=["near"],reference="far",method="wilcoxon")
    r=sc.get.rank_genes_groups_df(a,group="near").rename(columns={"names":"gene","logfoldchanges":"log2FC","pvals":"pval","pvals_adj":"padj"})
    r=r[~r.gene.str.startswith("Blank")].copy(); r["spillover"]=r.gene.isin(T_SPILL); return r
res={}
for s in keep:
    near=is_mic&hasxy&(nearsub[s]>=1); res[s]=deg(near,base)
    res[s].to_csv(NEW/("DEG_microglia_near_%s.csv"%s.replace(" ","_").replace("/","")),index=False)
    sig=res[s][(~res[s].spillover)&(res[s].padj<0.05)]
    up=sig[sig.log2FC>0].nsmallest(8,"pval").gene.tolist(); dn=sig[sig.log2FC<0].nsmallest(8,"pval").gene.tolist()
    print(f"\n{s} (n_near={int((is_mic&hasxy&(nearsub[s]>=1)).sum())}, intrinsic sig={len(sig)}): UP {up}  DOWN {dn}")

ncol=len(keep); fig,axes=plt.subplots(1,ncol,figsize=(6.3*ncol,6.5)); axes=np.atleast_1d(axes)
nmicd={s:int((is_mic&hasxy&(nearsub[s]>=1)).sum()) for s in keep}
for ax,s in zip(axes,keep):
    r=res[s]; intr=r[~r.spillover]
    up=intr[intr.log2FC>0].nsmallest(8,"pval"); dn=intr[intr.log2FC<0].nsmallest(8,"pval")
    d=pd.concat([dn,up]).drop_duplicates("gene").sort_values("log2FC"); y=np.arange(len(d))
    cols=[(0.78,0.24,0.20,1 if p<0.05 else .35) if v>0 else (0.12,0.47,0.71,1 if p<0.05 else .35) for v,p in zip(d.log2FC,d.padj)]
    ax.barh(y,d.log2FC,color=cols,edgecolor="#333",lw=0.3)
    for yi,(_,rr) in zip(y,d.iterrows()):
        ax.text(rr.log2FC+(0.04 if rr.log2FC>0 else -0.04),yi,rr.gene+(" *" if rr.padj<0.05 else ""),va="center",ha="left" if rr.log2FC>0 else "right",fontsize=8)
    ax.axvline(0,color="#333",lw=0.7); ax.set_yticks([]); mm=max(abs(d.log2FC).max(),0.5); ax.set_xlim(-mm*1.9,mm*1.9)
    ax.set_xlabel("log2FC (near vs outside niche)")
    nsig=int(((intr.padj<0.05)).sum())
    ax.set_title(f"Microglia near {s}\nvs outside T/NK niche\n(n_near={nmicd[s]}; {nsig} intrinsic padj<0.05)",fontsize=9.5,fontweight="bold")
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"microglia_DEG_by_Tsubset_bars.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nkept subsets:",keep,"\nSaved: microglia_DEG_by_Tsubset_bars.png")
