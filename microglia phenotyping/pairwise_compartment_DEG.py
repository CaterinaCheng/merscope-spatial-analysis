"""
366_pairwise_compartment_DEG.py
Pairwise compartment DEG of clean microglia (like the earlier 333 figure):
 (left)  perivascular vs parenchymal
 (right) vessel-adjacent vs parenchymal
Cross-lineage STRUCTURAL spillover (vascular/astro/oligo/OPC/neuron) EXCLUDED -> intrinsic
microglial/immune genes only. red = up in peri/adj, blue = up in parenchymal. * padj<0.05.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
DEC=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
UP_RED="#C0392B"; DN_BLUE="#2471A3"; NUP,NDN=9,9
# exclude only canonical NON-immune structural markers (vascular/astro/oligo/OPC/neuron)
EXCLUDE=set(("PECAM1 CLDN5 VWF PDGFRB ACTA2 RGS5 NOTCH3 COL1A1 COL3A1 COL4A3 COL9A3 FN1 A2M FLT1 DCN IFITM3 "
             "AQP4 GJA1 SLC1A3 SLC1A2 GLUL GFAP AQP9 VCAN PLPP3 "
             "MOG MAL PLP1 MOBP CNP MBP UGT8 OLIG2 CD22 S1PR5 GLDN "
             "PDGFRA "
             "RBFOX3 SYT1 SNAP25 GAD1 MEG3 NRGN SLC17A7 NELL2 BCL11B CNR1 MEG3 KCNMA1").split())
co=pd.read_csv(NEW/"microglia_final_coords.csv",index_col=0); co=co[~co.cluster_flag]
spc=pd.read_csv(NEW/"clean_microglia_spatial.csv",index_col=0)[["comp"]]; co=co.join(spc,how="inner"); co=co[co.comp!="n/a"]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; rowi=np.array([pos[c] for c in co.index])
A=ad.AnnData(X=Xd[rowi].copy(),obs=pd.DataFrame({"comp":co.comp.values},index=co.index),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
def pair_deg(g1):
    sub=A[A.obs.comp.isin([g1,"parenchymal"])].copy()
    sc.tl.rank_genes_groups(sub,"comp",groups=[g1],reference="parenchymal",method="wilcoxon")
    r=sc.get.rank_genes_groups_df(sub,group=g1)
    r=r[~r.names.isin(EXCLUDE)]; r=r[r.pvals_adj<0.05]
    up=r[r.logfoldchanges>0].sort_values("logfoldchanges",ascending=False).head(NUP)
    dn=r[r.logfoldchanges<0].sort_values("logfoldchanges").head(NDN)
    return pd.concat([up,dn]), r
DEGp,_=pair_deg("perivascular"); DEGa,_=pair_deg("vessel-adjacent")
DEGp.assign(cmp="peri_vs_paren").to_csv(NEW/"pairwise_DEG_peri_vs_paren.csv",index=False)
DEGa.assign(cmp="adj_vs_paren").to_csv(NEW/"pairwise_DEG_adj_vs_paren.csv",index=False)
def panel(ax,db,title):
    db=db.sort_values("logfoldchanges")
    y=np.arange(len(db))
    for yi,(_,r) in zip(y,db.iterrows()):
        col=UP_RED if r.logfoldchanges>0 else DN_BLUE
        ax.barh(yi,r.logfoldchanges,color=col)
        ha="left" if r.logfoldchanges>0 else "right"; off=0.05 if r.logfoldchanges>0 else -0.05
        ax.text(r.logfoldchanges+off,yi,r.names+" *",va="center",ha=ha,fontsize=8)
    ax.axvline(0,color="k",lw=0.9); ax.set_yticks([]); ax.set_xlabel("log2FC")
    mx=np.abs(db.logfoldchanges).max()*1.5; ax.set_xlim(-mx,mx)
    ax.set_title(title,fontsize=10.5,fontweight="bold")
    for sp in ax.spines.values(): sp.set_visible(False)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,6))
panel(ax1,DEGp,"Perivascular vs Parenchymal microglia\n(intrinsic; structural spillover excluded; *padj<0.05)")
panel(ax2,DEGa,"Vessel-adjacent vs Parenchymal microglia\n(intrinsic; structural spillover excluded; *padj<0.05)")
plt.tight_layout(); fig.savefig(NEW/"pairwise_compartment_DEG.png",dpi=200,bbox_inches="tight"); plt.close()
print("peri vs paren: up=",", ".join(DEGp[DEGp.logfoldchanges>0].names)," | down=",", ".join(DEGp[DEGp.logfoldchanges<0].names))
print("adj  vs paren: up=",", ".join(DEGa[DEGa.logfoldchanges>0].names)," | down=",", ".join(DEGa[DEGa.logfoldchanges<0].names))
print("\nSaved: pairwise_compartment_DEG.png + pairwise_DEG_{peri,adj}_vs_paren.csv")
