"""
356_cluster_compartment_DEG.py
(2) compartment composition of EACH clean microglia cluster (peri/adj/paren).
(3) DEGs between perivascular / vessel-adjacent / parenchymal microglia (one-vs-rest Wilcoxon
    on decontam, BH FDR), with cross-lineage spillover flag (ambient vs real microglial program).
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
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
CCOL={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
SPILL=set(("AQP4 GJA1 SLC1A3 SLC1A2 GLUL GFAP AQP9 VCAN PLPP3 SLC1A2 "         # astro
           "MOG MAL PLP1 MOBP CNP MBP UGT8 OLIG2 CD22 SORT1 "                   # oligo/OPC
           "SNAP25 SYT1 RBFOX3 GAD1 MEG3 NRGN SLC17A7 NELL2 BCL11B "            # neuron
           "PECAM1 CLDN5 VWF PDGFRB ACTA2 RGS5 COL1A1 COL3A1 FN1 IFITM3 A2M FLT1 "  # vascular
           "CD3D CD3E IL7R CD8A CXCR6 NKG7 MS4A1 SKAP1 CCL5 SPIB LIME1 NCR3").split())  # lymphoid
co=pd.read_csv(NEW/"microglia_final_coords.csv",index_col=0)
sp=pd.read_csv(NEW/"clean_microglia_spatial.csv",index_col=0)
co=co[~co.cluster_flag].join(sp[["comp"]],how="inner"); co=co[co.comp!="n/a"]
print("validated microglia with compartment:",len(co))
clusters=sorted(co.leiden.unique(),key=int); comps=["perivascular","vessel-adjacent","parenchymal"]
# ---- (2) per-cluster compartment composition ----
print("\n=== (2) compartment composition of each cluster ===")
comp_tab=[]
for cl in clusters:
    m=co.leiden==cl; st=co.state[m].value_counts().index[0]
    fr={c:100*(co.comp[m]==c).mean() for c in comps}; comp_tab.append(dict(cl=cl,state=st,n=int(m.sum()),**{c:round(fr[c],1) for c in comps}))
CT=pd.DataFrame(comp_tab); print(CT.to_string(index=False)); CT.to_csv(NEW/"cluster_compartment_composition.csv",index=False)
# ---- load decontam for DEG ----
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; rowi=np.array([pos[c] for c in co.index])
A=ad.AnnData(X=Xd[rowi].copy(),obs=pd.DataFrame({"comp":co.comp.values},index=co.index),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
sc.tl.rank_genes_groups(A,"comp",method="wilcoxon")
print("\n=== (3) compartment DEGs (one-vs-rest, top up genes; [S]=cross-lineage spillover) ===")
deg_rows=[]
for c in comps:
    r=sc.get.rank_genes_groups_df(A,group=c); r=r[r.pvals_adj<0.05]
    up=r[r.logfoldchanges>0].sort_values("logfoldchanges",ascending=False)
    up["spillover"]=up.names.isin(SPILL); up["comp"]=c
    deg_rows.append(up)
    top=up.head(15)
    print(f"\n  {c} (n={int((co.comp==c).sum())}):")
    print("    "+", ".join(f"{g}{'[S]' if s else ''}" for g,s in zip(top.names,top.spillover)))
DEG=pd.concat(deg_rows); DEG.to_csv(NEW/"compartment_DEG.csv",index=False)
# ================= FIGURE =================
fig=plt.figure(figsize=(16,6.5)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.25])
# A per-cluster compartment stacked bars
axA=fig.add_subplot(gs[0,0]); order=sorted(range(len(CT)),key=lambda i:(list(SCOL).index(CT.state[i]),-CT.perivascular[i]))
CTo=CT.iloc[order].reset_index(drop=True); x=np.arange(len(CTo)); bottom=np.zeros(len(CTo))
for c in comps:
    axA.bar(x,CTo[c],bottom=bottom,color=CCOL[c],label=c); bottom+=CTo[c].values
axA.set_xticks(x); axA.set_xticklabels([f"cl{c}\n{s[:6]}\n(n={n})" for c,s,n in zip(CTo.cl,CTo.state,CTo.n)],fontsize=7)
axA.set_ylabel("% of cluster"); axA.set_title("(2) vascular compartment of each microglia cluster",fontsize=10,fontweight="bold")
axA.legend(fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.18),ncol=3)
# B compartment DEG: top genes per compartment, colored by spillover
axB=fig.add_subplot(gs[0,1]); yoff=0; yticks=[]; ylabs=[]
for c in comps:
    up=DEG[DEG.comp==c].head(12)
    for _,row in up.iloc[::-1].iterrows():
        col=CCOL[c] if not row.spillover else "#999999"
        axB.barh(yoff,row.logfoldchanges,color=col,edgecolor="k" if row.spillover else None,linewidth=0.5)
        axB.text(row.logfoldchanges+0.03,yoff,row.names+(" (ambient)" if row.spillover else ""),va="center",fontsize=6.5)
        yticks.append(yoff); ylabs.append(""); yoff+=1
    yoff+=1
axB.set_yticks([]); axB.set_xlabel("log2 fold-change vs rest"); axB.set_xlim(0,None)
axB.set_title("(3) top DEGs per compartment  (grey/outlined = cross-lineage ambient)",fontsize=10,fontweight="bold")
for i,c in enumerate(comps):
    axB.scatter([],[],color=CCOL[c],label=c)
axB.legend(fontsize=8,loc="lower right")
for sp_ in axB.spines.values(): sp_.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"cluster_compartment_DEG.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: cluster_compartment_DEG.png + cluster_compartment_composition.csv + compartment_DEG.csv")
