"""
370_regen_clean_qc_fig.py
Regenerate the CLEAN-microglia scHPF QC figure from saved CSVs (no scHPF retrain), with the
UMAP state legend moved outside (to the right) of the UMAP panel.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
GORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
SIGN=["Microglia-core","Astrocyte","Oligo","OPC","Neuron","Vascular","Lymphoid"]
T=pd.read_csv(NEW/"microglia_clean_cluster_qc.csv")
co=pd.read_csv(NEW/"microglia_clean_coords.csv",index_col=0)
U=co[["umap1","umap2"]].values; st=co["state"].values
ncl=len(T); yt=np.arange(ncl)
fig=plt.figure(figsize=(16,max(6,0.42*ncl))); gsf=fig.add_gridspec(1,4,width_ratios=[0.22,1.0,0.55,1.05],wspace=0.06)
axs=fig.add_subplot(gsf[0,0])
for i,s in enumerate(T.state): axs.add_patch(plt.Rectangle((0,i-0.5),1,1,color=SCOL.get(s,"#ccc")))
axs.set_xlim(0,1); axs.set_ylim(ncl-0.5,-0.5); axs.set_xticks([]); axs.set_yticks(yt); axs.set_yticklabels([f"cl{c}" for c in T.cl],fontsize=8); axs.set_title("state",fontsize=9)
for sp in axs.spines.values(): sp.set_visible(False)
axh=fig.add_subplot(gsf[0,1]); H=T[SIGN].values; Hz=(H-H.mean(0))/(H.std(0)+1e-9)
im=axh.imshow(Hz,cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
axh.set_xticks(range(len(SIGN))); axh.set_xticklabels(SIGN,rotation=35,ha="right"); axh.set_yticks(yt); axh.set_yticklabels([])
for i in range(ncl):
    for j in range(len(SIGN)): axh.text(j,i,f"{H[i,j]:.2f}",ha="center",va="center",fontsize=6.5,color="white" if abs(Hz[i,j])>1.3 else "#333")
axh.set_title("identity / contamination (number=raw score; color=z across clusters)",fontsize=9.5,fontweight="bold"); fig.colorbar(im,ax=axh,shrink=0.5,label="z")
axq=fig.add_subplot(gsf[0,2]); axq.barh(yt-0.2,T.med_counts,height=0.4,color="#555"); axq2=axq.twiny(); axq2.barh(yt+0.2,T.med_genes,height=0.4,color="#E67E22")
axq.set_ylim(ncl-0.5,-0.5); axq.set_yticks([]); axq.set_xlabel("median counts",fontsize=8); axq2.set_xlabel("median n_genes",color="#E67E22",fontsize=8); axq.set_title("depth/complexity",fontsize=9.5,fontweight="bold")
for sp in axq.spines.values(): sp.set_visible(False)
axU=fig.add_subplot(gsf[0,3])
for s in [x for x in GORD if (st==x).any()]+(["Mixed/low"] if (st=="Mixed/low").any() else []):
    m=st==s; axU.scatter(U[m,0],U[m,1],s=3,c=SCOL.get(s,"#bbb"),linewidths=0,alpha=0.85,rasterized=True,label=f"{s} ({100*m.mean():.0f}%)")
axU.set_xticks([]); axU.set_yticks([]); axU.set_aspect("equal"); axU.set_title("clean microglia UMAP\n(scHPF + Green 5-state)",fontsize=9.5,fontweight="bold")
axU.legend(markerscale=3,fontsize=8,loc="center left",bbox_to_anchor=(1.06,0.5),frameon=False)
for sp in axU.spines.values(): sp.set_visible(False)
fig.suptitle(f"CLEAN microglia ({len(co)} of 83133 cells passed QC) — scHPF clustering QC",fontsize=13,fontweight="bold",y=1.0)
plt.tight_layout(); fig.savefig(NEW/"microglia_clean_cluster_qc.png",dpi=200,bbox_inches="tight"); plt.close()
print("Saved: microglia_clean_cluster_qc.png (legend moved right)")
