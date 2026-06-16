"""
324_plot_microglia_DEG_by_lineage_bars.py
Replot the lineage-stratified microglia DEGs as 3 independent bar panels (same style as
niche_DEG_Tmic.png). Each panel = microglia NEAR <lineage> vs BASELINE microglia (no T/NK <=30um).
Reads the CSVs written by 323. Intrinsic genes only (T-spillover excluded); red=up, blue=down, *padj<0.05.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
PANELS=[("DEG_microglia_near_CD8.csv","Microglia near CD8 vs outside T/NK niche","#27AE60"),
        ("DEG_microglia_near_CD4.csv","Microglia near CD4 vs outside T/NK niche","#27AE60"),
        ("DEG_microglia_near_NK.csv","Microglia near NK vs outside T/NK niche","#27AE60")]
N={"DEG_microglia_near_CD8.csv":490,"DEG_microglia_near_CD4.csv":113,"DEG_microglia_near_NK.csv":159}
fig,axes=plt.subplots(1,3,figsize=(19,6.5))
for ax,(fn,title,_) in zip(axes,PANELS):
    r=pd.read_csv(NEW/fn); intr=r[~r.spillover]
    up=intr[intr.log2FC>0].nsmallest(8,"pval"); dn=intr[intr.log2FC<0].nsmallest(8,"pval")
    d=pd.concat([dn,up]).drop_duplicates("gene").sort_values("log2FC"); y=np.arange(len(d))
    cols=[(0.78,0.24,0.20,1 if p<0.05 else .35) if v>0 else (0.12,0.47,0.71,1 if p<0.05 else .35) for v,p in zip(d.log2FC,d.padj)]
    ax.barh(y,d.log2FC,color=cols,edgecolor="#333",lw=0.3)
    for yi,(_,rr) in zip(y,d.iterrows()):
        ax.text(rr.log2FC+(0.04 if rr.log2FC>0 else -0.04),yi,rr.gene+(" *" if rr.padj<0.05 else ""),
                va="center",ha="left" if rr.log2FC>0 else "right",fontsize=8)
    ax.axvline(0,color="#333",lw=0.7); ax.set_yticks([]); mm=max(abs(d.log2FC).max(),0.5); ax.set_xlim(-mm*1.9,mm*1.9)
    ax.set_xlabel("log2FC (near vs outside niche)")
    nsig=int(((intr.padj<0.05)).sum())
    ax.set_title(f"{title}\n(n_near={N[fn]}; {nsig} intrinsic genes padj<0.05; * padj<0.05)",fontsize=9.5,fontweight="bold")
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"microglia_DEG_by_lineage_bars.png",dpi=140,bbox_inches="tight"); plt.close()
print("Saved: microglia_DEG_by_lineage_bars.png")
