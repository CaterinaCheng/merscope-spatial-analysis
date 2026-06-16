"""
348_state_separability_fig.py
Visualize why Phagocytic/Activated-DAM/APC microglia don't separate on our panel:
 A) Green signature-score correlation (collinear blocks)
 B) state x scHPF factor correlation (which states share a factor = not separable)
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
S=pd.read_csv(NEW/"microglia_green_scores.csv",index_col=0)
theta=pd.read_csv(NEW/"microglia_schpf_cell_scores.csv",index_col=0).reindex(S.index)
SH=[c.split(" (")[0] for c in S.columns]
C=S.corr(method="spearman").values
M=np.zeros((S.shape[1],theta.shape[1]))
for i in range(S.shape[1]):
    for j in range(theta.shape[1]): M[i,j]=spearmanr(S.iloc[:,i],theta.iloc[:,j]).correlation
fig,(a1,a2)=plt.subplots(1,2,figsize=(15,5.6),gridspec_kw={"width_ratios":[1,1.25]})
# A
im1=a1.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1)
a1.set_xticks(range(len(SH))); a1.set_xticklabels(SH,rotation=40,ha="right"); a1.set_yticks(range(len(SH))); a1.set_yticklabels(SH)
for i in range(len(SH)):
    for j in range(len(SH)): a1.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(C[i,j])>0.5 else "black")
a1.set_title("A. Green signature-score correlation\n(positive blocks = states that co-vary)",fontsize=10,fontweight="bold")
fig.colorbar(im1,ax=a1,shrink=0.7,label="Spearman r")
# B
vmax=np.abs(M).max(); im2=a2.imshow(M,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
a2.set_xticks(range(theta.shape[1])); a2.set_xticklabels(theta.columns); a2.set_yticks(range(len(SH))); a2.set_yticklabels(SH)
for i in range(len(SH)):
    j=int(np.argmax(M[i])); a2.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="k",lw=2))
    for jj in range(theta.shape[1]):
        if abs(M[i,jj])>0.25: a2.text(jj,i,f"{M[i,jj]:.2f}",ha="center",va="center",fontsize=7.5,color="white" if abs(M[i,jj])>0.45 else "black")
a2.set_title("B. Green state x scHPF factor correlation\n(boxed = top factor; shared box = same program)",fontsize=10,fontweight="bold")
a2.set_xlabel("scHPF factor"); fig.colorbar(im2,ax=a2,shrink=0.7,label="Spearman r")
plt.tight_layout(); fig.savefig(NEW/"microglia_state_separability.png",dpi=200,bbox_inches="tight"); plt.close()
print("Saved: microglia_state_separability.png")
