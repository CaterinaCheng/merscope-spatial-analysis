"""
361_state_compartment_heatmap_DEG.py
One figure:
 (left)  heatmap microglia subtype x compartment = log2 fold-enrichment
         (observed / expected); Fisher per cell, BH; * padj<0.05. Cell annot = % + star.
 (right) DEGs of the three compartments (peri / vessel-adj / paren), top genes per compartment,
         cross-lineage ambient flagged grey.
Strict CLEAN labels (canonical).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import fisher_exact
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
comps=["perivascular","vessel-adjacent","parenchymal"]
CCOL={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
sp=pd.read_csv(NEW/"clean_microglia_spatial.csv",index_col=0); sp=sp[sp.comp!="n/a"]
N=len(sp)
ct=pd.crosstab(sp.state,sp.comp).reindex(index=STATEORD,columns=comps).fillna(0).astype(int)
ns=ct.sum(1); nc=ct.sum(0)
L=np.zeros((len(STATEORD),len(comps))); P=np.ones_like(L); PCT=np.zeros_like(L)
for i,s in enumerate(STATEORD):
    for j,c in enumerate(comps):
        obs=ct.loc[s,c]; exp=ns[s]*nc[c]/N; L[i,j]=np.log2((obs+1e-6)/(exp+1e-6)); PCT[i,j]=100*obs/nc[c]
        a=obs; b=ns[s]-obs; d=nc[c]-obs; e=N-ns[s]-nc[c]+obs
        P[i,j]=fisher_exact([[a,b],[d,e]])[1]
pad=np.minimum(P.ravel()*P.size/np.argsort(np.argsort(P.ravel())).reshape(-1).clip(1)+0,1)  # placeholder
ps=P.ravel(); o=np.argsort(ps); rk=np.empty(len(ps),int); rk[o]=np.arange(1,len(ps)+1); padj=np.minimum(ps*len(ps)/rk,1).reshape(P.shape)
pd.DataFrame(L,index=STATEORD,columns=comps).to_csv(NEW/"state_compartment_enrichment_matrix.csv")
print("log2 fold-enrichment (state x compartment):"); print(pd.DataFrame(L,index=STATEORD,columns=comps).round(2).to_string())
# DEGs
DEG=pd.read_csv(NEW/"compartment_DEG.csv")
# ================= FIGURE =================
fig=plt.figure(figsize=(15.5,6.2)); gs=fig.add_gridspec(1,2,width_ratios=[0.85,1.15],wspace=0.28)
# heatmap
axH=fig.add_subplot(gs[0,0]); vmax=np.abs(L).max()
im=axH.imshow(L,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
axH.set_xticks(range(len(comps))); axH.set_xticklabels([f"{c}\n(n={nc[c]})" for c in comps],fontsize=9)
axH.set_yticks(range(len(STATEORD))); axH.set_yticklabels(STATEORD)
for i in range(len(STATEORD)):
    for j in range(len(comps)):
        star="*" if padj[i,j]<0.05 else ""
        axH.text(j,i,f"{L[i,j]:+.2f}{star}\n{PCT[i,j]:.0f}%",ha="center",va="center",fontsize=8.5,color="white" if abs(L[i,j])>vmax*0.55 else "#222")
axH.set_title("Microglia subtype × compartment\nlog2 fold-enrichment (obs/exp); * padj<0.05; %=share of compartment",fontsize=10,fontweight="bold")
fig.colorbar(im,ax=axH,shrink=0.7,label="log2 enrichment")
# DEG bars
axD=fig.add_subplot(gs[0,1]); yoff=0; yt=[]; ylab=[]
for c in comps:
    sub=DEG[DEG.comp==c].sort_values("logfoldchanges",ascending=False).head(11)
    yt.append(yoff+ (len(sub)-1)/2); ylab.append(c)
    for _,r in sub.iloc[::-1].iterrows():
        col="#999" if r.spillover else CCOL[c]
        axD.barh(yoff,r.logfoldchanges,color=col,edgecolor="k" if r.spillover else "none",linewidth=0.5)
        axD.text(r.logfoldchanges+0.02,yoff,r.names+(" (amb)" if r.spillover else ""),va="center",fontsize=6.8)
        yoff+=1
    yoff+=1.2
axD.set_yticks([]); axD.set_xlabel("log2 fold-change vs other compartments"); axD.set_xlim(0,None)
for c in comps: axD.scatter([],[],color=CCOL[c],label=c)
axD.scatter([],[],color="#999",label="cross-lineage ambient")
axD.legend(fontsize=8,loc="lower right")
axD.set_title("Compartment DEGs (peri / vessel-adj / paren) — top up genes",fontsize=10,fontweight="bold")
for spn in axD.spines.values(): spn.set_visible(False)
# label compartment groups on left of DEG panel
for yc,c in zip(yt,comps): axD.text(-0.02,yc,c,rotation=90,va="center",ha="right",fontsize=8.5,fontweight="bold",color=CCOL[c],transform=axD.get_yaxis_transform())
plt.tight_layout(); fig.savefig(NEW/"state_compartment_heatmap_DEG.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: state_compartment_heatmap_DEG.png + state_compartment_enrichment_matrix.csv")
