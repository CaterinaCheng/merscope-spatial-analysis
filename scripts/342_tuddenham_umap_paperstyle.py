"""
342_tuddenham_umap_paperstyle.py
Render the merged-meta-group microglia UMAP in the Tuddenham 2024 (Nat Neurosci Fig 2a)
HOUSE STYLE: a HEX-BINNED UMAP where each hexagon is colored by the MAJORITY subtype of
the ~50 cells it contains, white cluster numbers at centroids, no axis box, small
UMAP1/UMAP2 arrow axes lower-left, legend mapping number -> meta-group.
Reads merged labels (341) + umap coords (336).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans"})
lab=pd.read_csv(NEW/"microglia_tuddenham_merged_labels.csv",index_col=0)
co=pd.read_csv(NEW/"microglia_umap_coords.csv",index_col=0)
df=lab.join(co[["umap1","umap2"]],how="inner")
ORDER=["Homeostatic","Metabolic/Tx","Stress","APOE/Lipid","Motility","Cytokine/IL","APC (HLA/Compl)","DAM/GPNMB","Mixed/low"]
ORDER=[o for o in ORDER if (df.subtype==o).any()]
# Tuddenham-like discrete palette (distinct, mid-saturation)
PAL={"Homeostatic":"#4C9F70","Metabolic/Tx":"#8E6FB0","Stress":"#E15759","APOE/Lipid":"#F28E2B",
     "Motility":"#76B7B2","Cytokine/IL":"#E377C2","APC (HLA/Compl)":"#4E79A7","DAM/GPNMB":"#B07A3C","Mixed/low":"#D9D9D9"}
NUM={g:(i+1) for i,g in enumerate([o for o in ORDER if o!="Mixed/low"])}
CODE={g:i for i,g in enumerate(ORDER)}                  # subtype -> int code
colors=[PAL[g] for g in ORDER]
cmap=ListedColormap(colors); norm=BoundaryNorm(np.arange(-0.5,len(ORDER)+0.5,1),len(ORDER))
codes=df.subtype.map(CODE).values.astype(float)
# target ~50 cells/hexagon (paper): gridsize ~ sqrt(N/50)
gridsize=int(round(np.sqrt(len(df)/50.0))); print("cells:",len(df),"gridsize:",gridsize)
def majority(vals):
    v=np.asarray(vals).astype(int); return np.bincount(v,minlength=len(ORDER)).argmax()
fig,ax=plt.subplots(figsize=(8.6,8))
ax.hexbin(df.umap1.values,df.umap2.values,C=codes,reduce_C_function=majority,gridsize=gridsize,
          cmap=cmap,norm=norm,mincnt=1,linewidths=0.2,edgecolors="white")
# centroid number labels (robust: mean of densest 60% core)
for g,nidx in NUM.items():
    pts=df[df.subtype.values==g][["umap1","umap2"]].values
    if len(pts)<10: continue
    c=np.median(pts,0); d=np.linalg.norm(pts-c,axis=1); core=pts[d<=np.percentile(d,60)]; c=core.mean(0)
    ax.text(c[0],c[1],str(nidx),fontsize=15,fontweight="bold",ha="center",va="center",color="white",
            path_effects=[pe.withStroke(linewidth=3.0,foreground="#2b2b2b")],zorder=10)
# corner UMAP axis arrows
x0,x1=df.umap1.min(),df.umap1.max(); y0,y1=df.umap2.min(),df.umap2.max()
ax_len=0.16*(x1-x0); ox=x0-0.02*(x1-x0); oy=y0-0.02*(y1-y0)
ax.annotate("",xy=(ox+ax_len,oy),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.6))
ax.annotate("",xy=(ox,oy+ax_len),xytext=(ox,oy),arrowprops=dict(arrowstyle="-|>",color="#333",lw=1.6))
ax.text(ox+ax_len*0.5,oy-0.03*(y1-y0),"UMAP1",fontsize=8,ha="center",va="top",color="#333")
ax.text(ox-0.03*(x1-x0),oy+ax_len*0.5,"UMAP2",fontsize=8,ha="right",va="center",color="#333",rotation=90)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_aspect("equal"); ax.set_title("Human brain microglia — Tuddenham 2024 subset mapping",fontsize=12,fontweight="bold",pad=12)
handles=[Line2D([0],[0],marker="h",linestyle="",markersize=11,markerfacecolor=PAL[g],markeredgecolor="white",
                label=(f"{NUM[g]}  {g}" if g in NUM else f"–  {g}")) for g in ORDER]
leg=ax.legend(handles=handles,fontsize=9.5,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False,handletextpad=0.5,labelspacing=0.7,title="Microglia subset",title_fontsize=10)
leg._legend_box.align="left"
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_tuddenham_paperstyle.png",dpi=200,bbox_inches="tight"); plt.close()
print("Saved: microglia_umap_tuddenham_paperstyle.png")
print("numbering:",{v:k for k,v in NUM.items()})
