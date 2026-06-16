"""
367_LR_dotplot.py
Microglia <-> T-cell ligand-receptor coexpression dotplot.
Keep curated L-R pairs whose BOTH partners are on the 550-gene panel; show each gene's
expression (dot size=%detect, color=z mean across cell types) in microglia + T subsets,
rows grouped by pair (ligand then receptor) and split into Microglia->T and T->Microglia.
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
MIC_TO_T=[("CXCL16","CXCR6"),("CCL3","CCR5"),("CCL4","CCR5"),("CXCL9","CXCR3"),("CXCL10","CXCR3"),
 ("CD86","CD28"),("CD86","CTLA4"),("CD80","CD28"),("CD274","PDCD1"),("PDCD1LG2","PDCD1"),
 ("IL15","IL2RB"),("IL18","IL18R1"),("ICOSLG","ICOS"),("CLEC2D","KLRB1"),("SPP1","CD44"),
 ("TGFB1","TGFBR2"),("ICAM1","ITGAL"),("VCAM1","ITGA4"),("NECTIN2","TIGIT"),("PVR","TIGIT"),
 ("CD48","CD2"),("CD58","CD2"),("TNFSF13B","TNFRSF13C"),("HLA-DRA","CD4")]
T_TO_MIC=[("IFNG","IFNGR1"),("CD40LG","CD40"),("TNF","TNFRSF1A"),("TNF","TNFRSF1B"),("CSF1","CSF1R"),
 ("CSF2","CSF2RA"),("CSF2","CSF2RB"),("CCL5","CCR1"),("CCL5","CCR5"),("LTB","LTBR"),("FASLG","FAS"),("IL10","IL10RA")]
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
co=pd.read_csv(NEW/"microglia_final_coords.csv",index_col=0); micids=set(co[~co.cluster_flag].index)
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
vset=set(var)
def valid(pairs): return [(l,r) for l,r in pairs if l in vset and r in vset]
M2T=valid(MIC_TO_T); T2M=valid(T_TO_MIC)
print("Microglia->T pairs on panel:",M2T)
print("T->Microglia pairs on panel:",T2M)
dropped=[p for p in MIC_TO_T+T_TO_MIC if p not in M2T+T2M]; print("dropped (partner off-panel):",dropped)
# cell groups
A=ad.AnnData(X=Xd.copy(),obs=pd.DataFrame(index=idx),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
grp=pd.Series(index=idx,dtype=object)
grp[idx.isin(micids)]="Microglia"
labv=lab.reindex(idx)
for s in SUBS: grp[(labv==s).values]=s
A.obs["grp"]=grp.values; A=A[A.obs.grp.notna()].copy()
GROUPS=["Microglia"]+SUBS
# build ordered gene list with pair/role metadata
rows=[]  # (gene, role, pairlabel, block)
for (l,r) in M2T: rows.append((l,"L",f"{l}→{r}","Mic→T")); rows.append((r,"R",f"{l}→{r}","Mic→T"))
for (l,r) in T2M: rows.append((l,"L",f"{l}→{r}","T→Mic")); rows.append((r,"R",f"{l}→{r}","T→Mic"))
genes=[g for g,_,_,_ in rows]
# mean + %det per group
mean=np.zeros((len(rows),len(GROUPS))); pct=np.zeros_like(mean)
for gi,(gn,_,_,_) in enumerate(rows):
    e=np.asarray(A[:,gn].X.todense()).ravel()
    for cj,G in enumerate(GROUPS):
        m=A.obs.grp.values==G; mean[gi,cj]=e[m].mean(); pct[gi,cj]=100*(e[m]>0).mean()
z=(mean-mean.mean(1,keepdims=True))/(mean.std(1,keepdims=True)+1e-9)
pd.DataFrame(mean,index=[f"{g} ({rl})" for g,rl,_,_ in rows],columns=GROUPS).to_csv(NEW/"LR_meanexpr.csv")
# ================= FIGURE =================
nrow=len(rows); fig,ax=plt.subplots(figsize=(10.5,0.32*nrow+2))
for gi in range(nrow):
    for cj in range(len(GROUPS)):
        ax.scatter(cj,nrow-1-gi,s=8+pct[gi,cj]*3.2,c=[plt.cm.RdBu_r((z[gi,cj]+2.5)/5)],edgecolors="#888",linewidths=0.3)
ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(GROUPS,rotation=35,ha="right")
ylabs=[f"{g}" for g,_,_,_ in rows]; ax.set_yticks(range(nrow)); ax.set_yticklabels(ylabs[::-1],fontsize=8)
# color the L/R and ligand-side cell, plus pair brackets
for gi,(gn,role,pl,blk) in enumerate(rows):
    yy=nrow-1-gi
    ax.text(-1.5,yy,role,ha="center",va="center",fontsize=7,fontweight="bold",color="#C0392B" if role=="L" else "#2471A3")
# pair labels on far left (one per pair, centered between its 2 rows)
seen=set();
for gi,(gn,role,pl,blk) in enumerate(rows):
    if pl in seen: continue
    seen.add(pl); yy=nrow-1-(gi+0.5)
    ax.text(-3.2,yy,pl,ha="right",va="center",fontsize=7.5,color="#444")
# block separators
b1=sum(1 for r in rows if r[3]=="Mic→T")
ax.axhline(nrow-b1-0.5,color="#aaa",lw=1,ls="--")
ax.text(len(GROUPS)-0.4,nrow-b1/2,"Microglia→T\n(ligand on microglia)",fontsize=8,fontweight="bold",rotation=90,va="center",ha="left",color="#C0392B")
ax.text(len(GROUPS)-0.4,(nrow-b1)/2,"T→Microglia\n(ligand on T)",fontsize=8,fontweight="bold",rotation=90,va="center",ha="left",color="#2471A3")
ax.set_xlim(-3.6,len(GROUPS)+0.8); ax.set_ylim(-0.7,nrow-0.3)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("Microglia ↔ T-cell ligand–receptor coexpression\n(dot size = % expressing; color = z mean expr across cell types; L=ligand red, R=receptor blue)",fontsize=10,fontweight="bold")
# legends
import matplotlib.lines as ml
sizeleg=[ml.Line2D([],[],marker='o',linestyle='',markerfacecolor='#bbb',markeredgecolor='#888',markersize=np.sqrt(8+p*3.2),label=f"{p}%") for p in [10,30,60]]
ax.legend(handles=sizeleg,title="% expressing",loc="upper left",bbox_to_anchor=(1.02,1),fontsize=7,labelspacing=1.2)
sm=plt.cm.ScalarMappable(cmap="RdBu_r",norm=plt.Normalize(-2.5,2.5)); fig.colorbar(sm,ax=ax,shrink=0.4,label="z mean expr",pad=0.13)
plt.tight_layout(); fig.savefig(NEW/"LR_microglia_Tcell_dotplot.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: LR_microglia_Tcell_dotplot.png + LR_meanexpr.csv")
