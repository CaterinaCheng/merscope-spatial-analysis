"""
368_spatial_LR_dotplot.py
Spatial ligand-receptor co-expression dotplot (one row per pair).
For each pair (ligand L on microglia, receptor R on a T subset S):
  - among microglia within 30um of receptor+ S cells, fraction expressing L (local)
  - spatial fold enrichment = local / global microglial L+ rate  (null = random ligand placement)
  - binomial test (BH) -> black outline if significant
Dot size = % of subset expressing receptor; color = spatial fold enrichment.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import binomtest
import warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","NK"]
PAIRS=[("CXCL16","CXCR6","CXCL16→CXCR6 (retention)"),("CCL2","CCR2","CCL2→CCR2 (chemotaxis)"),
       ("CD86","CD28","CD86→CD28 (costim)"),
       ("HLA-DRA","CD4","MHC-II→CD4 (Ag-pres)")]
RADIUS=30
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vset=set(var)
PAIRS=[p for p in PAIRS if p[0] in vset and p[1] in vset]
print("on-panel pairs:",[p[2] for p in PAIRS])
for l,r,_ in [("CCL2","CCR2","")]:
    print(f"  CCL2 on panel={ 'CCL2' in vset}  CCR2 on panel={'CCR2' in vset}")
is_mic=(v2=="Mic"); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
def expr(gene): return np.asarray(Xd[:,var.index(gene)].todense()).ravel() if gene in vset else np.zeros(len(idx))
# coordinates
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
FOLD=np.full((len(PAIRS),len(SUBS)),np.nan); SIZE=np.zeros_like(FOLD); P=np.ones_like(FOLD); NREC=np.zeros_like(FOLD,int); POOL=np.zeros_like(FOLD,int)
for pi,(L,R,plab) in enumerate(PAIRS):
    Le=expr(L)>0; Re=expr(R)>0
    micAll=is_mic&hasxy; gL=Le[micAll].mean()  # global microglial ligand+ rate
    for si,S in enumerate(SUBS):
        # CD4 is a coreceptor only on CD4 T cells -> MHC-II→CD4 valid for CD4 subsets only (CD4 on CD8/NK = ambient)
        if R=="CD4" and not S.startswith("CD4"): continue
        recv=(labv==S)&Re&hasxy; SIZE[pi,si]=100*((labv==S)&hasxy&Re).sum()/max(((labv==S)&hasxy).sum(),1)
        NREC[pi,si]=int(recv.sum()); pool_tot=0; pool_L=0
        for r in np.unique(run[recv]):
            mm=np.where(is_mic&(run==r)&hasxy)[0]
            if len(mm)==0: continue
            tree=cKDTree(np.column_stack([mx[mm],my[mm]]))
            rc=np.where(recv&(run==r))[0]
            if len(rc)==0: continue
            nb=tree.query_ball_point(np.column_stack([mx[rc],my[rc]]),RADIUS)
            for lst in nb:
                if lst: pool_tot+=len(lst); pool_L+=Le[mm[lst]].sum()
        POOL[pi,si]=pool_tot
        if pool_tot>=10 and gL>0:
            FOLD[pi,si]=(pool_L/pool_tot)/gL
            P[pi,si]=binomtest(int(pool_L),int(pool_tot),gL).pvalue
ok=~np.isnan(FOLD); ps=P[ok]; o=np.argsort(ps); rk=np.empty(len(ps),int); rk[o]=np.arange(1,len(ps)+1)
padj=np.ones_like(P); padj[ok]=np.minimum(ps*len(ps)/rk,1)
pd.DataFrame(FOLD,index=[p[2] for p in PAIRS],columns=SUBS).to_csv(NEW/"spatial_LR_fold.csv")
print(f"\n# testable cells: {int(ok.sum())}")
print("raw p (uncorrected):"); print(pd.DataFrame(np.where(ok,P,np.nan),index=[p[1] for p in PAIRS],columns=SUBS).round(3).to_string())
print("BH padj:"); print(pd.DataFrame(np.where(ok,padj,np.nan),index=[p[1] for p in PAIRS],columns=SUBS).round(3).to_string())
sig=[(PAIRS[pi][2],SUBS[si],round(FOLD[pi,si],2),round(P[pi,si],4),round(padj[pi,si],3)) for pi in range(len(PAIRS)) for si in range(len(SUBS)) if ok[pi,si] and padj[pi,si]<0.05]
sigraw=[(PAIRS[pi][2],SUBS[si],round(FOLD[pi,si],2),round(P[pi,si],4)) for pi in range(len(PAIRS)) for si in range(len(SUBS)) if ok[pi,si] and P[pi,si]<0.05]
print("\nSIGNIFICANT after BH (padj<0.05):",sig if sig else "NONE")
print("nominally significant (raw p<0.05, pre-correction):",sigraw)
# ================= FIGURE =================
fig,ax=plt.subplots(figsize=(10,3.2)); vmax=min(5,np.nanmax(FOLD)) if np.isfinite(FOLD).any() else 2
norm=TwoSlopeNorm(vcenter=1,vmin=0,vmax=vmax)
for pi in range(len(PAIRS)):
    for si in range(len(SUBS)):
        if np.isnan(FOLD[pi,si]): continue   # insufficient receptor+ cells -> leave blank
        c=plt.cm.RdBu_r(norm(min(FOLD[pi,si],vmax)))
        ax.scatter(si,len(PAIRS)-1-pi,s=8+SIZE[pi,si]*9,color=c,edgecolors="black" if padj[pi,si]<0.05 else "#aaa",linewidths=1.3 if padj[pi,si]<0.05 else 0.4)
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels(SUBS,rotation=35,ha="right")
ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([p[2] for p in PAIRS][::-1],fontsize=9)
ax.set_xlim(-0.6,len(SUBS)-0.4); ax.set_ylim(-0.6,len(PAIRS)-0.4)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title(f"Microglia→T spatial ligand–receptor co-expression (≤{RADIUS}µm)\ncolor = spatial fold enrichment of ligand⁺ microglia near receptor⁺ T cells; black ring = BH-FDR<0.05  (blank = too few receptor⁺ cells)",fontsize=9,fontweight="bold")
sm=plt.cm.ScalarMappable(cmap="RdBu_r",norm=norm); fig.colorbar(sm,ax=ax,shrink=0.7,label="spatial fold enrichment vs null")
import matplotlib.lines as ml
sl=[ml.Line2D([],[],marker='o',linestyle='',markerfacecolor='#bbb',markeredgecolor='#888',markersize=np.sqrt(8+p*9)/1.6,label=f"{p}%") for p in [5,25,50]]
ax.legend(handles=sl,title="% expressing receptor",loc="center left",bbox_to_anchor=(1.18,0.5),fontsize=8,labelspacing=1.3)
plt.tight_layout(); fig.savefig(NEW/"spatial_LR_dotplot.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nspatial fold enrichment:"); print(pd.DataFrame(FOLD,index=[p[0]+'_'+p[1] for p in PAIRS],columns=SUBS).round(2).to_string())
print("\nreceptor+ cell count per cell (NREC); blank/× cells have too few -> microglia-neighbor pool<10:")
print(pd.DataFrame(NREC,index=[p[1] for p in PAIRS],columns=SUBS).to_string())
print("\nmicroglia-neighbor pool size (POOL); '× = n/a' where <10:")
print(pd.DataFrame(POOL,index=[p[0]+'_'+p[1] for p in PAIRS],columns=SUBS).to_string())
print("Saved: spatial_LR_dotplot.png + spatial_LR_fold.csv")
