"""
369_spatial_LR_coexpression.py
Spatial ligand-receptor CO-EXPRESSION (not just enrichment). For each pair (L on microglia,
R on T) and subset S:
  coexpr = mean over S cells of [ mean ligand expr in microglia within 30um ] x [ receptor expr in the S cell ]
So if S cells barely express R (e.g. CD8 don't express CD4), coexpr is low BY DESIGN -- the
expected specificity control. Color = per-pair relative co-expression (0-1); dot size = % S
expressing receptor; black ring = permutation BH-FDR<0.05 (ligand shuffled among microglia).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import cKDTree
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","NK"]
PAIRS=[("CXCL16","CXCR6","CXCL16→CXCR6 (retention)"),("CCL2","CCR2","CCL2→CCR2 (chemotaxis)"),
       ("CD86","CD28","CD86→CD28 (costim)"),("HLA-DRA","CD4","MHC-II→CD4 (Ag-pres)")]
RADIUS=30; NPERM=300; rng=np.random.RandomState(0)
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
A=ad.AnnData(X=Xd.copy(),obs=pd.DataFrame(index=idx),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
def lexpr(gene): return np.asarray(A[:,gene].X.todense()).ravel() if gene in var else np.zeros(len(idx))
def rawpos(gene): return np.asarray(Xd[:,var.index(gene)].todense()).ravel()>0 if gene in var else np.zeros(len(idx),bool)
is_mic=(v2=="Mic"); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); micmask=is_mic&hasxy; micglob=np.where(micmask)[0]; micpos={gi:k for k,gi in enumerate(micglob)}
# precompute neighbor microglia (incidence) per subset cell
NEIG={}  # subset -> (incidence csr [cells x nmic], R cell indices)
for S in SUBS:
    rows=[]; cols=[]; rcells=[]
    sc_idx=np.where((labv==S)&hasxy)[0]
    ci=0
    for r in np.unique(run[sc_idx]):
        mm=np.where(is_mic&(run==r)&hasxy)[0]
        if len(mm)==0: continue
        tree=cKDTree(np.column_stack([mx[mm],my[mm]]))
        cc=sc_idx[run[sc_idx]==r]
        nb=tree.query_ball_point(np.column_stack([mx[cc],my[cc]]),RADIUS)
        for k,lst in enumerate(nb):
            if lst:
                for j in lst: rows.append(ci); cols.append(micpos[mm[j]])
                rcells.append(cc[k]); ci+=1
    if ci>0:
        M=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(ci,len(micglob)))
        NEIG[S]=(M,np.array(rcells))
COEX=np.full((len(PAIRS),len(SUBS)),np.nan); SIZE=np.zeros_like(COEX); PV=np.ones_like(COEX); NC=np.zeros_like(COEX,int)
for pi,(L,R,plab) in enumerate(PAIRS):
    Lf=lexpr(L); Rf=lexpr(R); Lmic=Lf[micglob]; Rpos=rawpos(R)
    for si,S in enumerate(SUBS):
        SIZE[pi,si]=100*((labv==S)&hasxy&Rpos).sum()/max(((labv==S)&hasxy).sum(),1)
        if S not in NEIG: continue
        M,rcells=NEIG[S]; cnt=np.asarray(M.sum(1)).ravel(); NC[pi,si]=len(rcells)
        if len(rcells)<10: continue
        Rcell=Rf[rcells]
        localL=np.asarray(M@Lmic).ravel()/np.maximum(cnt,1)
        obs=np.mean(localL*Rcell)
        null=np.empty(NPERM)
        for p in range(NPERM):
            Lp=rng.permutation(Lmic); null[p]=np.mean((np.asarray(M@Lp).ravel()/np.maximum(cnt,1))*Rcell)
        COEX[pi,si]=obs; PV[pi,si]=(1+np.sum(null>=obs))/(1+NPERM)
ok=~np.isnan(COEX); ps=PV[ok]; o=np.argsort(ps); rk=np.empty(len(ps),int); rk[o]=np.arange(1,len(ps)+1)
padj=np.ones_like(PV); padj[ok]=np.minimum(ps*len(ps)/rk,1)
# per-row min-max scaled color
REL=np.full_like(COEX,np.nan)
for pi in range(len(PAIRS)):
    row=COEX[pi]; m=~np.isnan(row)
    if m.sum(): lo,hi=np.nanmin(row),np.nanmax(row); REL[pi,m]=(row[m]-lo)/(hi-lo+1e-12)
pd.DataFrame(COEX,index=[p[2] for p in PAIRS],columns=SUBS).to_csv(NEW/"spatial_LR_coexpr.csv")
print("co-expression (raw):"); print(pd.DataFrame(COEX,index=[p[0]+'_'+p[1] for p in PAIRS],columns=SUBS).round(3).to_string())
sig=[(PAIRS[pi][2],SUBS[si]) for pi in range(len(PAIRS)) for si in range(len(SUBS)) if ok[pi,si] and padj[pi,si]<0.05]
print("\nSIGNIFICANT (perm BH<0.05):",sig)
# ================= FIGURE =================
fig,ax=plt.subplots(figsize=(9.5,3.2))
for pi in range(len(PAIRS)):
    for si in range(len(SUBS)):
        if np.isnan(COEX[pi,si]): continue
        ax.scatter(si,len(PAIRS)-1-pi,s=8+SIZE[pi,si]*9,color=plt.cm.Reds(0.15+0.8*REL[pi,si]),
                   edgecolors="black" if padj[pi,si]<0.05 else "#aaa",linewidths=1.4 if padj[pi,si]<0.05 else 0.4)
ax.set_xticks(range(len(SUBS))); ax.set_xticklabels(SUBS,rotation=35,ha="right")
ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([p[2] for p in PAIRS][::-1],fontsize=9)
ax.set_xlim(-0.6,len(SUBS)-0.4); ax.set_ylim(-0.6,len(PAIRS)-0.4)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title(f"Microglia→T spatial ligand–receptor CO-EXPRESSION (≤{RADIUS}µm)\ncolor = relative co-expression (ligand×receptor in adjacent cells, per pair); black ring = perm BH<0.05",fontsize=9,fontweight="bold")
sm=plt.cm.ScalarMappable(cmap="Reds",norm=plt.Normalize(0,1)); fig.colorbar(sm,ax=ax,shrink=0.7,label="relative co-expression")
import matplotlib.lines as ml
sl=[ml.Line2D([],[],marker='o',linestyle='',markerfacecolor='#d66',markeredgecolor='#888',markersize=np.sqrt(8+p*9)/1.6,label=f"{p}%") for p in [5,25,50]]
ax.legend(handles=sl,title="% expressing receptor",loc="center left",bbox_to_anchor=(1.2,0.5),fontsize=8,labelspacing=1.3)
plt.tight_layout(); fig.savefig(NEW/"spatial_LR_coexpression.png",dpi=200,bbox_inches="tight"); plt.close()
print("Saved: spatial_LR_coexpression.png + spatial_LR_coexpr.csv")
