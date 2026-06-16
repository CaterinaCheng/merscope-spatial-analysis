"""
364_rawscore_heatmap_DEGbars.py
(left)  microglia state x compartment RAW averaged Green score heatmap (same as 362).
(right) compartment DEGs as RED/BLUE bars (up=red, down=blue) per compartment: log2FC vs
        the other two compartments; cross-lineage ambient genes marked. (NOT a heatmap.)
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, rankdata
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
DEC=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
comps=["perivascular","vessel-adjacent","parenchymal"]
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
UP_RED="#C0392B"; DN_BLUE="#2471A3"
SPILL=set(("AQP4 GJA1 SLC1A3 SLC1A2 GLUL GFAP AQP9 VCAN PLPP3 MOG MAL PLP1 MOBP CNP MBP UGT8 OLIG2 CD22 SORT1 SNAP25 SYT1 RBFOX3 GAD1 MEG3 NRGN SLC17A7 NELL2 BCL11B PECAM1 CLDN5 VWF PDGFRB ACTA2 RGS5 COL1A1 COL3A1 FN1 A2M FLT1 CD3D CD3E IL7R CD8A CXCR6 NKG7 MS4A1 SKAP1 CCL5 SPIB LIME1 NCR3").split())
g6=pd.read_csv(NEW/"green_mic_state_signatures.csv")
def colg(key): return [x for x in g6[[c for c in g6.columns if key in c][0]].dropna()]
GSIG={"Homeostatic":colg("Mic.2"),"MHC-II/APC":colg("Mic.9"),"DAM":sorted(set(colg("Mic.12"))|set(colg("Mic.13"))),"Phagocytic":colg("Mic.7"),"Inflammatory/IEG":colg("Mic.15")}
co=pd.read_csv(NEW/"microglia_final_coords.csv",index_col=0); co=co[~co.cluster_flag]
spc=pd.read_csv(NEW/"clean_microglia_spatial.csv",index_col=0)[["comp"]]
co=co.join(spc,how="inner"); co=co[co.comp!="n/a"]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; rowi=np.array([pos[c] for c in co.index])
A=ad.AnnData(X=Xd[rowi].copy(),obs=pd.DataFrame({"comp":co.comp.values},index=co.index),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,gl in GSIG.items(): sc.tl.score_genes(A,[x for x in gl if x in A.var_names],score_name=k,ctrl_size=50)
# z-score each state signature across all microglia (common scale for the heatmap)
for st in STATEORD:
    v=A.obs[st].values.astype(float); A.obs[st]=(v-v.mean())/(v.std()+1e-9)
compv=A.obs.comp.values
def cliffs(a,b):
    n1,n2=len(a),len(b)
    if n1==0 or n2==0: return 0
    r=rankdata(np.concatenate([a,b])); U=r[:n1].sum()-n1*(n1+1)/2; return 2*U/(n1*n2)-1
M=np.zeros((len(STATEORD),len(comps))); D=np.zeros_like(M); Pv=np.ones_like(M)
for i,st in enumerate(STATEORD):
    s=A.obs[st].values
    for j,c in enumerate(comps):
        a=s[compv==c]; b=s[compv!=c]; M[i,j]=a.mean(); D[i,j]=cliffs(a,b); Pv[i,j]=mannwhitneyu(a,b).pvalue
ps=Pv.ravel(); o=np.argsort(ps); rk=np.empty(len(ps),int); rk[o]=np.arange(1,len(ps)+1); padj=np.minimum(ps*len(ps)/rk,1).reshape(Pv.shape)
star=(padj<0.05)&(np.abs(D)>=0.1)
# DEG both directions
sc.tl.rank_genes_groups(A,"comp",method="wilcoxon")
NUP,NDN=7,4; degbars={}
for c in comps:
    r=sc.get.rank_genes_groups_df(A,group=c)
    up=r[(r.pvals_adj<0.05)&(r.logfoldchanges>0)].sort_values("logfoldchanges",ascending=False).head(NUP)
    dn=r[(r.pvals_adj<0.05)&(r.logfoldchanges<0)].sort_values("logfoldchanges").head(NDN)
    degbars[c]=pd.concat([up,dn])
# ================= FIGURE =================
fig=plt.figure(figsize=(15.5,7.5)); gs=fig.add_gridspec(1,2,width_ratios=[0.8,1.0],wspace=0.32)
axA=fig.add_subplot(gs[0,0]); vm=np.abs(M).max(); norm=TwoSlopeNorm(vcenter=0,vmin=-vm,vmax=vm)
im=axA.imshow(M,cmap="RdBu_r",norm=norm,aspect="auto")
axA.set_xticks(range(len(comps))); axA.set_xticklabels([f"{c}\n(n={int((compv==c).sum())})" for c in comps]); axA.set_yticks(range(len(STATEORD))); axA.set_yticklabels(STATEORD)
for i in range(len(STATEORD)):
    for j in range(len(comps)): axA.text(j,i,f"{M[i,j]:.2f}"+("*" if star[i,j] else ""),ha="center",va="center",fontsize=9,color="white" if abs(M[i,j])>vm*0.55 else "#222")
axA.set_title("Green state score by compartment (Z-SCORED across microglia)\n(* BH-FDR<0.05 & |Cliff δ|≥0.1) — color = SD from mean",fontsize=9.5,fontweight="bold")
fig.colorbar(im,ax=axA,shrink=0.7,label="mean z-score")
# DEG bars: 3 compartment blocks, up=red down=blue
axB=fig.add_subplot(gs[0,1]); yoff=0; sep=1.2; blockc=[]
for c in comps:
    db=degbars[c]; y0=yoff
    for _,r in db.iloc[::-1].iterrows():
        col=UP_RED if r.logfoldchanges>0 else DN_BLUE
        axB.barh(yoff,r.logfoldchanges,color=col,edgecolor="k" if r.names in SPILL else "none",linewidth=0.6,hatch="///" if r.names in SPILL else None)
        ha="left" if r.logfoldchanges>0 else "right"; off=0.02 if r.logfoldchanges>0 else -0.02
        axB.text(r.logfoldchanges+off,yoff,r.names+(" (amb)" if r.names in SPILL else ""),va="center",ha=ha,fontsize=7)
        yoff+=1
    blockc.append((y0+ (yoff-1-y0)/2,c)); yoff+=sep
axB.axvline(0,color="k",lw=0.9); axB.set_yticks([]); axB.set_xlabel("log2 fold-change vs other compartments")
axB.set_title("Compartment DEGs  (red = up, blue = down; hatched = cross-lineage ambient)",fontsize=10,fontweight="bold")
for yc,c in blockc: axB.text(-0.02,yc,c,rotation=90,va="center",ha="right",fontweight="bold",fontsize=9,color={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}[c],transform=axB.get_yaxis_transform())
axB.invert_yaxis()
for spn in axB.spines.values(): spn.set_visible(False)
from matplotlib.patches import Patch
axB.legend(handles=[Patch(color=UP_RED,label="up in compartment"),Patch(color=DN_BLUE,label="down in compartment")],fontsize=8,loc="lower right")
plt.tight_layout(); fig.savefig(NEW/"zscore_heatmap_DEGbars.png",dpi=200,bbox_inches="tight"); plt.close()
print("Saved: zscore_heatmap_DEGbars.png")
print("z-scored state x compartment:"); print(pd.DataFrame(M,index=STATEORD,columns=comps).round(3).to_string())
for c in comps: print(f"\n{c}: up="+", ".join(degbars[c][degbars[c].logfoldchanges>0].names)+" | down="+", ".join(degbars[c][degbars[c].logfoldchanges<0].names))
