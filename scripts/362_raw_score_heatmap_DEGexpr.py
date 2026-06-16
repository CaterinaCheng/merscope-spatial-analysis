"""
362_raw_score_heatmap_DEGexpr.py
(left)  microglia state x compartment heatmap = RAW averaged Green signature score
        (score_genes, background-corrected) per compartment; diverging colormap centered 0;
        * = BH-FDR<0.05 & |Cliff delta|>=0.1 (compartment vs other two).
(right) compartment DEGs shown as EXPRESSION heatmap: top DEG genes x compartment,
        number=raw mean log-expression, color=z across compartments; ambient genes marked.
Strict CLEAN microglia (canonical), 5 Green states (DAM=Mic.12+Mic.13).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
comps=["perivascular","vessel-adjacent","parenchymal"]
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
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
compv=A.obs.comp.values
def cliffs(a,b):
    n1,n2=len(a),len(b);
    if n1==0 or n2==0: return 0
    from scipy.stats import rankdata; r=rankdata(np.concatenate([a,b])); U=r[:n1].sum()-n1*(n1+1)/2; return 2*U/(n1*n2)-1
# RAW mean score per state x compartment + one-vs-rest test
M=np.zeros((len(STATEORD),len(comps))); D=np.zeros_like(M); Pv=np.ones_like(M)
for i,st in enumerate(STATEORD):
    sc_all=A.obs[st].values
    for j,c in enumerate(comps):
        a=sc_all[compv==c]; b=sc_all[compv!=c]; M[i,j]=a.mean(); D[i,j]=cliffs(a,b)
        Pv[i,j]=mannwhitneyu(a,b).pvalue
ps=Pv.ravel(); o=np.argsort(ps); rk=np.empty(len(ps),int); rk[o]=np.arange(1,len(ps)+1); padj=np.minimum(ps*len(ps)/rk,1).reshape(Pv.shape)
star=(padj<0.05)&(np.abs(D)>=0.1)
pd.DataFrame(M,index=STATEORD,columns=comps).to_csv(NEW/"state_compartment_rawscore.csv")
print("RAW mean Green score (state x compartment):"); print(pd.DataFrame(M,index=STATEORD,columns=comps).round(3).to_string())
# DEG genes x compartment expression
DEG=pd.read_csv(NEW/"compartment_DEG.csv"); genes=[]; gspill={}
for c in comps:
    for _,r in DEG[DEG.comp==c].sort_values("logfoldchanges",ascending=False).head(8).iterrows():
        if r.names not in genes and r.names in A.var_names: genes.append(r.names); gspill[r.names]=bool(r.spillover)
E=np.asarray(A[:,genes].X.todense()); Edf=pd.DataFrame(E,columns=genes)
meanE=Edf.groupby(compv).mean().reindex(comps)   # comp x gene
Ez=(meanE-meanE.mean(0))/(meanE.std(0)+1e-9)
# ================= FIGURE =================
fig=plt.figure(figsize=(15.5,7)); gs=fig.add_gridspec(1,2,width_ratios=[0.8,1.2],wspace=0.3)
axA=fig.add_subplot(gs[0,0]); vm=np.abs(M).max(); norm=TwoSlopeNorm(vcenter=0,vmin=-vm,vmax=vm)
im=axA.imshow(M,cmap="RdBu_r",norm=norm,aspect="auto")
axA.set_xticks(range(len(comps))); axA.set_xticklabels([f"{c}\n(n={int((compv==c).sum())})" for c in comps]); axA.set_yticks(range(len(STATEORD))); axA.set_yticklabels(STATEORD)
for i in range(len(STATEORD)):
    for j in range(len(comps)): axA.text(j,i,f"{M[i,j]:.3f}"+("*" if star[i,j] else ""),ha="center",va="center",fontsize=9,color="white" if abs(M[i,j])>vm*0.55 else "#222")
axA.set_title("RAW averaged Green state score by compartment\n(* BH-FDR<0.05 & |Cliff δ|≥0.1, compartment vs other two)",fontsize=9.5,fontweight="bold")
fig.colorbar(im,ax=axA,shrink=0.7,label="raw mean score_genes")
# DEG expression heatmap (genes rows x comp cols)
axB=fig.add_subplot(gs[0,1]); Hz=Ez.T.values; Hraw=meanE.T.values   # gene x comp
im2=axB.imshow(Hz,cmap="Reds",vmin=Hz.min(),vmax=Hz.max(),aspect="auto")
axB.set_xticks(range(len(comps))); axB.set_xticklabels([f"{c}\n(n={int((compv==c).sum())})" for c in comps])
axB.set_yticks(range(len(genes))); axB.set_yticklabels([g+(" (amb)" if gspill[g] else "") for g in genes],fontsize=8)
for i in range(len(genes)):
    for j in range(len(comps)): axB.text(j,i,f"{Hraw[i,j]:.2f}",ha="center",va="center",fontsize=7.5,color="white" if Hz[i,j]>Hz.max()*0.6 else "#333")
for i,g in enumerate(genes):
    if gspill[g]: axB.get_yticklabels()[i].set_color("#999")
axB.set_title("Compartment DEG expression\n(number=raw mean log-expr; color=z across compartments; grey label=ambient)",fontsize=9.5,fontweight="bold")
fig.colorbar(im2,ax=axB,shrink=0.7,label="z (across compartments)")
plt.tight_layout(); fig.savefig(NEW/"state_compartment_rawscore_DEGexpr.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: state_compartment_rawscore_DEGexpr.png + state_compartment_rawscore.csv")
