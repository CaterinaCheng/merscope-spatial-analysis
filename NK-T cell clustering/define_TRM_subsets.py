"""
303_define_TRM_subsets.py
Define the two literature-based CD8 TRM subsets (Smolders 2018 / brain TRM signature):
  CD103+ memory/quiescent TRM : ITGAE/CXCR6+, PD1/CTLA4+, LOW cytotoxic, low/int T-bet/Eomes.
  CD103- GZMK+ effector TRM   : GZMK/GZMB/GNLY+, T-bet+, LACK memory (IL7R/CD27/CD28/TCF7/LEF1), CCR7-.
Keep CD8 TEMRA (circulating effector: FCGR3A/CX3CR1/S1PR5) separate.
Also relabel 'CD4 T (unresolved)' after checking it resembles CD4 Tcm/mem.
Re-make the subset dot plot with literature markers.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
sc.settings.verbosity=0
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
H5=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")

cd8=pd.read_csv(NEW/"Tcell_CD8_robust_label.csv",index_col=0)["robust"].replace({"CD8 TEMRA-like":"CD8 TEMRA"})
cd4=pd.read_csv(NEW/"Tcell_CD4_robust_label.csv",index_col=0)["robust"]
lin=pd.read_csv(NEW/"Tcell_lineage_assignment.csv").set_index("cell_id")["lineage"]
nk=pd.Series("NK",index=lin[lin=="NK"].index)

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
pos={c:i for i,c in enumerate(idx)}; vs=set(var)
A=ad.AnnData(X=X,var=pd.DataFrame(index=var)); sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)

# --- split the CD8 TRM pool (TRM + cytotoxic, not TEMRA) into the two literature subsets ---
trm_pool=cd8[cd8.isin(["CD8 TRM","CD8 cytotoxic (Tem/Trm)"])].index
cells=[c for c in trm_pool if c in pos]; sub=A[[pos[c] for c in cells]].copy()
EFF=[g for g in ["GZMK","GZMB","GZMA","GZMH","PRF1","GNLY","NKG7","TBX21","EOMES","KLRG1"] if g in vs]
MEMRES=[g for g in ["ITGAE","CXCR6","IL7R","CD27","CD28","TCF7","LEF1","CCR7","SELL","PDCD1","CTLA4","ZNF683"] if g in vs]
sc.tl.score_genes(sub,EFF,score_name="eff"); sc.tl.score_genes(sub,MEMRES,score_name="memres")
ez=(sub.obs.eff-sub.obs.eff.mean())/sub.obs.eff.std(); mz=(sub.obs.memres-sub.obs.memres.mean())/sub.obs.memres.std()
newlab=np.where(ez>mz,"CD8 TRM (CD103- GZMK+ effector)","CD8 TRM (CD103+ memory)")
trm_assign=pd.Series(newlab,index=cells)
print("CD8 TRM pool split:",trm_assign.value_counts().to_dict())

# --- CD4 unresolved: confirm it resembles Tcm/mem ---
def mean_log(genes,mask_ids):
    gi=[pos[c] for c in mask_ids if c in pos]; sl=A[gi]
    return {g:float(np.asarray(sl[:,g].X.todense()).ravel().mean()) for g in genes if g in vs}
memg=["CCR7","SELL","IL7R","TCF7","LEF1","CD27","CD28"]
unres=cd4[cd4=="CD4 T (unresolved)"].index; tcm=cd4[cd4=="CD4 Tcm/mem"].index
print("\nCD4 unresolved vs Tcm/mem mean memory-marker expr:")
mu=mean_log(memg,unres); mt=mean_log(memg,tcm)
for g in memg:
    if g in mu: print(f"  {g:6} unresolved={mu[g]:.2f}  Tcm/mem={mt.get(g,float('nan')):.2f}")
cd4=cd4.replace({"CD4 T (unresolved)":"CD4 Tcm/mem (broad)"})

# --- assemble final labels ---
final=cd8.copy()
final.loc[trm_assign.index]=trm_assign.values
final=pd.concat([final,cd4,nk]); final=final[~final.index.duplicated()]
final.name="subset_final"; final.to_csv(NEW/"Tcell_subset_final_labels.csv")
print("\nFINAL subset counts:",final.value_counts().to_dict())

# --- dot plot with literature markers ---
cells2=[c for c in final.index if c in pos]; D=A[[pos[c] for c in cells2]].copy(); D.obs["subset"]=final.loc[cells2].values
GROUPS={"Lineage":["CD3D","CD3E","CD8A","CD8B","CD4"],
 "Residency/CD103":["CD69","ITGAE","CXCR6","ITGA1","ZNF683"],
 "Memory/quiescent":["IL7R","CD27","CD28","TCF7","LEF1","CCR7","SELL","PDCD1","CTLA4"],
 "Cytotoxic/effector":["GZMK","GZMB","GZMA","GZMH","PRF1","GNLY","NKG7","TBX21","EOMES"],
 "Circulating (TEMRA)":["FCGR3A","FGFBP2","CX3CR1","S1PR5","KLRG1"],
 "Treg":["FOXP3","IL2RA","IKZF2","TIGIT"],"NK":["KLRD1","KLRF1","KLRC1"]}
GROUPS={k:[g for g in v if g in vs] for k,v in GROUPS.items()}; GROUPS={k:v for k,v in GROUPS.items() if v}
order=[s for s in ["CD8 TRM (CD103+ memory)","CD8 TRM (CD103- GZMK+ effector)","CD8 TEMRA",
 "CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Tcm/mem (broad)","CD4 Treg","NK"] if s in set(D.obs["subset"])]
D.obs["subset"]=pd.Categorical(D.obs["subset"],categories=order,ordered=True)
sc.pl.dotplot(D,GROUPS,groupby="subset",standard_scale="var",show=False,figsize=(15,4.6),dot_max=0.8,colorbar_title="scaled mean")
plt.savefig(NEW/"Tcell_TRM2_dotplot.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: Tcell_subset_final_labels.csv + Tcell_TRM2_dotplot.png")
