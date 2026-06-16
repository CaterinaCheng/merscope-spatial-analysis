"""
293_project_merscope.py  (Stage 3: project MERSCOPE onto abl5197 scHPF reference)
Subset MERSCOPE raw counts to the reference gene order, scHPF.project() -> per-cell factor
scores in the reference space. Nearest-centroid label transfer using reference
factor x celltype centroids. Saves projected factors + transferred T/NK subset labels.
Run AFTER 292b finishes.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, schpf
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import normalize
REFD=Path(r"D:\Caterina\MERSCOPE\reference"); SAVE=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
LAB=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")   # existing phenotype labels (pre-tonight)
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")

genes=pd.read_csv(REFD/"abl5197_T_schpf_genes.txt",header=None)[0].tolist()
model=schpf.load_model(str(REFD/"abl5197_T_schpf_consensus.joblib"))
fct=pd.read_csv(REFD/"abl5197_T_schpf_factor_by_celltype.csv",index_col=0)  # celltype x F
print(f"reference: {model.nfactors} factors, {len(genes)} genes, {fct.shape[0]} celltypes")

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float64)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}
cols=[vp[g] for g in genes]
# project the T/NK compartment (reference is a T/NK atlas; non-T projections are meaningless)
ist=np.where(v2=="T/NK")[0]; tidx=idx[ist]
Xq=X[ist][:,cols].tocsr()
print(f"MERSCOPE T/NK query: {Xq.shape} (cells x reference genes)")

proj=model.project(coo_matrix(Xq))
theta=proj.cell_score() if hasattr(proj,"cell_score") else np.asarray(proj)
F=[f"F{i}" for i in range(theta.shape[1])]
pd.DataFrame(theta,index=tidx,columns=F).to_csv(SAVE/"merscope_Tcell_projected_factors.csv")
print("projected factor scores saved")

# nearest-centroid label transfer (cosine in factor space)
cent=fct[F].values
qn=normalize(theta); cn=normalize(cent)
sim=qn@cn.T
best=sim.argmax(1); margin=np.sort(sim,1)[:,-1]-np.sort(sim,1)[:,-2]
lab=pd.DataFrame({"cell_id":tidx,"ref_celltype":[fct.index[b] for b in best],"sim":sim.max(1),"margin":margin})
# merge existing phenotype labels for comparison
ph=pd.concat([pd.read_csv(LAB/"schpf_CD8_final_labels.csv")[["cell_id","phenotype"]],
              pd.read_csv(LAB/"schpf_CD4_final_labels.csv")[["cell_id","phenotype"]]]).set_index("cell_id")["phenotype"]
lab["existing_phenotype"]=lab.cell_id.map(ph).fillna("(unphenotyped T/NK)")
lab.to_csv(SAVE/"merscope_Tcell_transferred_celltype.csv",index=False)
print("\ntransferred abl5197 ref-celltype distribution (T/NK compartment):")
print(lab.ref_celltype.value_counts().to_string())
print("\nexisting phenotype vs transferred ref-celltype (crosstab):")
print(pd.crosstab(lab.existing_phenotype,lab.ref_celltype).to_string())
print("\nSaved: merscope_Tcell_projected_factors.csv + merscope_Tcell_transferred_celltype.csv")
