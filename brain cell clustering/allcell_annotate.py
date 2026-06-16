"""
291_allcell_annotate.py  (Stage 1b: annotate rebuilt all-cell Leiden clusters)
Marker-module scoring per cluster (primary, robust on targeted panel) + CellTypist
(Adult_Human_PrefrontalCortex) cross-check + old cell_type_v2 cross-check.
Saves cluster->lineage map and per-cell cell_type_rebuild.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, h5py
sc.settings.verbosity=0
CMAP=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap"); SAVE=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
A=sc.read_h5ad(CMAP/"allcell_rebuild.h5ad")   # X = raw counts, obs.leiden
print(f"loaded {A.n_obs} cells, {A.obs.leiden.nunique()} clusters")
A.layers["counts"]=A.X.copy(); sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)
vp=set(A.var_names)
MARK={"Exc":["SLC17A7","RORB","CUX2","SATB2","RBFOX3"],"Inh":["GAD1","GAD2","PVALB","SST","VIP"],
 "Ast":["AQP4","GJA1","SLC1A2","GFAP"],"Oli":["MOG","MOBP","PLP1","MAL","MBP"],
 "OPC":["PDGFRA","OLIG1","OLIG2","CSPG4"],"Mic":["P2RY12","CX3CR1","TMEM119","C1QA","CSF1R"],
 "Mono/Mac":["CD163","MRC1","F13A1","LYZ","CD14"],"End":["PECAM1","FLT1","VWF","CLDN5"],
 "Per":["PDGFRB","RGS5","NOTCH3"],"SMC":["ACTA2","MYH11","TAGLN"],
 "T/NK":["CD3D","CD3E","CD2","CD8A","NKG7","GNLY"],"B":["MS4A1","CD79A","CD79B","IGHM"]}
MARK={k:[g for g in v if g in vp] for k,v in MARK.items()}; MARK={k:v for k,v in MARK.items() if v}
for k,v in MARK.items(): sc.tl.score_genes(A,v,score_name=f"m_{k}")
S=A.obs.groupby("leiden")[[f"m_{k}" for k in MARK]].mean()
# z-score across clusters per module, assign argmax
Sz=(S-S.mean())/S.std()
assign={cl:max(MARK,key=lambda k:Sz.loc[cl,f"m_{k}"]) for cl in S.index}

# CellTypist cross-check
ct_major={}
try:
    import celltypist; from celltypist import models
    mdl=models.Model.load("Adult_Human_PrefrontalCortex.pkl")
    pred=celltypist.annotate(A,model=mdl,majority_voting=True)
    A.obs["celltypist"]=pred.predicted_labels["predicted_labels"].values
    ct_major={cl:A.obs.celltypist[A.obs.leiden==cl].mode().iloc[0] for cl in S.index}
except Exception as e:
    print("CellTypist skipped:",str(e)[:120])

# old cell_type_v2 cross-check
with h5py.File(CMAP/"merged_qc_brain_remapped.h5ad","r") as f:
    og=f["obs"]; oidx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in og[og.attrs.get("_index","_index")][:]])
    n=og["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
oldv2=pd.Series(v2,index=oidx).reindex(A.obs_names).values
A.obs["old_v2"]=oldv2
old_major={cl:pd.Series(A.obs.old_v2[A.obs.leiden==cl]).mode().iloc[0] for cl in S.index}

rows=[]
for cl in S.index:
    rows.append(dict(leiden=cl,n=int((A.obs.leiden==cl).sum()),marker_lineage=assign[cl],
        top_markers=", ".join(Sz.loc[cl].sort_values(ascending=False).head(3).index.str.replace("m_","")),
        celltypist=ct_major.get(cl,"?"),old_v2=old_major[cl]))
M=pd.DataFrame(rows); M.to_csv(SAVE/"allcell_rebuild_annotation.csv",index=False)
pd.set_option("display.width",220,"display.max_colwidth",40); print(M.to_string(index=False))
A.obs["cell_type_rebuild"]=A.obs.leiden.map(assign).astype(str)
A.obs[["leiden","cell_type_rebuild","celltypist" if "celltypist" in A.obs else "old_v2","old_v2"]].to_csv(SAVE/"allcell_rebuild_celltypes.csv")
print("\nrebuilt cell-type counts:"); print(A.obs.cell_type_rebuild.value_counts().to_string())
# crosstab old vs new
print("\nold_v2 vs cell_type_rebuild (row-normalized %):")
ct=pd.crosstab(A.obs.old_v2,A.obs.cell_type_rebuild,normalize="index")*100
print(ct.round(0).astype(int).to_string())
A.write(CMAP/"allcell_rebuild.h5ad")
print("\nSaved: allcell_rebuild_annotation.csv + allcell_rebuild_celltypes.csv (+ updated h5ad)")
