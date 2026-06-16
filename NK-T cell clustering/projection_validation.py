"""
294_projection_validation.py  (Stage 5a: validate existing T phenotypes against abl5197 projection)
Mean projected factor score per existing phenotype -> heatmap (phenotype x factor),
factors labelled by their top reference celltype. Tests whether our de-novo phenotypes
occupy the expected reference-factor regions (external cross-validation).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); LAB=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF"); REFD=Path(r"D:\Caterina\MERSCOPE\reference")
fac=pd.read_csv(NEW/"merscope_Tcell_projected_factors.csv",index_col=0)
fct=pd.read_csv(REFD/"abl5197_T_schpf_factor_by_celltype.csv",index_col=0)  # celltype x F
F=[c for c in fac.columns if c.startswith("F")]
toptype={f:fct[f].idxmax() for f in F}
ph=pd.concat([pd.read_csv(LAB/"schpf_CD8_final_labels.csv")[["cell_id","phenotype"]],
              pd.read_csv(LAB/"schpf_CD4_final_labels.csv")[["cell_id","phenotype"]]]).set_index("cell_id")["phenotype"]
fac=fac.join(ph.rename("phenotype"),how="inner")
print(f"phenotyped T cells with projection: {len(fac)}")
Z=fac[F].copy(); Z=(Z-Z.mean())/Z.std()   # z across cells
Z["phenotype"]=fac["phenotype"].values
M=Z.groupby("phenotype")[F].mean()
order=["CD8 TRM","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]
M=M.reindex([o for o in order if o in M.index])
print("\nmean z-scored projected factor per phenotype (top factor & its ref-celltype):")
for p in M.index:
    tf=M.loc[p].idxmax(); print(f"  {p:14} -> {tf} = {toptype[tf]}  (z={M.loc[p,tf]:+.2f})")

fig,ax=plt.subplots(figsize=(11,4.2))
im=ax.imshow(M.values,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
ax.set_xticks(range(len(F))); ax.set_xticklabels([f"{f}\n{toptype[f]}" for f in F],fontsize=7,rotation=45,ha="right")
ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index)
for i in range(len(M.index)):
    j=int(np.argmax(M.values[i])); ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="black",lw=2))
fig.colorbar(im,ax=ax,shrink=0.7,label="mean z projected factor")
ax.set_title("Existing T phenotypes vs abl5197 reference factors (projection)\nblack box = top factor per phenotype",fontsize=10,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"projection_validation_heatmap.png",dpi=130,bbox_inches="tight"); plt.close()
M.to_csv(NEW/"projection_validation_matrix.csv")
print("\nSaved: projection_validation_heatmap.png + projection_validation_matrix.csv")
