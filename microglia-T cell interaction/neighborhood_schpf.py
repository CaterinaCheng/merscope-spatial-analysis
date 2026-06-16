"""
295_neighborhood_schpf.py  (Stage 4: neighborhood-augmented scHPF to absorb spillover)
User idea: "add cell neighborhood as a factor since there are many gene spillover."
For each T cell, append neighbor cell-type-composition counts (within 30um) as extra
features, then train scHPF on [T-gene-counts | neighbor-counts]. Factors that load on the
neighbor block are NICHE/SPILLOVER factors; their co-loaded real genes = the leaked signature.
Demonstrates that known spillover genes (CX3CR1/S1PR5/...) attach to neighborhood factors.
Output: neighborhood_factor_loadings.png + neighborhood_schpf_factors.csv
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, schpf, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix, hstack
from scipy.spatial import cKDTree
plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0; K=12; NBTYPES=["Mic","Mono/Mac","Oli","Ast","Exc","Inh","OPC","End","Per","SMC","B"]
SPILL=["CX3CR1","S1PR5","FCGR3A","KLRF1","MOG","MAL","C1QA","C1QB","CSF1R","PLP1","AQP4","GFAP","SNAP25"]

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float64)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c:
        pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx)

ist=np.where((v2=="T/NK")&hasxy)[0]; tidx=idx[ist]
print(f"T/NK cells with coords: {len(ist)}")
# neighbor composition per T cell
NB=np.zeros((len(ist),len(NBTYPES))); tpos={gi:k for k,gi in enumerate(ist)}
for r in np.unique(run):
    tk=[gi for gi in ist if run[gi]==r]
    if not tk: continue
    rows=[tpos[gi] for gi in tk]; txy=np.column_stack([mx[tk],my[tk]])
    allsel=np.where((run==r)&hasxy)[0]
    tree=cKDTree(np.column_stack([mx[allsel],my[allsel]]))
    nbrs=tree.query_ball_point(txy,r=R)
    for rr,h in zip(rows,nbrs):
        vt=v2[allsel[h]]
        for j,t in enumerate(NBTYPES): NB[rr,j]=np.sum(vt==t)
print("mean neighbors within 30um:",{t:round(NB[:,j].mean(),1) for j,t in enumerate(NBTYPES)})

# gene matrix (filter genes detected in >=10 T cells), drop Blank
Xg=X[ist].tocsr(); det=np.asarray((Xg>0).sum(0)).ravel()
gkeep=[j for j,gn in enumerate(var) if det[j]>=10 and not gn.startswith("Blank")]
genes=[var[j] for j in gkeep]; Xg=Xg[:,gkeep]
# upweight neighbor features so scHPF allocates structure to them (x3)
NBc=csr_matrix(np.round(NB*3).astype(np.float64))
Xaug=hstack([Xg,NBc]).tocoo()
feat=genes+[f"NB_{t}" for t in NBTYPES]
print(f"augmented matrix: {Xaug.shape} ({len(genes)} genes + {len(NBTYPES)} neighbor features)")

m=schpf.scHPF(nfactors=K,verbose=False)
try: m.verbose=False
except Exception: pass
m.fit(Xaug)
GS=m.gene_score()  # (n_feat) x K
gsd=pd.DataFrame(GS,index=feat,columns=[f"F{i}" for i in range(K)])
nb_load=gsd.loc[[f"NB_{t}" for t in NBTYPES]]
# normalize neighbor loadings per factor (fraction of neighbor-block weight)
print("\nneighbor-feature loading per factor (which factors are NICHE/SPILLOVER factors):")
nbz=(nb_load-nb_load.values.mean())/nb_load.values.std()
for fc in gsd.columns:
    topnb=nb_load[fc].idxmax(); frac=nb_load[fc].max()/ (nb_load[fc].sum()+1e-9)
    # is this a neighborhood factor? neighbor block weight vs gene block
    nbw=nb_load[fc].sum(); gw=gsd.loc[genes,fc].sum(); ratio=nbw/(gw+1e-9)
    topg=gsd.loc[genes,fc].nlargest(8).index.tolist()
    flag="<-- NICHE/SPILLOVER" if ratio>0.05 else ""
    print(f"  {fc}: nbr/gene wt={ratio:.3f} topNB={topnb.replace('NB_','')} | topgenes: {', '.join(topg)} {flag}")
gsd.to_csv(NEW/"neighborhood_schpf_factors.csv")

# spillover demonstration: for known spillover genes, which factor do they load on, and does it co-load a neighbor type?
print("\nspillover-gene check (gene -> top factor -> that factor's top neighbor type):")
for sg in SPILL:
    if sg in gsd.index:
        tf=gsd.loc[sg].idxmax(); tnb=nb_load[tf].idxmax().replace("NB_","")
        print(f"  {sg:8} -> {tf} (top neighbor={tnb})")

# figure: factor x neighbor-type loading heatmap
fig,ax=plt.subplots(figsize=(9,4.5))
Z=nb_load.T  # factors x neighbortypes
im=ax.imshow(Z.values,cmap="viridis",aspect="auto")
ax.set_xticks(range(len(NBTYPES))); ax.set_xticklabels(NBTYPES,rotation=45,ha="right")
ax.set_yticks(range(K)); ax.set_yticklabels(gsd.columns)
fig.colorbar(im,ax=ax,shrink=0.7,label="neighbor-feature loading")
ax.set_title("Neighborhood-augmented scHPF: neighbor cell-type loading per factor\n(high = niche/spillover factor capturing that neighbor's leaked transcripts)",fontsize=10,fontweight="bold")
plt.tight_layout(); fig.savefig(NEW/"neighborhood_factor_loadings.png",dpi=130,bbox_inches="tight"); plt.close()
print("\nSaved: neighborhood_factor_loadings.png + neighborhood_schpf_factors.csv")
