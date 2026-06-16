"""
313_Tmic_LR.py
T cell <-> microglia spatial ligand-receptor interaction analysis.
For each panel-present L-R pair: count spatial contacts where a ligand+ sender cell is within
30um of a receptor+ receiver cell, vs a permutation null (shuffle ligand labels within the
sender cell type per section). Reports fold-enrichment, z, empirical p. Both directions.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
plt.rcParams.update({"font.size":9}); rng=np.random.default_rng(0)
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0; NPERM=500
# candidate pairs: (ligand, ligand-source, receptor, receptor-target, label)
CAND=[("CXCL16","Mic","CXCR6","T","Mic->T retention"),
      ("CCL2","Mic","CCR2","T","Mic->T chemotaxis"),
      ("CD86","Mic","CD28","T","Mic->T costim"),
      ("CD86","Mic","CTLA4","T","Mic->T inhibitory"),
      ("IL15","Mic","IL2RB","T","Mic->T survival"),
      ("HLA-DRA","Mic","CD4","T","Mic->T MHC-II/CD4"),
      ("CD40LG","T","CD40","Mic","T->Mic CD40L"),
      ("CSF1","T","CSF1R","Mic","T->Mic CSF1"),
      ("LTB","T","LTBR","Mic","T->Mic lymphotoxin"),
      ("CCL5","T","CCR1","Mic","T->Mic CCL5"),
      ("FASLG","T","FAS","Mic","T->Mic Fas"),
      ("TNF","T","TNFRSF1A","Mic","T->Mic TNF"),
      ("IFNG","T","IFNGR1","Mic","T->Mic IFNg"),
      ("ICAM1","Mic","ITGAL","T","Mic->T adhesion")]

lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
Tids=set(lab[~lab.isin(["NK"])].index)   # T cells = CD8+CD4 subsets
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"]))
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}; vs=set(var)
is_T=np.array([c in Tids for c in idx]); is_Mic=(v2=="Mic")
print("panel L-R availability:")
pairs=[]
for lg,ls,rc,rt,desc in CAND:
    ok=(lg in vs) and (rc in vs); print(f"  {desc:22} {lg}->{rc}: ligand={lg in vs} receptor={rc in vs} {'OK' if ok else 'SKIP'}")
    if ok: pairs.append((lg,ls,rc,rt,desc))

# coords + run
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan); run=np.array(["?"]*len(idx),dtype=object)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); run[i]=pre; mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx)
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel()

def coloc(lg,ls,rc,rt):
    lig=E(lg); rec=E(rc)
    src_mask=(is_Mic if ls=="Mic" else is_T)&hasxy
    rcv_mask=(is_Mic if rt=="Mic" else is_T)&hasxy&(rec>0)   # receptor+ receiver
    obs=0; null=np.zeros(NPERM)
    for r in np.unique(run):
        s=np.where(src_mask&(run==r))[0]; rv=np.where(rcv_mask&(run==r))[0]
        if len(s)<5 or len(rv)<3: continue
        tree=cKDTree(np.column_stack([mx[s],my[s]]))
        nbr=tree.query_ball_point(np.column_stack([mx[rv],my[rv]]),r=R)
        lh=(lig[s]>0).astype(int)
        obs+=sum(lh[h].sum() for h in nbr)
        for k in range(NPERM):
            lp=rng.permutation(lh); null[k]+=sum(lp[h].sum() for h in nbr)
    mu=null.mean(); sd=null.std()+1e-9; return obs,mu,(obs-mu)/sd,(np.sum(null>=obs)+1)/(NPERM+1)

rows=[]
print(f"\nL-R colocalization (contacts <=30um vs permuted null, n_perm={NPERM}):")
for lg,ls,rc,rt,desc in pairs:
    obs,mu,z,p=coloc(lg,ls,rc,rt)
    fold=obs/(mu+1e-9); rows.append(dict(pair=f"{lg}->{rc}",desc=desc,obs=obs,null=round(mu,1),fold=round(fold,2),z=round(z,1),p=p))
    print(f"  {desc:22} {lg:7}->{rc:8} obs={obs:5} null={mu:7.1f} fold={fold:.2f} z={z:+.1f} p={p:.3f}")
Rdf=pd.DataFrame(rows).sort_values("z",ascending=False); Rdf.to_csv(NEW/"Tmic_LR_coloc.csv",index=False)

# figure
fig,ax=plt.subplots(figsize=(9,5)); R2=Rdf.iloc[::-1]; y=np.arange(len(R2))
cols=["#c0392b" if p<0.05 else "#bbb" for p in R2.p]
ax.barh(y,R2.fold,color=cols,edgecolor="#333",lw=0.4)
ax.axvline(1,color="#333",ls="--",lw=0.8)
for yi,(_,r) in zip(y,R2.iterrows()): ax.text(r.fold+0.02,yi,f"z={r.z:+.1f} p={r.p:.3f}",va="center",fontsize=8)
ax.set_yticks(y); ax.set_yticklabels([f"{r.desc}\n{r.pair}" for _,r in R2.iterrows()],fontsize=8)
ax.set_xlabel("fold-enrichment of ligand⁺ sender ↔ receptor⁺ receiver contacts vs null"); ax.set_xlim(0,max(R2.fold)*1.2)
ax.set_title("T cell ↔ microglia ligand-receptor spatial colocalization (red = p<0.05)",fontsize=10,fontweight="bold")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"Tmic_LR_coloc.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: Tmic_LR_coloc.png + Tmic_LR_coloc.csv")
