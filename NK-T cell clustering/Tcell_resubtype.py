"""
300_Tcell_resubtype.py
New T-cell pipeline (user spec):
 1. Hard-separate CD4 / CD8 / NK by markers (T=CD3+; CD8=CD8+CD4-; CD4=CD4+CD8-; NK=CD3-CD8B-NK+).
 2. Run NEIGHBORHOOD-AUGMENTED scHPF on CD4 and CD8 SEPARATELY (append neighbor cell-type
    composition as extra features so gene spillover is absorbed by niche factors).
 3. Use intrinsic (non-neighbor) factors -> Leiden subclusters; annotate with CellTypist (Immune_All_Low).
Output per lineage: factors, subclusters, CellTypist labels, UMAP. + lineage assignment table.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, schpf, celltypist, matplotlib.pyplot as plt
from celltypist import models
from scipy.sparse import csr_matrix, coo_matrix, hstack
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); LAB=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0; NBTYPES=["Mic","Mono/Mac","Oli","Ast","Exc","Inh","OPC","End","Per","SMC","B"]

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float64)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel() if gn in vp else np.zeros(X.shape[0])
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
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

# ---- 1. hard separation within T/NK compartment ----
ist=np.where((v2=="T/NK")&hasxy)[0]
cd3=E("CD3D")[ist]+E("CD3E")[ist]+E("CD3G")[ist]; cd8=E("CD8A")[ist]+E("CD8B")[ist]; cd8b=E("CD8B")[ist]; cd4=E("CD4")[ist]
nk=E("NKG7")[ist]+E("GNLY")[ist]+E("KLRD1")[ist]+E("KLRF1")[ist]+E("FCGR3A")[ist]
ph=pd.concat([pd.read_csv(LAB/"schpf_CD8_final_labels.csv")[["cell_id","phenotype"]],
              pd.read_csv(LAB/"schpf_CD4_final_labels.csv")[["cell_id","phenotype"]]]).set_index("cell_id")["phenotype"].reindex(idx[ist]).values
lin=np.array(["unassigned"]*len(ist),dtype=object)
isT=cd3>=1
lin[isT&(cd8>=1)&(cd4==0)]="CD8"; lin[isT&(cd4>=1)&(cd8==0)]="CD4"
dp=isT&(cd8>=1)&(cd4>=1); lin[dp]=np.where(cd8[dp]>=cd4[dp],"CD8","CD4")
lin[(cd3==0)&(cd8b==0)&(nk>=1)]="NK"
# DN CD3+ T (no CD4/CD8 detected): rescue via existing phenotype lineage
dn=isT&(cd8==0)&(cd4==0)
for i in np.where(dn)[0]:
    p=ph[i]
    if isinstance(p,str) and p.startswith("CD8"): lin[i]="CD8"
    elif isinstance(p,str) and p.startswith("CD4"): lin[i]="CD4"
print("hard-rule lineage:",pd.Series(lin).value_counts().to_dict())
lindf=pd.DataFrame({"cell_id":idx[ist],"lineage":lin,"existing_phenotype":ph}); lindf.to_csv(NEW/"Tcell_lineage_assignment.csv",index=False)

# neighbor composition for all T/NK cells (reused per lineage)
NB=np.zeros((len(ist),len(NBTYPES))); tpos={gi:k for k,gi in enumerate(ist)}
for r in np.unique(run):
    tk=[gi for gi in ist if run[gi]==r]
    if not tk: continue
    rows=[tpos[gi] for gi in tk]; txy=np.column_stack([mx[tk],my[tk]]); allsel=np.where((run==r)&hasxy)[0]
    tree=cKDTree(np.column_stack([mx[allsel],my[allsel]]))
    for rr,h in zip(rows,tree.query_ball_point(txy,r=R)):
        vt=v2[allsel[h]]
        for j,t in enumerate(NBTYPES): NB[rr,j]=np.sum(vt==t)

ctmodel=models.Model.load("Immune_All_Low.pkl")
def run_lineage(name,K):
    sel=np.where(lin==name)[0]
    if len(sel)<40: print(f"{name}: n={len(sel)} too few"); return None
    gi=[ist[s] for s in sel]
    Xg=X[gi].tocsr(); det=np.asarray((Xg>0).sum(0)).ravel()
    gk=[j for j,gn in enumerate(var) if det[j]>=8 and not gn.startswith("Blank")]; genes=[var[j] for j in gk]; Xg=Xg[:,gk]
    NBc=csr_matrix(np.round(NB[sel]*3).astype(np.float64))
    Xaug=hstack([Xg,NBc]).tocoo(); feat=genes+[f"NB_{t}" for t in NBTYPES]
    m=schpf.scHPF(nfactors=K,verbose=False)
    try: m.verbose=False
    except Exception: pass
    m.fit(Xaug)
    GS=pd.DataFrame(m.gene_score(),index=feat,columns=[f"F{i}" for i in range(K)]); CS=m.cell_score()
    nbw=GS.loc[[f"NB_{t}" for t in NBTYPES]].sum(); gw=GS.loc[genes].sum(); ratio=nbw/(gw+1e-9)
    intrinsic=[f for f in GS.columns if ratio[f]<0.05]
    print(f"\n{name}: n={len(sel)} | scHPF K={K} | intrinsic factors={intrinsic} (spillover dropped: {[f for f in GS.columns if f not in intrinsic]})")
    # leiden on intrinsic factor scores
    A=ad.AnnData(X=Xg,obs=pd.DataFrame({"cell_id":idx[gi]},index=idx[gi]),var=pd.DataFrame(index=genes))
    A.obsm["X_schpf"]=CS[:,[int(f[1:]) for f in intrinsic]]
    sc.pp.neighbors(A,use_rep="X_schpf",n_neighbors=15); sc.tl.leiden(A,resolution=0.6,flavor="igraph",n_iterations=2,directed=False); sc.tl.umap(A,min_dist=0.3,random_state=0)
    # ---- decontaminate via intrinsic-factor reconstruction (remove neighbor-spillover factors) ----
    Te=m.theta.e_x; Be=m.beta.e_x; Bg=Be[:len(genes),:]; ii=[int(f[1:]) for f in intrinsic]
    full=Te@Bg.T; intr=Te[:,ii]@Bg[:,ii].T; keep=np.clip(intr/(full+1e-9),0,1)
    Xclean=csr_matrix(Xg.multiply(keep))
    print(f"  decontam: mean kept fraction={keep.mean():.2f} (spillover removed where neighbor factors dominated)")
    # CellTypist annotate on DECONTAMINATED log-norm expression
    Act=ad.AnnData(X=Xclean,var=pd.DataFrame(index=genes),obs=A.obs.copy()); sc.pp.normalize_total(Act,target_sum=1e4); sc.pp.log1p(Act)
    pred=celltypist.annotate(Act,model=ctmodel,majority_voting=True)
    A.obs["celltypist"]=pred.predicted_labels["majority_voting"].values if "majority_voting" in pred.predicted_labels else pred.predicted_labels["predicted_labels"].values
    A.obs["existing_phenotype"]=ph[sel]
    A.obs[["cell_id","leiden","celltypist","existing_phenotype"]].to_csv(NEW/f"Tcell_{name}_subtype.csv",index=False)
    pd.DataFrame(CS,index=idx[gi],columns=GS.columns).to_csv(NEW/f"Tcell_{name}_schpf_scores.csv")
    print(f"  Leiden clusters: {A.obs.leiden.nunique()}")
    print("  cluster x CellTypist (top per cluster):")
    for cl in A.obs.leiden.cat.categories:
        ct=A.obs.celltypist[A.obs.leiden==cl].mode().iloc[0]; ex=A.obs.existing_phenotype[A.obs.leiden==cl].mode().iloc[0]
        print(f"    cl{cl} (n={int((A.obs.leiden==cl).sum())}): CellTypist={ct} | existing={ex}")
    print("  CellTypist label distribution:",A.obs.celltypist.value_counts().head(8).to_dict())
    return A

A8=run_lineage("CD8",8); A4=run_lineage("CD4",8)
# UMAP figure
fig,axes=plt.subplots(1,2,figsize=(15,6))
for ax,A,nm in zip(axes,[A8,A4],["CD8","CD4"]):
    if A is None: continue
    U=A.obsm["X_umap"]; cl=A.obs.celltypist.astype("category")
    for c in cl.cat.categories:
        m=(cl==c).values; ax.scatter(U[m,0],U[m,1],s=16,label=f"{c} ({m.sum()})",linewidths=0,alpha=0.8)
    ax.set_title(f"{nm} T cells — neighborhood-scHPF + CellTypist",fontsize=10,fontweight="bold"); ax.legend(fontsize=7,markerscale=2)
    ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ("top","right")]
plt.tight_layout(); fig.savefig(NEW/"Tcell_resubtype_umap.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: Tcell_lineage_assignment.csv, Tcell_{CD8,CD4}_subtype.csv, *_schpf_scores.csv, Tcell_resubtype_umap.png")
