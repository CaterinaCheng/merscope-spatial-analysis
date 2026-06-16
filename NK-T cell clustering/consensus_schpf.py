"""
292b_consensus_schpf.py  (Stage 2: consensus scHPF reference on abl5197 T atlas, 494-gene space)
Guide-faithful: donor balancing + UMI downsampling -> gene filter -> multi-K x trials ->
walktrap consensus on factor gene-scores -> consensus K -> final run_trials model.
Saves reference model + gene list + gene/cell scores + factor x celltype table for projection.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, schpf, igraph, joblib
from scipy.sparse import coo_matrix, csr_matrix
sc.settings.verbosity=0
REFD=Path(r"D:\Caterina\MERSCOPE\reference"); A=sc.read_h5ad(REFD/"abl5197_T_panel494.h5ad")
rng=np.random.RandomState(0)
KGRID=[8,10,12,14,16,18,20]; NTRIAL_SCREEN=3; NTRIAL_FINAL=5; PERDONOR_CAP=5000; MIN_CELLS=10

# --- donor balancing ---
keep=[]
for dn,ix in A.obs.groupby("donor").indices.items():
    keep.append(rng.choice(ix,min(len(ix),PERDONOR_CAP),replace=False))
keep=np.sort(np.concatenate(keep)); A=A[keep].copy()
print(f"after donor balancing (cap {PERDONOR_CAP}/donor): {A.n_obs} cells from {A.obs.donor.nunique()} donors")

# --- gene filter ---
det=np.asarray((A.X>0).sum(0)).ravel(); gkeep=det>=MIN_CELLS; A=A[:,gkeep].copy()
genes=list(A.var_names); print(f"genes kept (>= {MIN_CELLS} cells): {len(genes)}")

# --- UMI downsample to median total (binomial thinning) ---
Xc=csr_matrix(A.X).astype(np.float64); tot=np.asarray(Xc.sum(1)).ravel(); target=np.median(tot)
rowidx=np.repeat(np.arange(Xc.shape[0]),np.diff(Xc.indptr)); p=np.minimum(1.0,target/np.maximum(tot,1))[rowidx]
Xc.data=rng.binomial(Xc.data.astype(int),p).astype(np.float64); Xc.eliminate_zeros()
print(f"UMI downsample to median={target:.0f}; nnz {Xc.nnz}")
Xcoo=coo_matrix(Xc)

def fit_one(K,seed):
    m=schpf.scHPF(nfactors=K,verbose=False);
    try: m.verbose=False
    except Exception: pass
    m.fit(Xcoo); return m

# --- multi-K x trials screen, collect factor gene-scores ---
print("screening K grid x trials...")
allfac=[]; best_per_K={}
for K in KGRID:
    best=None
    for t in range(NTRIAL_SCREEN):
        m=fit_one(K,t); gs=m.gene_score()  # G x K
        for k in range(K): allfac.append(gs[:,k])
        if best is None or m.loss[-1]<best.loss[-1]: best=m
    best_per_K[K]=best; print(f"  K={K} done; best loss={best.loss[-1]:.4f}")
Z=np.column_stack(allfac)  # G x totalfactors
print(f"total candidate factors: {Z.shape[1]}")

# --- walktrap consensus on factor-factor correlation graph ---
C=np.corrcoef(Z.T); np.fill_diagonal(C,0)
k_nn=10; edges=[]; weights=[]
for i in range(C.shape[0]):
    nn=np.argsort(C[i])[::-1][:k_nn]
    for j in nn:
        if C[i,j]>0: edges.append((i,int(j))); weights.append(float(C[i,j]))
g=igraph.Graph(n=C.shape[0],edges=edges,directed=False); g.es["weight"]=weights
comm=g.community_walktrap(weights="weight").as_clustering()
sizes=np.array([len(c) for c in comm]); minsize=max(5,NTRIAL_SCREEN)
kept=[c for c,s in zip(comm,sizes) if s>=minsize]
consensusK=len(kept)
print(f"walktrap communities: {len(comm)} ; kept (size>={minsize}): {consensusK}")

# --- final model at consensus K ---
print(f"training final consensus model at K={consensusK} (ntrials={NTRIAL_FINAL})...")
final=schpf.run_trials(Xcoo,nfactors=consensusK,ntrials=NTRIAL_FINAL,verbose=False)
schpf.save_model(final,str(REFD/"abl5197_T_schpf_consensus.joblib"))
pd.Series(genes).to_csv(REFD/"abl5197_T_schpf_genes.txt",index=False,header=False)
gs=final.gene_score(); cs=final.cell_score()
pd.DataFrame(gs,index=genes,columns=[f"F{i}" for i in range(consensusK)]).to_csv(REFD/"abl5197_T_schpf_gene_scores.csv")
# factor x celltype mean cell-score (interpretability)
cd=pd.DataFrame(cs,columns=[f"F{i}" for i in range(consensusK)]); cd["celltype"]=A.obs["celltype"].values
fct=cd.groupby("celltype").mean()
fct.to_csv(REFD/"abl5197_T_schpf_factor_by_celltype.csv")
print("consensus K =",consensusK)
print("\nfactor x celltype (mean cell-score, top celltype per factor):")
for f in fct.columns: print(f"  {f}: {fct[f].idxmax()} ({fct[f].max():.3f})")
# top genes per factor
print("\ntop genes per factor:")
gsd=pd.DataFrame(gs,index=genes,columns=fct.columns)
for f in fct.columns: print(f"  {f}: "+", ".join(gsd[f].nlargest(8).index))
print("\nSaved: abl5197_T_schpf_consensus.joblib + genes.txt + gene_scores.csv + factor_by_celltype.csv")
