"""
372_neighborhood_interactions.py
Cell-cell spatial co-localization (neighborhood enrichment, squidpy-style) within each
vascular compartment (PVS <=30um vs parenchyma >100um). For all major cell types:
 z = (observed A-B contacts within 30um - permuted null) / null sd  (labels shuffled within run).
Highlights T-astro, T-neuron, microglia-neuron, microglia-astro, T-microglia.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial import cKDTree
import warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
VESSEL=["End","Per","SMC"]; RADIUS=30; NPERM=200; rng=np.random.RandomState(0)
GRP={"Exc":"Neuron","Inh":"Neuron","Ast":"Astro","Oli":"Oligo","OPC":"OPC","Mic":"Microglia",
     "T/NK":"T/NK","Mono/Mac":"Mono/Mac","B":"B","End":"Vessel","Per":"Vessel","SMC":"Vessel"}
TYPES=["Neuron","Astro","Oligo","OPC","Microglia","T/NK","Mono/Mac","B","Vessel"]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
grp=np.array([GRP.get(x,"other") for x in v2],dtype=object)
is_ves=np.isin(v2,VESSEL); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
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
# compartment for ALL cells: distance to nearest vessel (per run)
dV=np.full(len(idx),np.inf)
for r in np.unique(run):
    cs=np.where((run==r)&hasxy)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs) and len(cs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[cs],my[cs]]),k=1); dV[cs]=dd
comp=np.where(dV<=30,"PVS",np.where(dV<=100,"adj",np.where(np.isfinite(dV),"parenchyma","na")))
tcode={t:i for i,t in enumerate(TYPES)}
def onehot(codes,n):
    m=codes>=0; r=np.where(m)[0]; return csr_matrix((np.ones(m.sum()),(r,codes[m])),shape=(n,len(TYPES)))
def enrich(compartment):
    sel=np.where((comp==compartment)&hasxy&np.isin(grp,TYPES))[0]
    # build adjacency within run
    rows=[];cols=[];blocks=[]; codes=np.array([tcode[g] for g in grp[sel]]); off=0; pos={gi:k for k,gi in enumerate(sel)}
    order=[]
    for r in np.unique(run[sel]):
        ii=sel[run[sel]==r]; loc=np.array([pos[i] for i in ii])
        tree=cKDTree(np.column_stack([mx[ii],my[ii]])); pairs=tree.query_pairs(RADIUS,output_type="ndarray")
        if len(pairs): rows+=list(loc[pairs[:,0]]); cols+=list(loc[pairs[:,1]])
        blocks.append((loc.min() if len(loc) else 0,loc.max()+1 if len(loc) else 0,loc))
    N=len(sel); A=coo_matrix((np.ones(len(rows)*2),(rows+cols,cols+rows)),shape=(N,N)).tocsr()
    L=onehot(codes,N); Cobs=np.asarray((L.T@A@L).todense())
    perm=np.zeros((NPERM,len(TYPES),len(TYPES)))
    for p in range(NPERM):
        cp=codes.copy()
        for r in np.unique(run[sel]):
            bidx=np.where(run[sel]==r)[0]; cp[bidx]=rng.permutation(cp[bidx])
        Lp=onehot(cp,N); perm[p]=np.asarray((Lp.T@A@Lp).todense())
    mu=perm.mean(0); sd=perm.std(0)+1e-9; Z=(Cobs-mu)/sd
    return Z,len(sel)
ZP,nP=enrich("PVS"); ZA,nA=enrich("parenchyma")
print(f"PVS cells: {nP}; parenchyma cells: {nA}")
pairs=[("T/NK","Astro"),("T/NK","Neuron"),("Microglia","Neuron"),("Microglia","Astro"),("T/NK","Microglia"),("T/NK","Vessel"),("Microglia","Vessel")]
print("\n=== neighborhood enrichment z (within 30um) ===")
print(f"{'pair':26}{'PVS':>8}{'parenchyma':>12}")
for a,b in pairs:
    i,j=tcode[a],tcode[b]; print(f"{a+'-'+b:26}{ZP[i,j]:>8.1f}{ZA[i,j]:>12.1f}")
pd.DataFrame(ZP,index=TYPES,columns=TYPES).to_csv(NEW/"nhood_enrichment_PVS.csv")
pd.DataFrame(ZA,index=TYPES,columns=TYPES).to_csv(NEW/"nhood_enrichment_parenchyma.csv")
# ===== figure: two heatmaps =====
fig,axes=plt.subplots(1,2,figsize=(15,6.2))
vmax=np.nanpercentile(np.abs(np.concatenate([ZP.ravel(),ZA.ravel()])),99)
for ax,(Z,title,nn) in zip(axes,[(ZP,"PVS (perivascular)",nP),(ZA,"Parenchyma",nA)]):
    im=ax.imshow(Z,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TYPES,rotation=40,ha="right"); ax.set_yticks(range(len(TYPES))); ax.set_yticklabels(TYPES)
    for i in range(len(TYPES)):
        for j in range(len(TYPES)): ax.text(j,i,f"{Z[i,j]:.0f}",ha="center",va="center",fontsize=6.5,color="white" if abs(Z[i,j])>vmax*0.55 else "#333")
    ax.set_title(f"{title}  (n={nn} cells)\nneighborhood enrichment z (red=co-localize, blue=avoid)",fontsize=10,fontweight="bold")
    fig.colorbar(im,ax=ax,shrink=0.7,label="z")
plt.tight_layout(); fig.savefig(NEW/"nhood_interactions_by_compartment.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: nhood_interactions_by_compartment.png + nhood_enrichment_{PVS,parenchyma}.csv")
