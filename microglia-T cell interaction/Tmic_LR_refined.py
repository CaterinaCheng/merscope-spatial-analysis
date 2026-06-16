"""
314_Tmic_LR_refined.py
Refine the T<->microglia L-R analysis:
 A. Break down the signalling pairs by T SUBSET (which subset is the receiver).
 B. Use ACTIVATED (DAM/MHC-II-high) microglia as the sender vs all microglia.
Permutation-null colocalization (ligand+ sender within 30um of receptor+ receiver).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9}); rng=np.random.default_rng(0)
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
R=30.0; NPERM=500
DAM=["CD68","APOE","SPP1","TREM2","GPNMB","FTL","CST7","ITGAX","LPL"]; MHCII=["CIITA","HLA-DRA","HLA-DPA1","HLA-DQB1","CD74"]
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]
MIC_T_PAIRS=[("CXCL16","CXCR6","retention"),("CCL2","CCR2","chemotaxis"),("CD86","CD28","costim"),
             ("CD86","CTLA4","inhibitory"),("HLA-DRA","CD4","MHC-II/CD4")]

lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vp={gn:i for i,gn in enumerate(var)}; vs=set(var)
is_Mic=(v2=="Mic")
def E(gn): return np.asarray(X[:,vp[gn]].todense()).ravel()

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

# activated microglia (top tertile DAM+MHC)
mglob=np.where(is_Mic&hasxy)[0]
am=ad.AnnData(X=X[mglob].copy(),var=pd.DataFrame(index=var)); sc.pp.normalize_total(am,target_sum=1e4); sc.pp.log1p(am)
sc.tl.score_genes(am,[g for g in DAM if g in vp],score_name="DAM"); sc.tl.score_genes(am,[g for g in MHCII if g in vp],score_name="MHC")
zc=lambda v:(v-v.mean())/v.std(); act=0.5*zc(am.obs["DAM"].values)+0.5*zc(am.obs["MHC"].values)
act_mask=np.zeros(len(idx),bool); act_mask[mglob[act>=np.quantile(act,2/3)]]=True
allmic_mask=is_Mic&hasxy

def coloc(lig_gene,sender_mask,recv_mask):
    lig=E(lig_gene); obs=0; null=np.zeros(NPERM)
    for r in np.unique(run):
        s=np.where(sender_mask&(run==r)&hasxy)[0]; rv=np.where(recv_mask&(run==r)&hasxy)[0]
        if len(s)<5 or len(rv)<3: continue
        nbr=cKDTree(np.column_stack([mx[s],my[s]])).query_ball_point(np.column_stack([mx[rv],my[rv]]),r=R)
        lh=(lig[s]>0).astype(int); obs+=sum(lh[h].sum() for h in nbr)
        for k in range(NPERM): null[k]+=sum(rng.permutation(lh)[h].sum() for h in nbr)
    mu=null.mean(); sd=null.std()+1e-9; return obs,mu,(obs-mu)/sd,(np.sum(null>=obs)+1)/(NPERM+1)

# ===== Part A: by T subset (sender = all microglia) for CCL2->CCR2 and CD86->CTLA4 =====
print("=== PART A: L-R by T subset (sender = all microglia) ===")
rowsA=[]
for lig,rec,desc in [("CCL2","CCR2","chemotaxis"),("CD86","CTLA4","inhibitory")]:
    recE=E(rec)
    print(f"\n{lig}->{rec} ({desc}):")
    for s in SUBS:
        rmask=(lab.reindex(idx).values==s)&(recE>0)
        nrec=int((rmask&hasxy).sum()); pctpos=100*((recE>0)[lab.reindex(idx).values==s]).mean()
        if nrec<3: print(f"   {s:14}: {rec}+ n={nrec} -> too few"); continue
        obs,mu,z,p=coloc(lig,allmic_mask,rmask)
        rowsA.append(dict(pair=f"{lig}->{rec}",subset=s,rec_pos=nrec,rec_pct=round(pctpos,0),fold=round(obs/(mu+1e-9),2),z=round(z,1),p=p))
        print(f"   {s:14}: {rec}+ n={nrec} ({pctpos:.0f}%)  fold={obs/(mu+1e-9):.2f} z={z:+.1f} p={p:.3f}")
pd.DataFrame(rowsA).to_csv(NEW/"Tmic_LR_bysubset.csv",index=False)

# ===== Part B: activated vs all microglia as sender (all Mic->T pairs) =====
print("\n=== PART B: sender = ALL microglia vs ACTIVATED (DAM/MHC-II-high) microglia ===")
T_all=lab.reindex(idx).isin(SUBS).values
rowsB=[]
for lig,rec,desc in MIC_T_PAIRS:
    if lig not in vs or rec not in vs: continue
    rmask=T_all&(E(rec)>0)
    oa,ma,za,pa=coloc(lig,allmic_mask,rmask); ob,mb,zb,pb=coloc(lig,act_mask,rmask)
    rowsB.append(dict(pair=f"{lig}->{rec}",desc=desc,fold_allMic=round(oa/(ma+1e-9),2),z_allMic=round(za,1),p_allMic=pa,
                      fold_actMic=round(ob/(mb+1e-9),2),z_actMic=round(zb,1),p_actMic=pb))
    print(f"  {desc:12} {lig:7}->{rec:7}: allMic fold={oa/(ma+1e-9):.2f}(z={za:+.1f},p={pa:.3f})   actMic fold={ob/(mb+1e-9):.2f}(z={zb:+.1f},p={pb:.3f})")
B=pd.DataFrame(rowsB); B.to_csv(NEW/"Tmic_LR_activated.csv",index=False)

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
A=pd.DataFrame(rowsA)
if len(A):
    piv=A.pivot_table(index="subset",columns="pair",values="fold").reindex([s for s in SUBS if s in set(A.subset)])
    piv.plot(kind="barh",ax=ax1,color=["#c0392b","#e67e22"],edgecolor="#333",lw=0.3)
    ax1.axvline(1,color="#333",ls="--",lw=0.8); ax1.set_xlabel("fold vs null"); ax1.set_title("A. L-R by T subset (sender=all microglia)",fontsize=10,fontweight="bold"); ax1.legend(fontsize=8)
y=np.arange(len(B)); w=0.38
ax2.barh(y+w/2,B.fold_allMic,w,color="#7f8c8d",edgecolor="#333",lw=0.3,label="all microglia")
ax2.barh(y-w/2,B.fold_actMic,w,color="#c0392b",edgecolor="#333",lw=0.3,label="activated (DAM/MHC-II) microglia")
for yi,(_,r) in zip(y,B.iterrows()):
    ax2.text(r.fold_actMic+0.02,yi-w/2,f"p={r.p_actMic:.3f}",va="center",fontsize=7)
ax2.axvline(1,color="#333",ls="--",lw=0.8); ax2.set_yticks(y); ax2.set_yticklabels([f"{r.desc}\n{r.pair}" for _,r in B.iterrows()],fontsize=8)
ax2.set_xlabel("fold vs null"); ax2.set_title("B. Sender: all vs activated microglia",fontsize=10,fontweight="bold"); ax2.legend(fontsize=8)
for ax in (ax1,ax2):
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"Tmic_LR_refined.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: Tmic_LR_refined.png + Tmic_LR_bysubset.csv + Tmic_LR_activated.csv")
