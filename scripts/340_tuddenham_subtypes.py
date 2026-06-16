"""
340_tuddenham_subtypes.py
Map our microglia onto the Tuddenham 2024 (Nat Neurosci) living-human-microglia SUBTYPES
(discrete clusters, validated in situ by MERFISH). Markers from Suppl Table 2 (MOESM4
'Upregulated genes', MAST pairwise DE). Build panel-restricted discriminative signatures
(cross-lineage filtered), score decontam microglia, assign dominant subtype, UMAP + spatial.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); REF=NEW/"reference"
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
VESSEL=["End","Per","SMC"]; TOPN=15; MINGENES=6
# Tuddenham cluster -> functional label (paper Fig2/text)
CLNAME={1:"C1 Metabolic/Tx",6:"C6 Metabolic/Tx",7:"C7 Stress/Metab",2:"C2 Homeostatic",3:"C3 Homeostatic",
        4:"C4 Homeo-active",9:"C9 APOE/Homeo-act",5:"C5 Motility",8:"C8 Cytokine/IL",10:"C10 APC/Complement",
        11:"C11 DAM(GPNMB)",12:"C12 Proliferative",13:"C13 APC/HLA"}
BLOCK=set(("CD3D CD3E CD3G CD2 CD8A CD8B CD4 CD28 CD247 IL7R CXCR6 CCL5 LIME1 SKAP1 IL32 LCK THEMIS GZMK GZMA NKG7 GNLY KLRD1 KLRB1 KLRG1 ICOS FOXP3 CD40LG TBX21 CD7 "
            "CD19 MS4A1 CD79A CD79B JCHAIN BANK1 IGHM "
            "AQP4 SLC1A3 SLC1A2 GJA1 GLUL GFAP "
            "MOG MAL PLP1 MOBP CNP MBP UGT8 GLDN "
            "RBFOX3 SYT1 SNAP25 GAD1 RORB FOXP2 NRGN MEG3 XIST "
            "PECAM1 CLDN5 VWF ACTA2 PDGFRB RGS5 NOTCH3 COL1A1 COL3A1 DCN AHNAK RNASE1").split())
up=pd.read_excel(REF/"tuddenham/MOESM4.xlsx",sheet_name="Upregulated genes")
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
vset=set(var)
SIG={}
for cl in sorted(up.up_type.unique()):
    sub=up[up.up_type==cl].sort_values("sum_logFC",ascending=False)
    genes=[g for g in sub.gene if g in vset and g not in BLOCK][:TOPN]
    if len(genes)>=MINGENES: SIG[CLNAME[cl]]=genes
print(f"Tuddenham subtypes scoreable (>= {MINGENES} panel markers): {len(SIG)} of 13")
for k,v in SIG.items(): print(f"  {k:20}: {', '.join(v)}")
drop=[CLNAME[cl] for cl in sorted(up.up_type.unique()) if CLNAME[cl] not in SIG]; print("dropped (too few panel markers):",drop)
STATEORD=list(SIG.keys())
is_mic=(v2=="Mic"); is_ves=np.isin(v2,VESSEL); micidx=np.where(is_mic)[0]
A=ad.AnnData(X=Xd[micidx].copy(),obs=pd.DataFrame(index=idx[micidx]),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A)
for k,genes in SIG.items(): sc.tl.score_genes(A,[g for g in genes if g in A.var_names],score_name=k,ctrl_size=50)
Z=(A.obs[STATEORD]-A.obs[STATEORD].mean())/A.obs[STATEORD].std()
dom=Z.idxmax(1).values; dom[Z.max(1).values<0.1]="Mixed/low"; A.obs["subtype"]=dom
print("\ndominant subtype composition:",{k:round(100*(A.obs.subtype==k).mean(),1) for k in STATEORD+['Mixed/low'] if (A.obs.subtype==k).any()})
# QC: mean dominant-score (positive = real assignment)
print("median score of dominant assignment:",round(np.median([A.obs[A.obs.subtype.values==dom[i]].iloc[0][dom[i]] if False else Z.values[i].max() for i in range(0)] or [0]),2) if False else "")
A.obs[STATEORD+["subtype"]].to_csv(NEW/"microglia_tuddenham_labels.csv")
# spatial
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); dV=np.full(len(idx),np.inf); near={s:np.zeros(len(idx)) for s in SUBS}; anyT=np.zeros(len(idx))
for r in np.unique(run):
    cs=np.where(is_mic&(run==r)&hasxy)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs) and len(cs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[cs],my[cs]]),k=1); dV[cs]=dd
    ms=np.where(is_mic&(run==r)&hasxy)[0]
    if len(ms):
        mxy=np.column_stack([mx[ms],my[ms]])
        for s in SUBS:
            ss=np.where((labv==s)&(run==r)&hasxy)[0]
            if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy,k=1); near[s][ms]=(d<=30).astype(int)
        for s in SUBS: anyT[ms]+=near[s][ms]
comp=np.where(dV<=30,"perivascular",np.where(dV<100,"vessel-adjacent","parenchymal")); A.obs["comp"]=comp[micidx]
mh=hasxy[micidx]; baseM=(anyT[micidx]==0)&mh
def cliffs(a,b):
    n1,n2=len(a),len(b); rk=stats.rankdata(np.concatenate([a,b])); U=rk[:n1].sum()-n1*(n1+1)/2; return 2*U/(n1*n2)-1
par=A.obs.comp.values=="parenchymal"; rc=[]
for st in STATEORD:
    b=A.obs.loc[par,st].values
    for cg in ["perivascular","vessel-adjacent"]: a=A.obs.loc[A.obs.comp.values==cg,st].values; rc.append(dict(factor=st,grp=cg,cliffs=cliffs(a,b),p=stats.mannwhitneyu(a,b).pvalue))
RC=pd.DataFrame(rc); rt=[]
for s in SUBS:
    nm=(near[s][micidx]>=1)&mh; nc=int(nm.sum())
    for st in STATEORD:
        if nc>=5: a=A.obs.loc[nm,st].values; bb=A.obs.loc[baseM,st].values; d=cliffs(a,bb); p=stats.mannwhitneyu(a,bb).pvalue
        else: d,p=np.nan,np.nan
        rt.append(dict(subset=s,n=nc,factor=st,cliffs=d,p=p))
RT=pd.DataFrame(rt)
for R in (RC,RT):
    ok=R.p.notna(); pv=R.loc[ok,"p"].values; o=np.argsort(pv); rk=np.empty(len(pv),int); rk[o]=np.arange(1,len(pv)+1)
    R.loc[ok,"padj"]=np.minimum(np.minimum.accumulate((pv*len(pv)/rk)[o][::-1])[::-1][np.argsort(o)],1)
RC.to_csv(NEW/"tuddenham_compartment.csv",index=False); RT.to_csv(NEW/"tuddenham_Tsubset.csv",index=False)
# UMAP
co=pd.read_csv(NEW/"microglia_umap_coords.csv",index_col=0); U=co.reindex(A.obs_names)[["umap1","umap2"]].values
order=STATEORD+["Mixed/low"]; pal=cm.get_cmap("tab20",len(order))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,8))
for i,st in enumerate(order):
    m=A.obs.subtype.values==st
    if m.sum(): ax1.scatter(U[m,0],U[m,1],s=3,c=[pal(i)],label=f"{st} ({100*m.mean():.0f}%)",linewidths=0,rasterized=True)
ax1.set_title("Microglia UMAP — Tuddenham2024 living-microglia subtypes (decontam)",fontsize=10.5,fontweight="bold")
ax1.set_xticks([]); ax1.set_yticks([]); ax1.legend(markerscale=3,fontsize=7.5,loc="center left",bbox_to_anchor=(1.0,0.5),frameon=False)
for sp in ax1.spines.values(): sp.set_visible(False)
for c in ["parenchymal","vessel-adjacent","perivascular"]:
    m=A.obs.comp.values==c
    ax2.scatter(U[m,0],U[m,1],s=3,c={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}[c],label=c,linewidths=0,rasterized=True)
ax2.set_title("Vascular compartment",fontsize=10.5,fontweight="bold"); ax2.set_xticks([]); ax2.set_yticks([]); ax2.legend(markerscale=4,fontsize=8,frameon=False)
for sp in ax2.spines.values(): sp.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"microglia_umap_tuddenham_subtypes.png",dpi=150,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_umap_tuddenham_subtypes.png + tuddenham_{compartment,Tsubset}.csv + microglia_tuddenham_labels.csv")
