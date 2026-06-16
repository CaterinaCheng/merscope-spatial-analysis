"""
359_loose_spatial_all.py
Redo ALL spatial analyses on the LOOSENED microglia set (358) for power:
 (A) vascular compartment composition + peri-vs-paren enrichment (Fisher,BH)
 (B) per-cluster compartment composition
 (C) compartment DEGs (peri/adj/paren one-vs-rest Wilcoxon) w/ cross-lineage spillover flag
 (D) microglia state enrichment near EACH T-cell subset (<=30um) vs baseline (Fisher,BH)
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact
import warnings; warnings.filterwarnings("ignore"); sc.settings.verbosity=0
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
VESSEL=["End","Per","SMC"]; STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
CCOL={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
SPILL=set(("AQP4 GJA1 SLC1A3 SLC1A2 GLUL GFAP AQP9 VCAN PLPP3 MOG MAL PLP1 MOBP CNP MBP UGT8 OLIG2 CD22 SORT1 "
           "SNAP25 SYT1 RBFOX3 GAD1 MEG3 NRGN SLC17A7 NELL2 BCL11B PECAM1 CLDN5 VWF PDGFRB ACTA2 RGS5 COL1A1 COL3A1 FN1 A2M FLT1 "
           "CD3D CD3E IL7R CD8A CXCR6 NKG7 MS4A1 SKAP1 CCL5 SPIB LIME1 NCR3").split())
co=pd.read_csv(NEW/"microglia_loose_coords.csv",index_col=0); co=co[~co.cluster_flag]
co=co[co.state!="Mixed/low"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_ves=np.isin(v2,VESSEL); is_tnk=(v2=="T/NK"); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan)
for i,c in enumerate(idx):
    if "_" in c: pre,eid=c.rsplit("_",1); mx[i],my[i]=ent.get((pre,eid),(np.nan,np.nan))
hasxy=np.isfinite(mx); mpos={c:i for i,c in enumerate(idx)}; mi=np.array([mpos[c] for c in co.index])
dV=np.full(len(mi),np.inf); near={s:np.zeros(len(mi),bool) for s in SUBS}; anyT=np.zeros(len(mi),int)
for r in np.unique(run[mi]):
    sel=np.where(run[mi]==r)[0]; mxy=np.column_stack([mx[mi[sel]],my[mi[sel]]]); ok=np.isfinite(mxy[:,0])
    if ok.sum()==0: continue
    vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(mxy[ok],k=1); tmp=np.full(len(sel),np.inf); tmp[ok]=dd; dV[sel]=tmp
    for s in SUBS:
        ss=np.where((labv==s)&(run==r)&hasxy)[0]
        if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy[ok],k=1); near[s][sel[ok]]=d<=30
    alls=np.where(is_tnk&(run==r)&hasxy)[0]
    if len(alls): d3,_=cKDTree(np.column_stack([mx[alls],my[alls]])).query(mxy[ok],k=1); anyT[sel[ok]]=(d3<=30).astype(int)
comp=np.where(dV<=30,"perivascular",np.where(dV<100,"vessel-adjacent",np.where(np.isfinite(dV),"parenchymal","n/a")))
co=co.assign(comp=comp); co=co[co.comp!="n/a"]
state=co.state.values; comps=["perivascular","vessel-adjacent","parenchymal"]
print(f"loose validated microglia w/ compartment: {len(co)}")
def fis(a1,n1,a2,n2):
    orr,p=fisher_exact([[a1,n1-a1],[a2,n2-a2]]); return np.log2(((a1/n1)+1e-6)/((a2/max(n2,1))+1e-6)),p
def bh(ps): o=np.argsort(ps); rk=np.empty(len(ps),int); rk[o]=np.arange(1,len(ps)+1); return np.minimum(np.array(ps)*len(ps)/rk,1)
# (A) compartment composition + enrichment
print("\n=== (A) compartment composition ===")
for c in comps: sub=co[co.comp==c]; print(f"  {c:15}(n={len(sub)}):",{s:f"{100*(sub.state==s).mean():.0f}%" for s in STATEORD})
rowsA=[]
for st in STATEORD:
    a1=int(((co.comp=="perivascular")&(co.state==st)).sum()); n1=int((co.comp=="perivascular").sum())
    a2=int(((co.comp=="parenchymal")&(co.state==st)).sum()); n2=int((co.comp=="parenchymal").sum())
    l,p=fis(a1,n1,a2,n2); rowsA.append(dict(state=st,f_peri=100*a1/n1,f_paren=100*a2/n2,log2=l,p=p))
RA=pd.DataFrame(rowsA); RA["padj"]=bh(RA.p.values); RA.to_csv(NEW/"loose_compartment_enrichment.csv",index=False)
print("=== peri vs paren (Fisher,BH) ==="); [print(f"  {r.state:18}: peri {r.f_peri:4.1f}% paren {r.f_paren:4.1f}% log2={r.log2:+.2f} padj={r.padj:.2g}{' *' if r.padj<0.05 else ''}") for _,r in RA.iterrows()]
# (B) per-cluster compartment
rowsB=[]
for cl in sorted(co.leiden.unique(),key=int):
    m=co.leiden==cl; rowsB.append(dict(cl=cl,state=co.state[m].value_counts().index[0],n=int(m.sum()),**{c:round(100*(co.comp[m]==c).mean(),1) for c in comps}))
CB=pd.DataFrame(rowsB); CB.to_csv(NEW/"loose_cluster_compartment.csv",index=False)
# (C) compartment DEG
A=ad.AnnData(X=Xd[mi[np.isfinite(dV)]].copy() if False else Xd[np.array([mpos[c] for c in co.index])].copy(),obs=pd.DataFrame({"comp":co.comp.values},index=co.index),var=pd.DataFrame(index=var))
sc.pp.normalize_total(A,target_sum=None); sc.pp.log1p(A); sc.tl.rank_genes_groups(A,"comp",method="wilcoxon")
degs={}
for c in comps:
    r=sc.get.rank_genes_groups_df(A,group=c); r=r[(r.pvals_adj<0.05)&(r.logfoldchanges>0)].sort_values("logfoldchanges",ascending=False)
    r["spill"]=r.names.isin(SPILL); degs[c]=r
pd.concat([d.assign(comp=c) for c,d in degs.items()]).to_csv(NEW/"loose_compartment_DEG.csv",index=False)
print("\n=== (C) compartment DEG top (non-spillover real microglial) ===")
for c in comps: print(f"  {c}:",", ".join(degs[c][~degs[c].spill].head(10).names))
# (D) T-subset proximity
base=anyT==0; nbase=int(base.sum()); print(f"\n=== (D) baseline (no T/NK<=30): {nbase} ===")
L=np.full((len(SUBS),len(STATEORD)),np.nan); Praw=[]; rowsD=[]; Ncnt=[]
for i,s in enumerate(SUBS):
    nm=near[s]; nn=int(nm.sum()); Ncnt.append(nn)
    for j,st in enumerate(STATEORD):
        if nn>=10:
            a1=int((nm&(state==st)).sum()); a2=int((base&(state==st)).sum()); l,p=fis(a1,nn,a2,nbase); L[i,j]=l; Praw.append(p); rowsD.append((i,j,p))
RD=pd.DataFrame([dict(subset=SUBS[i],n=Ncnt[i],state=STATEORD[j],log2=L[i,j],p=p) for (i,j,p) in rowsD])
if len(RD): RD["padj"]=bh(RD.p.values)
RD.to_csv(NEW/"loose_states_around_Tsubsets.csv",index=False)
padj={(r.subset,r.state):r.padj for _,r in RD.iterrows()} if len(RD) else {}
print("n near each subset:",{s:Ncnt[i] for i,s in enumerate(SUBS)})
print(f"{'subset':16}"+"".join(f"{st[:9]:>11}" for st in STATEORD))
for i,s in enumerate(SUBS):
    cells="".join((f"{L[i,j]:+.2f}{'*' if padj.get((s,STATEORD[j]),1)<0.05 else ' '}".rjust(11)) if not np.isnan(L[i,j]) else f"{'-':>11}" for j in range(len(STATEORD)))
    print(f"{s:11}(n={Ncnt[i]:<4})"+cells)
# ================= FIGURE =================
fig=plt.figure(figsize=(17,9)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.1],hspace=0.35,wspace=0.25)
# A composition
ax=fig.add_subplot(gs[0,0]); bottom=np.zeros(len(comps))
for st in STATEORD:
    vals=[100*(co[co.comp==c].state==st).mean() for c in comps]; ax.bar(range(len(comps)),vals,bottom=bottom,color=SCOL[st],label=st); bottom+=vals
ax.set_xticks(range(len(comps))); ax.set_xticklabels([f"{c}\n(n={int((co.comp==c).sum())})" for c in comps],fontsize=8); ax.set_ylabel("%"); ax.set_title("A. state composition by compartment",fontsize=10,fontweight="bold"); ax.legend(fontsize=7,ncol=2,loc="lower center",bbox_to_anchor=(0.5,-0.42))
# B peri vs paren
ax=fig.add_subplot(gs[0,1]); y=np.arange(len(STATEORD)); ax.barh(y,RA.log2,color=[SCOL[s] for s in RA.state]); ax.axvline(0,color="k",lw=.8)
for i,r in RA.iterrows(): ax.text(r.log2,i,(" *" if r.padj<0.05 else " ns"),va="center",ha="left" if r.log2>=0 else "right",fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(RA.state); ax.invert_yaxis(); ax.set_xlabel("log2(peri/paren)"); ax.set_title("B. perivascular vs parenchymal",fontsize=10,fontweight="bold")
for sp in ax.spines.values(): sp.set_visible(False)
# D T-subset heatmap
ax=fig.add_subplot(gs[1,:]); M=np.ma.masked_invalid(L); vmax=np.nanmax(np.abs(L)) if np.isfinite(L).any() else 1
im=ax.imshow(M,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
ax.set_xticks(range(len(STATEORD))); ax.set_xticklabels(STATEORD,rotation=20,ha="right"); ax.set_yticks(range(len(SUBS))); ax.set_yticklabels([f"{s} (n={Ncnt[i]})" for i,s in enumerate(SUBS)])
for i in range(len(SUBS)):
    for j in range(len(STATEORD)):
        if np.isnan(L[i,j]): ax.text(j,i,"n/a",ha="center",va="center",fontsize=8,color="#999"); continue
        ax.text(j,i,f"{L[i,j]:+.2f}"+("*" if padj.get((SUBS[i],STATEORD[j]),1)<0.05 else ""),ha="center",va="center",fontsize=8,color="white" if abs(L[i,j])>vmax*.6 else "#222")
ax.set_title("D. microglia state enrichment near each T-cell subset  [log2(near/baseline); * padj<0.05; n/a if <10]",fontsize=10,fontweight="bold"); fig.colorbar(im,ax=ax,shrink=0.7,label="log2")
fig.suptitle(f"Loosened-gate microglia ({len(co)} cells) — spatial state analysis",fontsize=13,fontweight="bold")
fig.savefig(NEW/"loose_spatial_all.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: loose_spatial_all.png + loose_{compartment_enrichment,cluster_compartment,compartment_DEG,states_around_Tsubsets}.csv")
