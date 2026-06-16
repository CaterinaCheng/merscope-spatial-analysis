"""
357_states_around_Tsubsets.py
Clean microglia 5-state enrichment around EACH T-cell subset (within 30um) vs baseline
(microglia with no T/NK within 30um). Fisher per state, BH FDR. Reports n per subset (power).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact
import warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
co=pd.read_csv(NEW/"microglia_final_coords.csv",index_col=0); co=co[~co.cluster_flag]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_tnk=(v2=="T/NK"); run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object); labv=lab.reindex(idx).values
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
near={s:np.zeros(len(mi),bool) for s in SUBS}; anyT=np.zeros(len(mi),int)
for r in np.unique(run[mi]):
    sel=np.where(run[mi]==r)[0]; mxy=np.column_stack([mx[mi[sel]],my[mi[sel]]]); ok=np.isfinite(mxy[:,0])
    if ok.sum()==0: continue
    for s in SUBS:
        ss=np.where((labv==s)&(run==r)&hasxy)[0]
        if len(ss): d,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(mxy[ok],k=1); near[s][sel[ok]]=d<=30
    alls=np.where(is_tnk&(run==r)&hasxy)[0]
    if len(alls): d3,_=cKDTree(np.column_stack([mx[alls],my[alls]])).query(mxy[ok],k=1); anyT[sel[ok]]=(d3<=30).astype(int)
state=co.state.values; base=anyT==0; nbase=int(base.sum())
print(f"baseline microglia (no T/NK <=30um): {nbase}")
L=np.full((len(SUBS),len(STATEORD)),np.nan); P=np.ones((len(SUBS),len(STATEORD))); Ncnt=[]
rows=[]
for i,s in enumerate(SUBS):
    nm=near[s]; nn=int(nm.sum()); Ncnt.append(nn)
    for j,st in enumerate(STATEORD):
        if nn>=10:
            a1=int((nm&(state==st)).sum()); a2=int((base&(state==st)).sum())
            orr,p=fisher_exact([[a1,nn-a1],[a2,nbase-a2]]); L[i,j]=np.log2(((a1/nn)+1e-6)/((a2/nbase)+1e-6)); P[i,j]=p
            rows.append(dict(subset=s,n=nn,state=st,f_near=100*a1/nn,f_base=100*a2/nbase,log2=L[i,j],p=p))
R=pd.DataFrame(rows)
if len(R):
    o=np.argsort(R.p.values); rk=np.empty(len(R),int); rk[o]=np.arange(1,len(R)+1); R["padj"]=np.minimum(R.p*len(R)/rk,1)
    padj=R.set_index(["subset","state"]).padj
R.to_csv(NEW/"states_around_Tsubsets.csv",index=False)
print("\nn microglia near each subset:",{s:Ncnt[i] for i,s in enumerate(SUBS)})
print("\n=== state log2(near/baseline) per subset (* padj<0.05; blank = n<10) ===")
print(f"{'subset':14}"+"".join(f"{st[:10]:>12}" for st in STATEORD))
for i,s in enumerate(SUBS):
    cells=""
    for j,st in enumerate(STATEORD):
        if np.isnan(L[i,j]): cells+=f"{'-':>12}"; continue
        pj=padj.get((s,st),1) if len(R) else 1; cells+=f"{L[i,j]:+.2f}{'*' if pj<0.05 else ' '}".rjust(12)
    print(f"{s:14}(n={Ncnt[i]:<4})"+cells)
# ================= FIGURE: heatmap =================
fig,ax=plt.subplots(figsize=(9,6))
M=np.ma.masked_invalid(L); vmax=np.nanmax(np.abs(L)) if np.isfinite(L).any() else 1
im=ax.imshow(M,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
ax.set_xticks(range(len(STATEORD))); ax.set_xticklabels(STATEORD,rotation=30,ha="right")
ax.set_yticks(range(len(SUBS))); ax.set_yticklabels([f"{s} (n={Ncnt[i]})" for i,s in enumerate(SUBS)])
for i in range(len(SUBS)):
    for j in range(len(STATEORD)):
        if np.isnan(L[i,j]): ax.text(j,i,"n/a",ha="center",va="center",fontsize=7,color="#999"); continue
        pj=padj.get((SUBS[i],STATEORD[j]),1) if len(R) else 1
        ax.text(j,i,f"{L[i,j]:+.2f}"+("*" if pj<0.05 else ""),ha="center",va="center",fontsize=8,color="white" if abs(L[i,j])>vmax*0.6 else "#222")
ax.set_title("Clean microglia state enrichment near each T-cell subset\nlog2(near ≤30um / baseline); * padj<0.05; n/a if <10 microglia near",fontsize=10,fontweight="bold")
fig.colorbar(im,ax=ax,shrink=0.7,label="log2 enrichment")
plt.tight_layout(); fig.savefig(NEW/"states_around_Tsubsets.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: states_around_Tsubsets.png + states_around_Tsubsets.csv")
