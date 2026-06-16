"""
355_clean_spatial.py
Carry the CLEAN microglia 5-state labels (353/354, flagged clusters excluded) into the
spatial analyses:
 (1) vascular compartment (peri<=30 / adj 30-100 / paren>=100 um to nearest End/Per/SMC):
     state composition per compartment + per-state peri-vs-paren enrichment (Fisher, BH).
 (2) microglia within 30um of a CD8 TRM 2 vs baseline (no T/NK within 30um):
     state composition + per-state enrichment.
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
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new"); QC=Path(r"<MERSCOPE_ROOT>\QC data")
DEC=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_decontaminated.h5ad")
VESSEL=["End","Per","SMC"]; STATEORD=["Homeostatic","Phagocytic","DAM","Inflammatory/IEG","MHC-II/APC"]  # largest ratio first -> bottom of stack
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
co=pd.read_csv(NEW/"microglia_final_coords.csv",index_col=0)
co=co[~co.cluster_flag]   # validated microglia only
mstate=co.state; print(f"validated microglia: {len(co)}")
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_ves=np.isin(v2,VESSEL); is_tnk=(v2=="T/NK")
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
labv=lab.reindex(idx).values
# coordinates for ALL cells
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
# microglia index positions
mpos={c:i for i,c in enumerate(idx)}; mi=np.array([mpos[c] for c in co.index])
dV=np.full(len(mi),np.inf); nearTRM2=np.zeros(len(mi),bool); anyT=np.zeros(len(mi),int)
TRM2=(labv=="CD8 TRM 2")
for r in np.unique(run[mi]):
    sel=np.where(run[mi]==r)[0]; mxy=np.column_stack([mx[mi[sel]],my[mi[sel]]]); ok=np.isfinite(mxy[:,0])
    if ok.sum()==0: continue
    vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(mxy[ok],k=1); tmp=np.full(len(sel),np.inf); tmp[ok]=dd; dV[sel]=tmp
    ts=np.where(TRM2&(run==r)&hasxy)[0]
    if len(ts): d2,_=cKDTree(np.column_stack([mx[ts],my[ts]])).query(mxy[ok],k=1); idx2=sel[ok]; nearTRM2[idx2]=d2<=30
    alls=np.where(is_tnk&(run==r)&hasxy)[0]
    if len(alls): d3,_=cKDTree(np.column_stack([mx[alls],my[alls]])).query(mxy[ok],k=1); idx3=sel[ok]; anyT[idx3]=(d3<=30).astype(int)
comp=np.where(dV<=30,"perivascular",np.where(dV<100,"vessel-adjacent",np.where(np.isfinite(dV),"parenchymal","n/a")))
co=co.assign(comp=comp,nearTRM2=nearTRM2,anyT=anyT)
val=co[co.comp!="n/a"]
def comptable(groupcol,g1,g2,sub=None):
    d=val if sub is None else val[sub]
    rows=[]
    for st in STATEORD:
        a1=int(((d[groupcol]==g1)&(d.state==st)).sum()); n1=int((d[groupcol]==g1).sum())
        a2=int(((d[groupcol]==g2)&(d.state==st)).sum()); n2=int((d[groupcol]==g2).sum())
        orr,p=fisher_exact([[a1,n1-a1],[a2,n2-a2]]); l2=np.log2(((a1/n1)+1e-6)/((a2/n2)+1e-6))
        rows.append(dict(state=st,f_g1=100*a1/n1,f_g2=100*a2/n2,log2=l2,p=p))
    R=pd.DataFrame(rows); o=np.argsort(R.p.values); rk=np.empty(len(R),int); rk[o]=np.arange(1,len(R)+1)
    R["padj"]=np.minimum(R.p*len(R)/rk,1); return R
COMP=comptable("comp","perivascular","parenchymal")
print("\n=== compartment: state composition ===")
for c in ["perivascular","vessel-adjacent","parenchymal"]:
    sub=val[val.comp==c]; print(f"  {c:15} (n={len(sub)}):",{s:f"{100*(sub.state==s).mean():.0f}%" for s in STATEORD})
print("\n=== peri vs paren enrichment (Fisher, *padj<0.05) ===")
for _,r in COMP.iterrows(): print(f"  {r.state:18}: peri {r.f_g1:4.1f}% vs paren {r.f_g2:4.1f}%  log2={r.log2:+.2f} padj={r.padj:.3g}{' *' if r.padj<0.05 else ''}")
# near CD8 TRM2 vs baseline (no T within 30)
co["grp"]=np.where(co.nearTRM2,"nearTRM2",np.where(co.anyT==0,"baseline","other"))
val2=co[co.grp.isin(["nearTRM2","baseline"])]
TR=comptable.__wrapped__ if False else None
rows=[]
for st in STATEORD:
    a1=int(((val2.grp=="nearTRM2")&(val2.state==st)).sum()); n1=int((val2.grp=="nearTRM2").sum())
    a2=int(((val2.grp=="baseline")&(val2.state==st)).sum()); n2=int((val2.grp=="baseline").sum())
    orr,p=fisher_exact([[a1,n1-a1],[a2,n2-a2]]); l2=np.log2(((a1/n1)+1e-6)/((a2/n2)+1e-6))
    rows.append(dict(state=st,f_near=100*a1/n1,f_base=100*a2/n2,log2=l2,p=p,n_near=n1,n_base=n2))
TRM=pd.DataFrame(rows); o=np.argsort(TRM.p.values); rk=np.empty(len(TRM),int); rk[o]=np.arange(1,len(TRM)+1); TRM["padj"]=np.minimum(TRM.p*len(TRM)/rk,1)
print(f"\n=== microglia near CD8 TRM2 (n={int((val2.grp=='nearTRM2').sum())}) vs baseline (n={int((val2.grp=='baseline').sum())}) ===")
for _,r in TRM.iterrows(): print(f"  {r.state:18}: near {r.f_near:4.1f}% vs base {r.f_base:4.1f}%  log2={r.log2:+.2f} padj={r.padj:.3g}{' *' if r.padj<0.05 else ''}")
COMP.to_csv(NEW/"clean_compartment_enrichment.csv",index=False); TRM.to_csv(NEW/"clean_nearTRM2_enrichment.csv",index=False)
co[["state","comp","nearTRM2","anyT","grp"]].to_csv(NEW/"clean_microglia_spatial.csv")
# ================= FIGURE =================
fig,axes=plt.subplots(1,3,figsize=(16,5.2),gridspec_kw={"width_ratios":[1,1,1]})
# A compartment composition stacked
ax=axes[0]; comps=["perivascular","vessel-adjacent","parenchymal"]; bottom=np.zeros(len(comps)); THR=4.0; small={i:[] for i in range(len(comps))}
for st in STATEORD:
    vals=np.array([100*(val[val.comp==c].state==st).mean() for c in comps]); ax.bar(range(len(comps)),vals,bottom=bottom,color=SCOL[st],label=st)
    for i,(v,b) in enumerate(zip(vals,bottom)):
        m=b+v/2
        if v>=THR: ax.text(i,m,f"{v:.0f}%",ha="center",va="center",fontsize=7.5,color="white",fontweight="bold")
        elif v>0: small[i].append((st,v,m))
    bottom+=vals
for i,items in small.items():
    for k,(st,v,m) in enumerate(sorted(items,key=lambda t:t[2])):
        ax.annotate(f"{st}: {v:.0f}%",xy=(i,m),xytext=(i,104+k*5),fontsize=6.5,color=SCOL[st],fontweight="bold",va="bottom",ha="center",
                    arrowprops=dict(arrowstyle="-",color=SCOL[st],lw=0.7,shrinkA=0,shrinkB=1))
ax.set_xticks(range(len(comps))); ax.set_xticklabels([f"{c}\n(n={int((val.comp==c).sum())})" for c in comps],fontsize=8)
ax.set_ylim(0,114); ax.set_ylabel("% of microglia"); ax.set_title("A. state composition by vascular compartment",fontsize=10,fontweight="bold"); ax.legend(fontsize=7.5,loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=3)
# B peri vs paren enrichment
ax=axes[1]; y=np.arange(len(STATEORD)); ax.barh(y,COMP.log2,color=[SCOL[s] for s in COMP.state]); ax.axvline(0,color="k",lw=0.8)
for i,r in COMP.iterrows():
    ax.text(r.log2+(0.05 if r.log2>=0 else -0.05),i,("*" if r.padj<0.05 else "ns"),va="center",ha="left" if r.log2>=0 else "right",fontsize=10,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(COMP.state); ax.invert_yaxis(); ax.set_xlabel("log2(peri / paren)"); ax.set_title("B. perivascular vs parenchymal enrichment",fontsize=10,fontweight="bold")
for sp in ax.spines.values(): sp.set_visible(False)
# C near TRM2 vs baseline
ax=axes[2]; y=np.arange(len(STATEORD)); ax.barh(y,TRM.log2,color=[SCOL[s] for s in TRM.state]); ax.axvline(0,color="k",lw=0.8)
for i,r in TRM.iterrows():
    ax.text(r.log2+(0.03 if r.log2>=0 else -0.03),i,("*" if r.padj<0.05 else "ns"),va="center",ha="left" if r.log2>=0 else "right",fontsize=10,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(TRM.state); ax.invert_yaxis(); ax.set_xlabel("log2(near CD8 TRM2 / baseline)"); ax.set_title(f"C. near CD8 TRM2 (n={TRM.n_near.iloc[0]}) vs baseline",fontsize=10,fontweight="bold")
for sp in ax.spines.values(): sp.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"clean_microglia_spatial.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: clean_microglia_spatial.png + clean_{compartment,nearTRM2}_enrichment.csv + clean_microglia_spatial.csv")
