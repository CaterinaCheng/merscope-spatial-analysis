"""
Tcell_compartments.py
Assign each final-annotated T/NK cell a vascular compartment by distance to nearest vessel
(End/Per/SMC): perivascular <=30um, vessel-adjacent 30-100um, parenchymal >100um.
Plots: (A) overall compartment composition, (B) compartment composition per subset.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
plt.rcParams.update({"font.size":9})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
H5=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"<MERSCOPE_ROOT>\QC data")
VESSEL=["End","Per","SMC"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
lab=lab[lab!="NK"]  # T cells only, exclude NK

with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
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
hasxy=np.isfinite(mx); pos={c:i for i,c in enumerate(idx)}
tcells=[c for c in lab.index if c in pos and hasxy[pos[c]]]; ti=np.array([pos[c] for c in tcells])
dV=np.full(len(ti),np.inf)
for r in np.unique(run[ti]):
    sel=np.where(run[ti]==r)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs):
        dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[ti[sel]],my[ti[sel]]]),k=1)
        dV[sel]=dd
comp=np.where(dV<=30,"perivascular",np.where(dV<=100,"vessel-adjacent","parenchymal"))  # calibrated cutoffs
D=pd.DataFrame({"cell_id":tcells,"subset":lab.loc[tcells].values,"dist_vessel":np.minimum(dV,300),"compartment":comp})
D.to_csv(NEW/"Tcell_compartment_assignment.csv",index=False)
print("overall compartment counts:",D.compartment.value_counts().to_dict())
print("overall %:",(100*D.compartment.value_counts(normalize=True)).round(1).to_dict())

ccol={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
corder=["perivascular","vessel-adjacent","parenchymal"]
sorder=[s for s in ["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Tcm/mem","CD4 CTL","CD4 Th","CD4 Treg","NK"] if s in set(D.subset)]
SUBCOL={"CD8 TRM 1":"#1f77b4","CD8 TRM 2":"#ff7f0e","CD8 TEMRA":"#2ca02c","CD4 Tcm/mem":"#d62728","CD4 CTL":"#9467bd","CD4 Th":"#8c564b","CD4 Treg":"#e377c2","NK":"#7f7f7f"}
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.2),gridspec_kw={"width_ratios":[1,1.5]})
# A overall: stacked subset counts per compartment
ctc=pd.crosstab(D.subset,D.compartment)[corder].reindex(sorder)   # counts subset x compartment
comptot={c:int(ctc[c].sum()) for c in corder}
bottom=np.zeros(len(corder))
for s in sorder:
    vals=ctc.loc[s].values.astype(float); ax1.bar(range(len(corder)),vals,bottom=bottom,color=SUBCOL[s],edgecolor="white",lw=0.4,label=s)
    for i,(v,b) in enumerate(zip(vals,bottom)):
        if v>=22: ax1.text(i,b+v/2,f"{int(v)}",ha="center",va="center",fontsize=7.5,color="white",fontweight="bold")
    bottom+=vals
for i,c in enumerate(corder):
    ax1.text(i,comptot[c]+8,f"{comptot[c]}",ha="center",fontsize=9,fontweight="bold")
# parenchymal segments are tiny -> leader-line callouts pointing outside the column
pidx=corder.index("parenchymal"); pt=comptot["parenchymal"]
acc=0.0; pmid={}
for s in sorder:
    v=ctc.loc[s,"parenchymal"]
    if v>0: pmid[s]=acc+v/2
    acc+=v
pseg=[s for s in sorder if ctc.loc[s,"parenchymal"]>0]; ny=len(pseg)
ys=np.linspace(pt+110,pt+110+(ny-1)*68,ny) if ny>1 else np.array([pt+150])
for k,s in enumerate(pseg):
    v=int(ctc.loc[s,"parenchymal"])
    ax1.annotate(f"{s}: {v}",xy=(pidx+0.18,pmid[s]),xytext=(pidx+0.45,ys[k]),
                 fontsize=7,va="center",ha="left",color=SUBCOL[s],fontweight="bold",
                 arrowprops=dict(arrowstyle="-",color=SUBCOL[s],lw=0.7,shrinkA=0,shrinkB=2))
ax1.set_xticks(range(len(corder))); ax1.set_xticklabels(corder,rotation=15); ax1.set_ylabel("T cell count")
ax1.set_title(f"All T cells (n={len(D)}) by compartment — counts per subset",fontsize=10,fontweight="bold")
ax1.set_ylim(0,bottom.max()*1.42); ax1.set_xlim(-0.6,pidx+1.5)
ax1.legend(fontsize=7,loc="upper left",frameon=False,ncol=2)
for sp in ("top","right"): ax1.spines[sp].set_visible(False)
# B per subset (stacked %)
ct=pd.crosstab(D.subset,D.compartment,normalize="index")[corder].reindex(sorder)*100
bottom=np.zeros(len(sorder))
for c in corder:
    ax2.barh(range(len(sorder)),ct[c].values,left=bottom,color=ccol[c],edgecolor="white",lw=0.5,label=c)
    for yi,(v,l) in enumerate(zip(ct[c].values,bottom)):
        if v>=5:  # fits inside the segment
            ax2.text(l+v/2,yi,f"{v:.0f}%",ha="center",va="center",fontsize=8,color="white",fontweight="bold")
        elif c=="parenchymal":  # always label parenchymal, even when tiny: place just outside the bar
            ax2.text(101,yi,f"{v:.0f}%",ha="left",va="center",fontsize=8,color=ccol[c],fontweight="bold")
    bottom+=ct[c].values
ax2.set_yticks(range(len(sorder))); ax2.set_yticklabels([f"{s} (n={int((D.subset==s).sum())})" for s in sorder]); ax2.invert_yaxis()
ax2.set_xlabel("% of subset"); ax2.set_xlim(0,105); ax2.set_title("Compartment composition per subset (perivascular <=30um)",fontsize=10,fontweight="bold")
ax2.legend(fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.12),ncol=3,frameon=False)
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"Tcell_compartments.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nper-subset compartment % :"); print(ct.round(0).astype(int).to_string())

# ===== second plot: subset composition (ratios) within each compartment + all =====
ctp=(pd.crosstab(D.subset,D.compartment,normalize="columns")[corder].reindex(sorder)*100)  # subset x compartment, cols sum to 100
ctp["all"]=(D.subset.value_counts(normalize=True)*100).reindex(sorder)   # overall (all compartments)
cols=corder+["all"]; comptot["all"]=len(D)
ctp[cols].to_csv(NEW/"Tcell_compartment_subset_ratios.csv")
fig2,axc=plt.subplots(figsize=(9,5.9))
THR=2.8; bottom=np.zeros(len(cols)); small={i:[] for i in range(len(cols))}
for s in sorder:
    vals=ctp.loc[s,cols].values; axc.bar(range(len(cols)),vals,bottom=bottom,color=SUBCOL[s],edgecolor="white",lw=0.5,label=s)
    for i,(v,b) in enumerate(zip(vals,bottom)):
        m=b+v/2
        if v>=THR: axc.text(i,m,f"{v:.0f}%",ha="center",va="center",fontsize=8,color="white",fontweight="bold")
        elif v>0: small[i].append((s,v,m))
    bottom+=vals
# leader-line callouts (straight up) for segments too small to label inline
for i,items in small.items():
    for k,(s,v,m) in enumerate(sorted(items,key=lambda t:t[2])):
        axc.annotate(f"{s}: {v:.1f}%",xy=(i,m),xytext=(i,104+k*5),fontsize=6.8,color=SUBCOL[s],fontweight="bold",
                     va="bottom",ha="center",arrowprops=dict(arrowstyle="-",color=SUBCOL[s],lw=0.7,shrinkA=0,shrinkB=1))
axc.axvline(len(corder)-0.5,color="#999",ls="--",lw=1)   # divider before 'all'
axc.set_xticks(range(len(cols))); axc.set_xticklabels([f"{c}\n(n={comptot[c]})" for c in cols]); axc.set_ylabel("% of compartment")
axc.set_ylim(0,114); axc.set_title("T-cell subset composition within each compartment",fontsize=11,fontweight="bold")
axc.legend(fontsize=8,loc="center left",bbox_to_anchor=(1.02,0.5),frameon=False)
for sp in ("top","right"): axc.spines[sp].set_visible(False)
plt.tight_layout(); fig2.savefig(NEW/"Tcell_compartment_subset_ratios.png",dpi=150,bbox_inches="tight"); plt.close()
print("\nsubset composition per compartment (%):"); print(ctp.round(1).to_string())
print("\nSaved: Tcell_compartment_assignment.csv + Tcell_compartments.png + Tcell_compartment_subset_ratios.png")
