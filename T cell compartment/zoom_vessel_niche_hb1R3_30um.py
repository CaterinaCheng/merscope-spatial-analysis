"""
332_zoom_vessel_niche_hb1R3_30um.py
Spatial vessel-niche figure for <SAMPLE> R3 (run <RUN_ID>) with the NEW compartment rule:
perivascular <=30um, vessel-adjacent 30-100um, parenchymal >=100um. New pipeline data
(decontam master cell types, final T subsets EXCLUDING NK). Full-ROI overview + zoom panel
with 30um (red) and 100um (dashed) shells around each vessel cell, T cells labeled by distance.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from scipy.spatial import cKDTree
plt.rcParams.update({"font.size":9})
NEW=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF\new")
H5=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad")
QC=Path(r"<MERSCOPE_ROOT>\QC data"); CMAP=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap")
RUN="<RUN_ID>"; ZOOM_HALF=350; VESSEL=["End","Per","SMC"]
TSUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]  # T cells only, no NK
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
# restrict to this run
inrun=np.array([c.startswith(RUN+"_") for c in idx])
ids=idx[inrun]; v2r=v2[inrun]
# coords
cm=pd.read_csv(QC/RUN/"cell_metadata.csv",usecols=["EntityID","center_x","center_y"]).set_index("EntityID")
eid=[c.split("_",1)[1] for c in ids]
try: eid_key=[int(e) for e in eid]; cm.index=cm.index.astype(int)
except Exception: eid_key=eid
xy=cm.reindex(eid_key)
X=xy["center_x"].values; Y=xy["center_y"].values
ok=np.isfinite(X)&np.isfinite(Y)
df=pd.DataFrame({"cid":ids,"ct":v2r,"x":X,"y":Y})[ok].reset_index(drop=True)
df["subset"]=lab.reindex(df["cid"]).values
print(f"{RUN} (<SAMPLE> R3): {len(df)} cells with coords")
is_ves=df["ct"].isin(VESSEL).values
isT=df["subset"].isin(TSUBS).values
print(f"  vessel cells={is_ves.sum()}  T cells (no NK)={isT.sum()}")
# distance of every T cell to nearest vessel
vxy=df.loc[is_ves,["x","y"]].values
tree=cKDTree(vxy)
dT,_=tree.query(df.loc[isT,["x","y"]].values,k=1)
Tdf=df[isT].copy(); Tdf["dist"]=dT
Tdf["comp"]=np.where(Tdf.dist<=30,"perivascular",np.where(Tdf.dist<100,"vessel_adjacent","parenchymal"))
print("  T compartment counts:",Tdf["comp"].value_counts().to_dict())
COMP_CLR={"perivascular":"#B11A1A","vessel_adjacent":"#888888","parenchymal":"#1F77B4"}

# pick zoom center: a parenchymal T cell with peri+adj T nearby
T_par=Tdf[Tdf.comp=="parenchymal"]; T_peri=Tdf[Tdf.comp=="perivascular"]; T_adj=Tdf[Tdf.comp=="vessel_adjacent"]
best=None; bs=-1
for _,p in T_par.iterrows():
    cx,cy=p.x,p.y
    box=lambda d:((d.x-cx).abs()<ZOOM_HALF)&((d.y-cy).abs()<ZOOM_HALF)
    npe,na,npa=int(box(T_peri).sum()),int(box(T_adj).sum()),int(box(T_par).sum())
    if npe==0 or na==0 or npa==0: continue
    sc=min(npe,na,npa)*10+(npe+na+npa)
    if sc>bs: bs=sc; best={"cx":cx,"cy":cy,"npe":npe,"na":na,"npa":npa}
if best is None:  # fallback: densest T region
    cx,cy=Tdf.x.median(),Tdf.y.median(); best={"cx":cx,"cy":cy}
cx,cy=best["cx"],best["cy"]; xmin,xmax,ymin,ymax=cx-ZOOM_HALF,cx+ZOOM_HALF,cy-ZOOM_HALF,cy+ZOOM_HALF
print(f"  zoom center=({cx:.0f},{cy:.0f}) box contents:",{k:best[k] for k in best if k.startswith('n')})

fig,axes=plt.subplots(1,2,figsize=(20,9),gridspec_kw={"width_ratios":[1,1.2]})
# ---- left: full ROI ----
ax=axes[0]
oli=df.ct.isin(["Oli","OPC"]).values; other=~oli&~is_ves
ax.scatter(df.x[other],df.y[other],s=0.4,c="#e8e8e8",alpha=0.35,linewidths=0,zorder=1)
ax.scatter(df.x[oli],df.y[oli],s=0.4,c="#8b6914",alpha=0.5,linewidths=0,zorder=2)
ax.scatter(df.x[is_ves],df.y[is_ves],s=1.0,c="#1f77b4",alpha=0.7,linewidths=0,zorder=3)
for comp,c in COMP_CLR.items():
    m=Tdf.comp==comp; s=60 if comp=="parenchymal" else 40
    ax.scatter(Tdf.x[m],Tdf.y[m],s=s,marker="o",c=c,edgecolors="black",linewidths=0.5,zorder=10)
ax.add_patch(Rectangle((xmin,ymin),2*ZOOM_HALF,2*ZOOM_HALF,facecolor="none",edgecolor="black",lw=2.5,ls="--",zorder=20))
ax.set_aspect("equal"); ax.set_xlabel("center_x (µm)"); ax.set_ylabel("center_y (µm)")
ax.set_title(f"<SAMPLE>  R3  — full ROI  (new rule: peri<=30µm, paren>=100µm)\nzoom box centered at ({cx:.0f}, {cy:.0f})",fontsize=12,fontweight="bold")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
# ---- right: zoom ----
ax=axes[1]
inbox=(df.x.between(xmin,xmax))&(df.y.between(ymin,ymax)); dz=df[inbox]
vz=dz.ct.isin(VESSEL).values; oz=~vz&~dz.ct.isin(["Oli","OPC"]).values
ax.scatter(dz.x[oz],dz.y[oz],s=14,c="#e8e8e8",alpha=0.5,edgecolors="grey",linewidths=0.2,zorder=1)
ax.scatter(dz.x[vz],dz.y[vz],s=85,c="#FFD700",alpha=0.95,edgecolors="black",linewidths=0.6,marker="o",zorder=3)
for vx,vy in dz.loc[vz,["x","y"]].values:
    ax.add_patch(Circle((vx,vy),30,fill=False,edgecolor="#B11A1A",lw=0.5,ls="-",alpha=0.35,zorder=4))
    ax.add_patch(Circle((vx,vy),100,fill=False,edgecolor="#444444",lw=0.5,ls="--",alpha=0.30,zorder=4))
Tz=Tdf[(Tdf.x.between(xmin,xmax))&(Tdf.y.between(ymin,ymax))]
for comp,c in COMP_CLR.items():
    m=Tz.comp==comp
    if m.sum()==0: continue
    ax.scatter(Tz.x[m],Tz.y[m],s=220,marker="o",c=c,edgecolors="black",linewidths=0.9,zorder=10)
    for _,r in Tz[m].iterrows():
        ax.text(r.x+15,r.y+15,f"{int(r.dist)}µm",fontsize=8,fontweight="bold",color=c,
                bbox=dict(boxstyle="round,pad=0.15",facecolor="white",edgecolor=c,lw=0.6,alpha=0.9),zorder=15)
ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect("equal")
ax.set_xlabel("center_x (µm)"); ax.set_ylabel("center_y (µm)")
np_,na_,npa_=(Tz.comp=="perivascular").sum(),(Tz.comp=="vessel_adjacent").sum(),(Tz.comp=="parenchymal").sum()
ax.set_title(f"Zoom — vessel niche ({2*ZOOM_HALF}×{2*ZOOM_HALF} µm)\nT cells: peri={np_}, adj={na_}, paren={npa_}",fontsize=12,fontweight="bold")
for sp in ("top","right"): ax.spines[sp].set_visible(False)
handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#e8e8e8",markeredgecolor="grey",markersize=8,label="other cells"),
    Line2D([0],[0],marker="o",color="w",markerfacecolor="#FFD700",markeredgecolor="black",markersize=11,label="End / Per / SMC (vessel)"),
    Line2D([0],[0],marker="o",color="w",markerfacecolor=COMP_CLR["perivascular"],markeredgecolor="black",markersize=12,label="T cell — perivascular ≤30µm"),
    Line2D([0],[0],marker="o",color="w",markerfacecolor=COMP_CLR["vessel_adjacent"],markeredgecolor="black",markersize=12,label="T cell — vessel-adj 30-100µm"),
    Line2D([0],[0],marker="o",color="w",markerfacecolor=COMP_CLR["parenchymal"],markeredgecolor="black",markersize=12,label="T cell — parenchymal >100µm"),
    Line2D([0],[0],color="#B11A1A",lw=1.2,ls="-",label="30 µm shell (perivascular boundary)"),
    Line2D([0],[0],color="#444444",lw=1.2,ls="--",label="100 µm shell (parenchymal boundary)")]
ax.legend(handles=handles,loc="center left",bbox_to_anchor=(1.02,0.5),frameon=False,fontsize=10)
plt.tight_layout(rect=[0,0,0.93,1.0])
out=NEW/"zoom_vessel_niche_<SAMPLE>_R3_30um.png"
plt.savefig(out,dpi=170,bbox_inches="tight"); plt.close()
print("Saved:",out)
