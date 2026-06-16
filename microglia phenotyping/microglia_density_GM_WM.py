"""
280_microglia_density_GM_WM.py
Microglia count / mm2 in gray vs white matter.
No GM/WM annotation -> derive spatially: grid each section into 100um bins, classify each
occupied bin as GM (neuron-rich) or WM (oligo-rich) by oligolineage fraction among
(neurons + oligolineage). Microglia/mm2 = microglia in class bins / (n bins * bin area).
Per donor + pooled. Output: microglia_density_GM_WM.png + .csv
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":10})
CMAP=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap"); SAVE=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
H5=CMAP/"merged_qc_brain_remapped.h5ad"; QC=Path(r"D:\Caterina\MERSCOPE\QC data")
BIN=100.0  # um  -> bin area = 0.01 mm2
BINAREA=(BIN/1000.0)**2
WM_THR=0.60; GM_THR=0.40  # oligolineage fraction thresholds (between = transition, excluded)
MINCELL=5  # min (neuron+oligo) anchor cells per bin to classify

with h5py.File(H5,"r") as f:
    og=f["obs"]; idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in og[og.attrs.get("_index","_index")][:]])
    def cat(n):
        node=og[n]; c=[s.decode() if isinstance(s,bytes) else s for s in node["categories"][:]]; return np.array([c[i] for i in node["codes"][:]])
    v2=cat("cell_type_v2"); donor=cat("donor")
is_mic=(v2=="Mic"); is_neuron=np.isin(v2,["Exc","Inh"]); is_oligo=np.isin(v2,["Oli","OPC"])

ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    df=pd.read_csv(csv,usecols=["EntityID","center_x","center_y"])
    for eid,x,y in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"]): ent[(d.name,eid)]=(x,y)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan); run=np.array(["?"]*len(idx),dtype=object)
for i,cid in enumerate(idx):
    if "_" not in cid: continue
    pre,eid=cid.rsplit("_",1); run[i]=pre
    if (pre,eid) in ent: mx[i],my[i]=ent[(pre,eid)]
hasxy=np.isfinite(mx)

rows=[]; perdonor={}
for r in np.unique(run):
    sel=np.where((run==r)&hasxy)[0]
    if len(sel)<500: continue
    bx=np.floor(mx[sel]/BIN).astype(int); by=np.floor(my[sel]/BIN).astype(int)
    key=bx.astype(np.int64)*100000+by
    df=pd.DataFrame({"key":key,"mic":is_mic[sel],"neu":is_neuron[sel],"oli":is_oligo[sel],"all":1})
    gb=df.groupby("key").sum()
    anchor=gb["neu"]+gb["oli"]
    ofrac=gb["oli"]/anchor.replace(0,np.nan)
    gb["class"]=np.where((anchor>=MINCELL)&(ofrac>=WM_THR),"WM",np.where((anchor>=MINCELL)&(ofrac<GM_THR),"GM","other"))
    don=donor[sel][0]
    for cls in ["GM","WM"]:
        sub=gb[gb["class"]==cls]
        nbin=len(sub); area=nbin*BINAREA; mic=int(sub["mic"].sum()); tot=int(sub["all"].sum())
        if nbin<5: continue
        rows.append(dict(run=r,donor=don,compartment=cls,n_bins=nbin,area_mm2=round(area,3),
                         microglia=mic,total_cells=tot,mic_per_mm2=round(mic/area,1),
                         allcell_per_mm2=round(tot/area,1),mic_pct=round(100*mic/tot,1)))
R=pd.DataFrame(rows); R.to_csv(SAVE/"microglia_density_GM_WM.csv",index=False)
print("Per-section microglia density (GM vs WM):")
print(R.to_string(index=False))

# pooled per donor (area-weighted)
print("\nPer-donor pooled (area-weighted) microglia/mm2:")
summ=[]
for don in sorted(R.donor.unique()):
    for cls in ["GM","WM"]:
        s=R[(R.donor==don)&(R.compartment==cls)]
        if len(s)==0: continue
        mic=s.microglia.sum(); area=s.area_mm2.sum(); tot=s.total_cells.sum()
        summ.append(dict(donor=don,compartment=cls,microglia=int(mic),area_mm2=round(area,2),mic_per_mm2=round(mic/area,1),mic_pct=round(100*mic/tot,1)))
SD=pd.DataFrame(summ); print(SD.to_string(index=False))
print("\nWM:GM microglia density ratio per donor:")
for don in sorted(SD.donor.unique()):
    g=SD[(SD.donor==don)&(SD.compartment=="GM")].mic_per_mm2.values
    w=SD[(SD.donor==don)&(SD.compartment=="WM")].mic_per_mm2.values
    if len(g) and len(w): print(f"  {don}: GM={g[0]:.0f}/mm2  WM={w[0]:.0f}/mm2  WM/GM={w[0]/g[0]:.2f}")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
dons=sorted(SD.donor.unique()); xx=np.arange(len(dons)); w=0.36
for k,(cls,cc) in enumerate([("GM","#27ae60"),("WM","#8e44ad")]):
    vals=[SD[(SD.donor==d)&(SD.compartment==cls)].mic_per_mm2.values for d in dons]; vals=[v[0] if len(v) else 0 for v in vals]
    ax1.bar(xx+(k-0.5)*w,vals,w,color=cc,edgecolor="#333",lw=0.4,label=cls)
    for xi,v in zip(xx+(k-0.5)*w,vals): ax1.text(xi,v+5,f"{v:.0f}",ha="center",fontsize=9)
ax1.set_xticks(xx); ax1.set_xticklabels(dons); ax1.set_ylabel("microglia / mm²"); ax1.set_title("Microglia density: GM vs WM (per donor)",fontsize=11,fontweight="bold"); ax1.legend()
for sp in ("top","right"): ax1.spines[sp].set_visible(False)
# per-section distribution
import matplotlib.patches as mp
for k,(cls,cc) in enumerate([("GM","#27ae60"),("WM","#8e44ad")]):
    d=R[R.compartment==cls]; ax2.scatter(np.full(len(d),k)+np.random.uniform(-0.12,0.12,len(d)),d.mic_per_mm2,c=cc,s=40,edgecolors="#333",lw=0.3,alpha=0.85)
    ax2.hlines(d.mic_per_mm2.median(),k-0.25,k+0.25,color="#222",lw=2)
ax2.set_xticks([0,1]); ax2.set_xticklabels(["GM","WM"]); ax2.set_ylabel("microglia / mm² (per section)"); ax2.set_title("Per-section microglia density",fontsize=11,fontweight="bold")
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
fig.suptitle(f"Microglia density in gray vs white matter (spatial 100µm-bin GM/WM classification; oligolineage frac ≥{WM_THR}=WM, <{GM_THR}=GM)",fontsize=10.5,fontweight="bold",y=1.02)
plt.tight_layout(); fig.savefig(SAVE/"microglia_density_GM_WM.png",dpi=120,bbox_inches="tight"); plt.close()
print("\nSaved: microglia_density_GM_WM.png + microglia_density_GM_WM.csv")
