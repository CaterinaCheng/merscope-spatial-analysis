"""
320_distance_QC_and_niche.py
PART A: QC the perivascular threshold. Distance-to-nearest-vessel distribution per cell type;
        compare immune (T/Mic/MonoMac) vs PARENCHYMAL baseline (Exc/Inh neurons). The distance
        where immune are ENRICHED over the neuron baseline defines true perivascular space.
PART B: per T subset, distance to nearest MICROGLION and nearest MACROPHAGE; % within 30um of
        each; TRM1 vs TRM2 test (do CD103+ memory TRM sit closer to microglia / farther from Mono-Mac?).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats
plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
H5=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_brain_remapped.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
VESSEL=["End","Per","SMC"]
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(H5,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    nn=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in nn["categories"][:]]; v2=np.array([cats[c] for c in nn["codes"][:]])
clabel=np.array(v2,dtype=object); ls=lab.reindex(idx)
for i in range(len(idx)):
    if isinstance(ls.iloc[i],str): clabel[i]=ls.iloc[i]
is_ves=np.isin(v2,VESSEL); is_mic=(v2=="Mic"); is_mac=(v2=="Mono/Mac")
run=np.array([c.rsplit("_",1)[0] if "_" in c else "?" for c in idx],dtype=object)
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
dV=np.full(len(idx),np.inf); dMic=np.full(len(idx),np.inf); dMac=np.full(len(idx),np.inf)
for r in np.unique(run):
    cs=np.where((run==r)&hasxy)[0]; cxy=np.column_stack([mx[cs],my[cs]])
    for sel,arr in [(is_ves,dV),(is_mic,dMic),(is_mac,dMac)]:
        ss=np.where(sel&(run==r)&hasxy)[0]
        if len(ss): dd,_=cKDTree(np.column_stack([mx[ss],my[ss]])).query(cxy,k=1); arr[cs]=dd

# ===== PART A: perivascular threshold QC =====
print("=== PART A: distance-to-nearest-vessel (um) per cell type ===")
ct_groups={"Exc (parenchyma)":(v2=="Exc"),"Inh (parenchyma)":(v2=="Inh"),"Oli (parenchyma)":(v2=="Oli"),
           "Mic":is_mic,"Mono/Mac":is_mac,"all T":lab.reindex(idx).isin(["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]).values,
           "CD8 TRM 1":(clabel=="CD8 TRM 1"),"CD8 TRM 2":(clabel=="CD8 TRM 2")}
for nm,m in ct_groups.items():
    dv=dV[m&hasxy&np.isfinite(dV)]; print(f"  {nm:18}: median={np.median(dv):5.0f}  25th={np.percentile(dv,25):4.0f}  %<=10um={100*(dv<=10).mean():3.0f}  %<=20um={100*(dv<=20).mean():3.0f}  %<=50um={100*(dv<=50).mean():3.0f}")
# enrichment of T over neuron baseline by distance bin
bins=[0,10,20,30,50,100,1e9]; blab=["0-10","10-20","20-30","30-50","50-100",">100"]
exc_dv=dV[(v2=="Exc")&hasxy&np.isfinite(dV)]; t_dv=dV[ct_groups["all T"]&hasxy&np.isfinite(dV)]
exc_h=np.histogram(exc_dv,bins=bins)[0]/len(exc_dv); t_h=np.histogram(t_dv,bins=bins)[0]/len(t_dv)
print("\n  enrichment of T-cells over Exc-neuron baseline by vessel-distance bin (>1 = perivascular-enriched):")
for b,e,t in zip(blab,exc_h,t_h): print(f"    {b:7}um: Exc={100*e:4.1f}%  T={100*t:4.1f}%  enrichment={t/e if e>0 else np.nan:.2f}")

# ===== PART B: nearest microglia vs macrophage per T subset =====
print("\n=== PART B: T subset distance to nearest MICROGLION vs MACROPHAGE ===")
SUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg","NK"]
rows=[]
for s in SUBS:
    m=(clabel==s)&hasxy&np.isfinite(dMic)&np.isfinite(dMac)
    rows.append(dict(subset=s,n=int(m.sum()),med_dMic=round(np.median(dMic[m]),1),med_dMac=round(np.median(dMac[m]),1),
                     pct_Mic_30=round(100*(dMic[m]<=30).mean(),0),pct_Mac_30=round(100*(dMac[m]<=30).mean(),0)))
RB=pd.DataFrame(rows); print(RB.to_string(index=False)); RB.to_csv(NEW/"Tsubset_nearest_myeloid.csv",index=False)
# TRM1 vs TRM2 tests
m1=(clabel=="CD8 TRM 1")&hasxy&np.isfinite(dMic); m2=(clabel=="CD8 TRM 2")&hasxy&np.isfinite(dMic)
for nm,arr in [("dist to nearest MICROGLION",dMic),("dist to nearest MACROPHAGE",dMac)]:
    u,p=stats.mannwhitneyu(arr[m1],arr[m2]); print(f"  TRM1 vs TRM2 {nm}: median TRM1={np.median(arr[m1]):.0f} TRM2={np.median(arr[m2]):.0f}  Mann-Whitney p={p:.3f}")

# figure
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
# A: CDF of vessel distance
for nm,m,c in [("Exc neuron (parenchyma)",(v2=="Exc"),"#2471A3"),("all T",ct_groups["all T"],"#C0392B"),("Mono/Mac",is_mac,"#E67E22"),("Mic",is_mic,"#27AE60")]:
    dv=np.sort(dV[m&hasxy&np.isfinite(dV)]); ax1.plot(dv,np.linspace(0,1,len(dv)),color=c,label=nm,lw=1.8)
ax1.axvline(50,color="#888",ls="--",lw=1); ax1.axvline(20,color="#555",ls=":",lw=1)
ax1.set_xlim(0,120); ax1.set_xlabel("distance to nearest vessel (µm)"); ax1.set_ylabel("cumulative fraction")
ax1.set_title("A. Vessel-distance CDF — calibrate perivascular cutoff\n(T shifted left of neurons = perivascular-enriched)",fontsize=10,fontweight="bold"); ax1.legend(fontsize=8)
for sp in ("top","right"): ax1.spines[sp].set_visible(False)
# B: % within 30um of Mic vs Mac per subset
y=np.arange(len(SUBS)); w=0.38
ax2.barh(y+w/2,RB.pct_Mic_30,w,color="#27ae60",edgecolor="#333",lw=0.3,label="≤30µm of microglia")
ax2.barh(y-w/2,RB.pct_Mac_30,w,color="#e67e22",edgecolor="#333",lw=0.3,label="≤30µm of macrophage")
ax2.set_yticks(y); ax2.set_yticklabels([f"{s} (n={int(RB.iloc[i].n)})" for i,s in enumerate(SUBS)],fontsize=8); ax2.invert_yaxis()
ax2.set_xlabel("% of subset within 30µm"); ax2.set_title("B. Proximity to nearest microglion vs macrophage",fontsize=10,fontweight="bold"); ax2.legend(fontsize=8)
for sp in ("top","right"): ax2.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"distance_QC_and_niche.png",dpi=140,bbox_inches="tight"); plt.close()
print("\nSaved: distance_QC_and_niche.png + Tsubset_nearest_myeloid.csv")
