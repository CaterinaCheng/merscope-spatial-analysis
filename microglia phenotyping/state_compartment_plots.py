"""
360_state_compartment_plots.py
(left)  T-subset proximity heatmap with the untestable CD4 rows (n<10) removed.
(right) compartment RATIO within EACH microglia state (% of each state's cells that are
        perivascular / vessel-adjacent / parenchymal).
Uses the strict CLEAN labels (canonical; retains MHC-II/APC).
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
STATEORD=["Homeostatic","MHC-II/APC","DAM","Phagocytic","Inflammatory/IEG"]
SCOL={"Homeostatic":"#3498DB","MHC-II/APC":"#9B59B6","DAM":"#E74C3C","Phagocytic":"#16A085","Inflammatory/IEG":"#F1C40F"}
CCOL={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
comps=["perivascular","vessel-adjacent","parenchymal"]
# ---- T-subset proximity (strict 357), keep only testable subsets (already n>=10 in csv) ----
TS=pd.read_csv(NEW/"states_around_Tsubsets.csv")
subs=[s for s in ["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","NK"] if s in TS.subset.unique()]
nmap={s:int(TS[TS.subset==s].n.iloc[0]) for s in subs}
L=np.full((len(subs),len(STATEORD)),np.nan); pad=np.ones_like(L)
for i,s in enumerate(subs):
    for j,st in enumerate(STATEORD):
        r=TS[(TS.subset==s)&(TS.state==st)]
        if len(r): L[i,j]=r.log2.iloc[0]; pad[i,j]=r.padj.iloc[0] if "padj" in r else 1
Ldisp=np.clip(L,-2,2)  # cap APC-zero artifact for color
# ---- compartment ratio per state (strict clean spatial) ----
sp=pd.read_csv(NEW/"clean_microglia_spatial.csv",index_col=0); sp=sp[sp.comp!="n/a"]
RAT=pd.DataFrame({st:[100*((sp.state==st)&(sp.comp==c)).sum()/max((sp.state==st).sum(),1) for c in comps] for st in STATEORD},index=comps).T
RAT.to_csv(NEW/"state_compartment_ratios.csv")
print("compartment ratio per state (%):"); print(RAT.round(1).to_string())
# ================= FIGURE =================
fig,(axH,axB)=plt.subplots(1,2,figsize=(15,5.4),gridspec_kw={"width_ratios":[1,1.05]})
vmax=2
im=axH.imshow(Ldisp,cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
axH.set_xticks(range(len(STATEORD))); axH.set_xticklabels(STATEORD,rotation=25,ha="right")
axH.set_yticks(range(len(subs))); axH.set_yticklabels([f"{s} (n={nmap[s]})" for s in subs])
for i in range(len(subs)):
    for j in range(len(STATEORD)):
        if np.isnan(L[i,j]): continue
        txt="~0" if L[i,j]<-5 else f"{L[i,j]:+.2f}"+("*" if pad[i,j]<0.05 else "")
        axH.text(j,i,txt,ha="center",va="center",fontsize=8.5,color="white" if abs(Ldisp[i,j])>1.2 else "#222")
axH.set_title("Microglia state near each T subset\nlog2(near ≤30um / baseline); * padj<0.05  (CD4 subsets n<10 omitted)",fontsize=10,fontweight="bold")
fig.colorbar(im,ax=axH,shrink=0.7,label="log2 (capped ±2)")
# compartment ratio per state stacked
x=np.arange(len(STATEORD)); bottom=np.zeros(len(STATEORD))
for c in comps:
    vals=RAT[c].values; axB.bar(x,vals,bottom=bottom,color=CCOL[c],label=c)
    for xi,(v,b) in enumerate(zip(vals,bottom)):
        if v>=4: axB.text(xi,b+v/2,f"{v:.0f}%",ha="center",va="center",fontsize=8,color="white",fontweight="bold")
    bottom+=vals
axB.set_xticks(x); axB.set_xticklabels([f"{s}\n(n={int((sp.state==s).sum())})" for s in STATEORD],fontsize=8)
axB.set_ylabel("% of state's microglia"); axB.set_ylim(0,100)
axB.set_title("Vascular compartment ratio within each microglia state",fontsize=10,fontweight="bold")
axB.legend(fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=3)
for sp_ in axB.spines.values(): sp_.set_visible(False)
plt.tight_layout(); fig.savefig(NEW/"state_compartment_ratios.png",dpi=200,bbox_inches="tight"); plt.close()
print("\nSaved: state_compartment_ratios.png + state_compartment_ratios.csv")
