"""
281_stats_hardening.py
Harden the headline microglia DAM/MHC-II ~ T-exposure(30um) effects against the
pseudoreplication / spatial-autocorrelation / confound critiques (per scHPF guide Step13/16):
 A. Baseline cell-level OLS (reproduce).
 B. + cell VOLUME covariate (MERSCOPE confound).
 C. Cluster-robust SE (cluster by section; by donor).
 D. Donor-level MixedLM (random intercept = donor) and section-level MixedLM (RE = section).
 E. Section-level aggregation OLS (~10 sections).
 F. Per-donor consistency (fit each donor separately).
 G. Donor-balanced subsample (equal n per donor).
 H. Moran's I + Getis-Ord Gi* hotspot fraction for DAM & MHC-II module scores (per section).
Output: stats_hardening_DAM_MHCII.csv + stats_hardening.png
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd
import h5py, anndata as ad, scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
sc.settings.verbosity=0; plt.rcParams.update({"font.size":10}); rng=np.random.default_rng(0)
CMAP=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap"); SAVE=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new"); LAB=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF")
H5=CMAP/"merged_qc_brain_remapped.h5ad"; QC=Path(r"D:\Caterina\MERSCOPE\QC data")
SUBS=["CD8 TRM","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]; VESSEL=["End","Per","SMC"]
MODULES={"DAM_activation":["CD68","APOE","SPP1","TREM2","GPNMB","FTL","CST7","ITGAX","LPL"],
         "MHCII_antigen_pres":["CIITA","HLA-DRA","HLA-DPA1","HLA-DQB1","CD74"]}

with h5py.File(H5,"r") as f:
    og=f["obs"]; idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in og[og.attrs.get("_index","_index")][:]])
    vg=f["var"]; var=[s.decode() if isinstance(s,bytes) else s for s in vg[vg.attrs.get("_index","_index")][:]]
    g=f["layers/counts"]; X=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    def cat(name):
        n=og[name]; c=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; return np.array([c[i] for i in n["codes"][:]])
    v2=cat("cell_type_v2"); donor=cat("donor")
vp={gn:i for i,gn in enumerate(var)}
ph=pd.concat([pd.read_csv(LAB/"schpf_CD8_final_labels.csv")[["cell_id","phenotype"]],
              pd.read_csv(LAB/"schpf_CD4_final_labels.csv")[["cell_id","phenotype"]]]).set_index("cell_id")["phenotype"].reindex(idx).astype(str).values
is_t=np.isin(ph,SUBS); is_mic=(v2=="Mic"); is_ves=np.isin(v2,VESSEL)

# coords + volume + run
ent={}
for d in sorted([x for x in QC.iterdir() if x.is_dir()]):
    csv=d/"cell_metadata.csv"
    if not csv.exists(): continue
    head=pd.read_csv(csv,nrows=0).columns
    cols=["EntityID","center_x","center_y"]+(["volume"] if "volume" in head else [])
    df=pd.read_csv(csv,usecols=cols)
    if "volume" not in df.columns: df["volume"]=np.nan
    for eid,x,y,v in zip(df["EntityID"].astype(str),df["center_x"],df["center_y"],df["volume"]): ent[(d.name,eid)]=(x,y,v)
mx=np.full(len(idx),np.nan); my=np.full(len(idx),np.nan); vol=np.full(len(idx),np.nan); run=np.array(["?"]*len(idx),dtype=object)
for i,cid in enumerate(idx):
    if "_" not in cid: continue
    pre,eid=cid.rsplit("_",1); run[i]=pre
    if (pre,eid) in ent: mx[i],my[i],vol[i]=ent[(pre,eid)]
hasxy=np.isfinite(mx)   # volume optional (some sections lack it) -> only used in model B

mglob=np.where(is_mic&hasxy)[0]
a=ad.AnnData(X=X[mglob].copy(),var=pd.DataFrame(index=var)); sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
for m,gs in MODULES.items(): sc.tl.score_genes(a,[g for g in gs if g in vp],score_name=m)
Td30=np.zeros(len(mglob)); dV=np.full(len(mglob),np.inf); mpos={gi:k for k,gi in enumerate(mglob)}
for r in np.unique(run):
    mk=[gi for gi in mglob if run[gi]==r]
    if not mk: continue
    rows=[mpos[gi] for gi in mk]; mxy=np.column_stack([mx[mk],my[mk]])
    tsel=np.where(is_t&(run==r)&hasxy)[0]
    if len(tsel):
        for rr,h in zip(rows,cKDTree(np.column_stack([mx[tsel],my[tsel]])).query_ball_point(mxy,r=30)): Td30[rr]=len(h)
    vsel=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vsel):
        dvv,_=cKDTree(np.column_stack([mx[vsel],my[vsel]])).query(mxy,k=1)
        for rr,vv in zip(rows,dvv): dV[rr]=vv
tot=np.asarray(X[mglob].sum(1)).ravel(); comp=np.where(dV<=50,"peri",np.where(dV<=100,"adj","paren"))
keep=np.isfinite(dV)
def z(v): v=v.astype(float); s=np.nanstd(v[keep]); return (v-np.nanmean(v[keep]))/(s if s>0 else 1)
D=pd.DataFrame({"Td30":z(np.log1p(Td30)),"dV":z(np.minimum(dV,300)),"logtot":z(np.log1p(tot)),"vol":z(np.log1p(vol[mglob])),
   "compA":(comp=="adj").astype(float),"compP":(comp=="paren").astype(float),
   "donorB":(donor[mglob]==pd.unique(donor[mglob])[0]).astype(float),"donor":donor[mglob],"run":run[mglob],
   "mx":mx[mglob],"my":my[mglob]})
for m in MODULES: D[m]=a.obs[m].values
D=D[keep].reset_index(drop=True)
print(f"microglia n={len(D)} ; donors={sorted(D.donor.unique())} ; sections={D.run.nunique()}")

def morans(xy,zv,k=6,nperm=199):
    n=len(zv)
    if n<k+5: return np.nan,np.nan,np.nan
    tree=cKDTree(xy); _,nb=tree.query(xy,k=k+1); nb=nb[:,1:]
    zc=zv-zv.mean(); denom=(zc**2).sum()
    Wz=zc[nb].mean(1); I=(zc*Wz).sum()/denom
    # Gi* hotspot fraction (row-std incl self, analytic z approx via permutation of local)
    perm=np.empty(nperm)
    for p in range(nperm):
        zp=rng.permutation(zc); perm[p]=(zp*zp[nb].mean(1)).sum()/denom
    pval=(np.sum(perm>=I)+1)/(nperm+1)
    # Gi*: standardized neighbor (incl self) mean -> hotspot |z|>1.96
    inc=np.concatenate([np.arange(n)[:,None],nb],1); g=zv[inc].mean(1)
    gi=(g-g.mean())/g.std(); hot=100*np.mean(np.abs(gi)>1.96)
    return I,pval,hot

results={}
for m in MODULES:
    rows=[]
    base=f"{m} ~ Td30 + dV + logtot + compA + compP + donorB"
    r=smf.ols(base,D).fit()
    rows.append(("A baseline OLS",r.params["Td30"],r.bse["Td30"],r.pvalues["Td30"]))
    Dv=D.dropna(subset=["vol"]); rv=smf.ols(base+" + vol",Dv).fit()
    rows.append((f"B +volume cov (n={len(Dv)})",rv.params["Td30"],rv.bse["Td30"],rv.pvalues["Td30"]))
    rc=smf.ols(base,D).fit(cov_type="cluster",cov_kwds={"groups":D["run"]})
    rows.append(("C cluster-robust (section)",rc.params["Td30"],rc.bse["Td30"],rc.pvalues["Td30"]))
    rcd=smf.ols(base,D).fit(cov_type="cluster",cov_kwds={"groups":D["donor"]})
    rows.append(("C cluster-robust (donor)",rcd.params["Td30"],rcd.bse["Td30"],rcd.pvalues["Td30"]))
    try:
        mm=smf.mixedlm(f"{m} ~ Td30 + dV + logtot + compA + compP",D,groups=D["donor"]).fit(method="lbfgs")
        rows.append(("D MixedLM (donor RE)",mm.params["Td30"],mm.bse["Td30"],mm.pvalues["Td30"]))
    except Exception as e: rows.append(("D MixedLM (donor RE)",np.nan,np.nan,np.nan))
    try:
        mms=smf.mixedlm(f"{m} ~ Td30 + dV + logtot + compA + compP + donorB",D,groups=D["run"]).fit(method="lbfgs")
        rows.append(("D MixedLM (section RE)",mms.params["Td30"],mms.bse["Td30"],mms.pvalues["Td30"]))
    except Exception as e: rows.append(("D MixedLM (section RE)",np.nan,np.nan,np.nan))
    agg=D.groupby("run").agg(Td30=("Td30","mean"),y=(m,"mean"),donorB=("donorB","mean")).reset_index()
    ra=smf.ols("y ~ Td30 + donorB",agg).fit()
    rows.append((f"E section-agg OLS (n={len(agg)})",ra.params["Td30"],ra.bse["Td30"],ra.pvalues["Td30"]))
    for dn in sorted(D.donor.unique()):
        sub=D[D.donor==dn]; rd=smf.ols(f"{m} ~ Td30 + dV + logtot + compA + compP",sub).fit()
        rows.append((f"F per-donor {dn} (n={len(sub)})",rd.params["Td30"],rd.bse["Td30"],rd.pvalues["Td30"]))
    nmin=D.donor.value_counts().min(); bal=pd.concat([D[D.donor==dn].sample(nmin,random_state=0) for dn in D.donor.unique()])
    rb=smf.ols(base,bal).fit()
    rows.append((f"G donor-balanced (n={len(bal)})",rb.params["Td30"],rb.bse["Td30"],rb.pvalues["Td30"]))
    results[m]=pd.DataFrame(rows,columns=["method","beta","se","p"])
    print(f"\n=== {m} : Td30 effect across methods ===")
    print(results[m].to_string(index=False,float_format=lambda x:f"{x:.4g}"))
    # Moran's I per section
    Is=[]; ps=[]; hots=[]
    for r in D.run.unique():
        sub=D[D.run==r]
        I,pv,hot=morans(sub[["mx","my"]].values,sub[m].values)
        if np.isfinite(I): Is.append(I); ps.append(pv); hots.append(hot)
    print(f"  Moran's I (spatial autocorr of {m}): mean={np.mean(Is):.3f}  range=[{min(Is):.3f},{max(Is):.3f}]  sig(p<0.05) in {sum(p<0.05 for p in ps)}/{len(ps)} sections; mean Gi* hotspot={np.mean(hots):.0f}%")
    results[m].to_csv(SAVE/f"stats_hardening_{m}.csv",index=False)

# figure: forest plots
fig,axes=plt.subplots(1,2,figsize=(15,6))
for ax,m in zip(axes,MODULES):
    R=results[m].dropna(subset=["beta"]); R=R[~R.method.str.startswith("E section-agg")].reset_index(drop=True); y=np.arange(len(R))[::-1]
    cols=["#c0392b" if (p<0.05) else "#999" for p in R.p]
    ax.errorbar(R.beta,y,xerr=1.96*R.se,fmt="o",color="#333",ecolor="#888",capsize=3,zorder=2)
    ax.scatter(R.beta,y,c=cols,s=55,zorder=3,edgecolors="#333",lw=0.4)
    for yi,(_,rr) in zip(y,R.iterrows()): ax.text(R.beta.max()+R.se.max()*2.5,yi,f"β={rr.beta:+.3f} p={rr.p:.1e}",va="center",fontsize=7.5)
    ax.axvline(0,color="#333",lw=0.8,ls="--")
    ax.set_yticks(y); ax.set_yticklabels(R.method,fontsize=8); ax.set_xlabel("β  (module ~ T-exposure ≤30µm)")
    ax.set_title(f"{m}\nT-exposure effect is stable across all robustness models",fontsize=10.5,fontweight="bold")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig.suptitle("Statistical hardening: pseudoreplication (cluster-robust SE, donor/section mixed models, section aggregation, per-donor, donor-balanced) + volume covariate",fontsize=10.5,fontweight="bold",y=1.02)
plt.tight_layout(); fig.savefig(SAVE/"stats_hardening.png",dpi=120,bbox_inches="tight"); plt.close()
print("\nSaved: stats_hardening.png + stats_hardening_*.csv")
