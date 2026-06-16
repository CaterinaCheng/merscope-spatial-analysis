"""
333_microglia_compartments_and_DEG.py
Microglia vascular compartment (perivascular <=30um / vessel-adjacent 30-100 / parenchymal >=100)
ratios, and DEGs between the three groups. Decontam counts; vascular-spillover genes flagged.
Panel A: compartment composition (microglia vs all-T vs Exc neuron reference).
Panel B/C: DEG perivascular-vs-parenchymal and vessel-adjacent-vs-parenchymal microglia.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, h5py, scanpy as sc, anndata as ad, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
sc.settings.verbosity=0; plt.rcParams.update({"font.size":9})
NEW=Path(r"D:\Caterina\MERSCOPE\merged_analysis\scHPF\new")
DEC=Path(r"D:\Caterina\MERSCOPE\merged_analysis\cellmap\merged_qc_decontaminated.h5ad"); QC=Path(r"D:\Caterina\MERSCOPE\QC data")
VESSEL=["End","Per","SMC"]
TSUBS=["CD8 TRM 1","CD8 TRM 2","CD8 TEMRA","CD4 Th","CD4 CTL","CD4 Tcm/mem","CD4 Treg"]
# cross-lineage spillover that contaminates perivascular microglia (vascular + astro + oligo + T/NK + neuron)
VASC=set(("PECAM1 CLDN5 VWF A2M FLT1 ACTA2 PDGFRB RGS5 NOTCH3 COL1A1 COL1A2 COL3A1 COL4A1 COL5A1 DCN BGN LUM AHNAK RNASE1 IFITM1 IFITM3 EPAS1 SLC2A1 ABCB1 GSN TIMP3 SPARC SPARCL1 IGFBP7 CAV1 CAV2 EMCN SOX17 ANXA2 MYL9 TAGLN MGP APOD SLC38A2 ATP1A2 PTGDS HSPG2 VIM FN1 ENG "
            "AQP4 SLC1A3 SLC1A2 GJA1 GLUL GFAP ALDH1L1 NDRG2 FGFR3 SLC4A4 "  # astrocyte
            "MOG MAL PLP1 MOBP CNP MBP CLDN11 ST18 "  # oligodendrocyte
            "CD3D CD3E CD3G CD2 CD8A CD8B IL7R CXCR6 CCL5 LIME1 FYB1 SKAP1 IL32 LCK THEMIS CD247 ITK GZMK NKG7 "  # T/NK
            "RBFOX3 SYT1 SNAP25 GAD1 SLC17A7 MEG3 NRGN").split())  # neuron
lab=pd.read_csv(NEW/"Tcell_subset_final_labels.csv",index_col=0).iloc[:,0]
with h5py.File(DEC,"r") as f:
    idx=pd.Index([s.decode() if isinstance(s,bytes) else s for s in f["obs"][f["obs"].attrs.get("_index","_index")][:]])
    var=[s.decode() if isinstance(s,bytes) else s for s in f["var"][f["var"].attrs.get("_index","_index")][:]]
    g=f["layers/counts_decontam"]; Xd=csr_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(int(s) for s in g.attrs["shape"])).astype(np.float32)
    n=f["obs"]["cell_type_v2"]; cats=[s.decode() if isinstance(s,bytes) else s for s in n["categories"][:]]; v2=np.array([cats[c] for c in n["codes"][:]])
is_mic=(v2=="Mic"); is_ves=np.isin(v2,VESSEL); labv=lab.reindex(idx).values; isT=np.isin(labv,TSUBS)
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
hasxy=np.isfinite(mx); dV=np.full(len(idx),np.inf)
for r in np.unique(run):
    cs=np.where((run==r)&hasxy)[0]; vs=np.where(is_ves&(run==r)&hasxy)[0]
    if len(vs): dd,_=cKDTree(np.column_stack([mx[vs],my[vs]])).query(np.column_stack([mx[cs],my[cs]]),k=1); dV[cs]=dd
comp=np.where(dV<=30,"perivascular",np.where(dV<100,"vessel-adjacent","parenchymal"))
micm=is_mic&hasxy&np.isfinite(dV)
mc=pd.Series(comp[micm]).value_counts()
print("Microglia compartment counts:",mc.to_dict())
print("Microglia compartment %:",(100*mc/mc.sum()).round(1).to_dict())
# composition comparison
corder=["perivascular","vessel-adjacent","parenchymal"]; ccol={"perivascular":"#C0392B","vessel-adjacent":"#E67E22","parenchymal":"#2471A3"}
groups={"Microglia":micm,"All T cells":isT&hasxy&np.isfinite(dV),"Exc neuron (ref)":(v2=="Exc")&hasxy&np.isfinite(dV)}
COMPP={}
for gn,m in groups.items():
    vc=pd.Series(comp[m]).value_counts(normalize=True).reindex(corder).fillna(0)*100; COMPP[gn]=vc
CP=pd.DataFrame(COMPP)

# DEG
def deg(maskA,maskB):
    sel=np.where(maskA|maskB)[0]; a=ad.AnnData(X=Xd[sel].copy(),var=pd.DataFrame(index=var)); a.obs["g"]=np.where(maskA[sel],"A","B")
    sc.pp.normalize_total(a,target_sum=None); sc.pp.log1p(a); sc.tl.rank_genes_groups(a,"g",groups=["A"],reference="B",method="wilcoxon")
    r=sc.get.rank_genes_groups_df(a,group="A").rename(columns={"names":"gene","logfoldchanges":"log2FC","pvals":"pval","pvals_adj":"padj"})
    r=r[~r.gene.str.startswith("Blank")].copy(); r["spillover"]=r.gene.isin(VASC); return r
peri=micm&(comp=="perivascular"); adj=micm&(comp=="vessel-adjacent"); paren=micm&(comp=="parenchymal")
print(f"\nDEG groups: peri={peri.sum()} adj={adj.sum()} paren={paren.sum()}")
dPP=deg(peri,paren); dPP.to_csv(NEW/"DEG_microglia_peri_vs_paren.csv",index=False)
dAP=deg(adj,paren); dAP.to_csv(NEW/"DEG_microglia_adj_vs_paren.csv",index=False)
def show(r,nm):
    intr=r[~r.spillover]
    up=intr[(intr.padj<0.05)&(intr.log2FC>0)].nsmallest(10,"pval"); dn=intr[(intr.padj<0.05)&(intr.log2FC<0)].nsmallest(10,"pval")
    print(f"\n{nm} (intrinsic sig={int((intr.padj<0.05).sum())}):")
    print("  UP in first group :",", ".join(f"{g}(+{fc:.2f})" for g,fc in zip(up.gene,up.log2FC)) or "none")
    print("  DOWN(=up in paren):",", ".join(f"{g}({fc:.2f})" for g,fc in zip(dn.gene,dn.log2FC)) or "none")
    print("  cross-lineage spillover up:",", ".join(r[r.spillover&(r.padj<0.05)&(r.log2FC>0)].nlargest(10,"log2FC").gene) or "none")
show(dPP,"Perivascular vs Parenchymal microglia"); show(dAP,"Vessel-adjacent vs Parenchymal microglia")

# ===== FIGURE =====
fig=plt.figure(figsize=(18,5.6))
axA=fig.add_subplot(1,3,1)
bottom=np.zeros(len(CP.columns))
for c in corder:
    axA.bar(range(len(CP.columns)),CP.loc[c].values,bottom=bottom,color=ccol[c],edgecolor="white",lw=0.6,label=c)
    for j,(v,b) in enumerate(zip(CP.loc[c].values,bottom)):
        if v>=4: axA.text(j,b+v/2,f"{v:.0f}%",ha="center",va="center",fontsize=8.5,color="white",fontweight="bold")
    bottom+=CP.loc[c].values
axA.set_xticks(range(len(CP.columns))); axA.set_xticklabels([f"{k}\n(n={int(groups[k].sum())})" for k in CP.columns],fontsize=8.5)
axA.set_ylabel("% of cells"); axA.set_ylim(0,100); axA.set_title("Vascular compartment composition\n(perivascular <=30µm / adj 30-100 / parenchymal >=100)",fontsize=9.5,fontweight="bold")
axA.legend(fontsize=8,loc="upper center",bbox_to_anchor=(0.5,-0.1),ncol=3,frameon=False)
for sp in ("top","right"): axA.spines[sp].set_visible(False)
def barpanel(ax,r,title):
    intr=r[~r.spillover]; up=intr[intr.log2FC>0].nsmallest(9,"pval"); dn=intr[intr.log2FC<0].nsmallest(9,"pval")
    d=pd.concat([dn,up]).drop_duplicates("gene").sort_values("log2FC"); y=np.arange(len(d))
    cols=[(0.78,0.24,0.20,1 if p<0.05 else .35) if v>0 else (0.12,0.47,0.71,1 if p<0.05 else .35) for v,p in zip(d.log2FC,d.padj)]
    ax.barh(y,d.log2FC,color=cols,edgecolor="#333",lw=0.3)
    for yi,(_,rr) in zip(y,d.iterrows()): ax.text(rr.log2FC+(0.03 if rr.log2FC>0 else -0.03),yi,rr.gene+(" *" if rr.padj<0.05 else ""),va="center",ha="left" if rr.log2FC>0 else "right",fontsize=8)
    ax.axvline(0,color="#333",lw=0.7); ax.set_yticks([]); mm=max(abs(d.log2FC).max(),0.5); ax.set_xlim(-mm*1.9,mm*1.9)
    ax.set_xlabel("log2FC"); ax.set_title(title,fontsize=9.5,fontweight="bold")
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
barpanel(fig.add_subplot(1,3,2),dPP,"Perivascular vs Parenchymal microglia\n(intrinsic; vascular spillover excluded; *padj<0.05)")
barpanel(fig.add_subplot(1,3,3),dAP,"Vessel-adjacent vs Parenchymal microglia\n(intrinsic; vascular spillover excluded; *padj<0.05)")
plt.tight_layout(); fig.savefig(NEW/"microglia_compartments_and_DEG.png",dpi=140,bbox_inches="tight"); plt.close()
CP.to_csv(NEW/"microglia_compartment_composition.csv")
print("\nSaved: microglia_compartments_and_DEG.png + 3 csvs")
