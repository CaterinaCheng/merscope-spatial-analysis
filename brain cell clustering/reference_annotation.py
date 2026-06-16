"""
291c_reference_annotation.py  (Stage 1b, reference-based label transfer — user choice)
Annotate ALL cells by CellTypist label transfer (Adult_Human_PrefrontalCortex brain model),
majority-voting-refined; map 113 fine labels -> broad lineages. Evaluate microglia/T recall
vs prior annotation. This replaces de-novo Leiden (which scattered microglia via myelin ambient).
"""
import sys, re
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, celltypist
from celltypist import models
sc.settings.verbosity=0
CMAP=Path(r"<MERSCOPE_ROOT>\merged_analysis\cellmap"); SAVE=Path(r"<MERSCOPE_ROOT>\merged_analysis\scHPF")
A=sc.read_h5ad(CMAP/"allcell_rebuild.h5ad")
A.X=A.layers["counts"].copy(); sc.pp.normalize_total(A,target_sum=1e4); sc.pp.log1p(A)

def broad(l):
    if l.startswith("Astro"): return "Ast"
    if l.startswith("B "): return "B"
    if l.startswith("COP") or l.startswith("OPC"): return "OPC"
    if l.startswith("Endo"): return "End"
    if l.startswith("InN"): return "Inh"
    if re.match(r"^L\d",l): return "Exc"
    if l.startswith("Macro") or l.startswith("Myeloid"): return "Mono/Mac"
    if l.startswith("Micro"): return "Mic"
    if l.startswith("Oligo"): return "Oli"
    if l.startswith("PC "): return "Per"
    if l.startswith("RB "): return "RBC"
    if l.startswith("SMC"): return "SMC"
    if l.startswith("T "): return "T/NK"
    if l.startswith("VLMC"): return "VLMC"
    return "Other"

mdl=models.Model.load("Adult_Human_PrefrontalCortex.pkl")
pred=celltypist.annotate(A,model=mdl,majority_voting=True)
P=pred.predicted_labels
A.obs["ct_percell"]=P["predicted_labels"].values
A.obs["ct_major"]=P["majority_voting"].values if "majority_voting" in P else P["predicted_labels"].values
A.obs["lineage_percell"]=[broad(x) for x in A.obs["ct_percell"]]
A.obs["lineage_major"]=[broad(x) for x in A.obs["ct_major"]]
old=A.obs["old_v2"].astype(str)

for mode in ["lineage_percell","lineage_major"]:
    lab=A.obs[mode].astype(str)
    print(f"\n=== {mode} ===")
    print(lab.value_counts().to_string())
    for ct in ["Mic","T/NK","Mono/Mac"]:
        r=set(A.obs_names[lab==ct]); o=set(A.obs_names[old==ct]); inter=len(r&o)
        print(f"  {ct}: ref={len(r)} old={len(o)} overlap={inter} recall_of_old={inter/max(len(o),1):.2f} Jaccard={inter/max(len(r|o),1):.2f}")

A.obs[["old_v2","ct_percell","ct_major","lineage_percell","lineage_major"]].to_csv(SAVE/"allcell_reference_annotation.csv")
A.write(CMAP/"allcell_rebuild.h5ad")
print("\nSaved: allcell_reference_annotation.csv (+ updated h5ad)")
