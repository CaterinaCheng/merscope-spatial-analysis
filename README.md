# Microglia–T cell spatial interactions in human brain (MERSCOPE)

Spatial transcriptomics analysis of T cell ↔ microglia interactions in human brain
(MERSCOPE, 550-gene immune-focused panel). Analysis code for characterizing microglial
states, their vascular-compartment organization (perivascular / vessel-adjacent /
parenchymal), and their interactions with T-cell subsets.

> **Data is not included.** This repository contains analysis **scripts only**. The input
> data (cell × gene matrices, coordinates, donor metadata) are unpublished/confidential and
> are excluded via `.gitignore`. Paths in the scripts point to the local data location.

## Inputs the scripts expect (produced upstream, not in this repo)
- `merged_qc_decontaminated.h5ad` — QC'd, ambient-decontaminated counts (`layers/counts`,
  `layers/counts_decontam`), `obs.cell_type_v2`.
- `Tcell_subset_final_labels.csv` — final T/NK subset labels
  (CD8 TRM 1/2, CD8 TEMRA, CD4 Th/CTL/Tcm-mem/Treg, NK).
- `QC data/*/cell_metadata.csv` — per-cell centroid coordinates (EntityID, center_x/y).
- Reference: Green 2024 microglia signatures, Tuddenham 2024 supplementary tables.

## Environment
Python (conda env `merscope`): `scanpy`, `anndata`, `schpf` (0.5.0), `scipy`, `numpy`,
`pandas`, `matplotlib`, `h5py`, `scikit-learn`. R (≥4.6) with `Seurat` for the marker
extraction script.

## Pipeline (by theme)

**T-cell vascular compartments**
- `305_Tcell_compartments.py` — assign T cells to vascular compartments; subset composition per compartment.

**Microglia DEG in the T-cell niche**
- `322_niche_DEG.py`, `323`–`327` — microglia DEGs near T cells, by T lineage / subset.

**Microglial state definition (reference-anchored)**
- `330_extract_green_mic_markers.R` — extract Green 2024 microglial state signatures (Seurat).
- `328`, `331`, `334`, `335` — Green-state scoring, compartment comparison, BAM control.
- `337`–`339` — reference label-transfer attempts (CellTypist / SingleR / scHPF factor mapping).
- `340`–`345` — Tuddenham 2024 subtype mapping and UMAPs.

**Microglia clustering via scHPF + QC (final)**
- `346`, `349`, `350`, `351` — microglia-specific scHPF, Green annotation, mixed-cluster handling.
- `347`, `348`, `352` — state separability diagnostics, cluster QC / contamination.
- `353_clean_microglia_schpf_qc.py` — QC-gate to bona-fide microglia, retrain scHPF, annotate.
- `354_within_cluster_qc.py` — within-cluster purity QC.
- `358_loose_microglia_schpf.py` — loosened-gate sensitivity.
- `370_regen_clean_qc_fig.py` — regenerate QC figure from saved tables.

**Spatial: state × compartment**
- `333`, `355`, `356`, `359`, `360`–`362`, `364`, `366` — state composition / enrichment by
  vascular compartment, compartment DEGs (raw + z-scored, pairwise).

**Microglia state around T-cell subsets**
- `357`, `365` — microglial state enrichment near each T subset.
- `363_MHCII_around_Tcells.py` — MHC-II machinery in microglia near each T subset (full set).

**Ligand–receptor**
- `367`–`369` — microglia↔T L–R coexpression and spatial co-expression / enrichment.

**Focused / interaction analyses**
- `371_CD8_PVS_DEG.py` — CD8 T-cell DEGs in the perivascular space.
- `372_neighborhood_interactions.py` — cell–cell neighborhood enrichment by compartment
  (T–astro, T–neuron, microglia–neuron, microglia–astro, etc.).

## Notes
- Scripts are numbered roughly in execution order; many are independent and read the shared
  input files above.
- Panel blind spots to keep in mind: MHC-I (HLA-A/B/C, B2M, TAP), interferon-stimulated
  genes, and cell-cycle genes are absent from the 550-gene panel.
