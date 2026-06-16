# MERSCOPE spatial analysis — microglia & T cells in human brain

Spatial transcriptomics (MERSCOPE, 550-gene immune-focused panel) analysis of brain cell
types, T/NK-cell subsets, microglial states, and their spatial interactions in human brain.

> **Scripts only — no data.** Input matrices, coordinates, and donor metadata are
> unpublished/confidential and excluded via `.gitignore`. Paths in the scripts point to the
> local data location. Number prefixes (execution order) have been removed; scripts are
> grouped by theme below.

## Repository structure

### `brain cell clustering/`
All-cell pipeline: scHPF training, ambient **decontamination** + hard-rule lineage
assignment, annotation, and supervised UMAP across all brain cell types
(`schpf_allcell_train`, `decontaminate_and_hardrules`, `decontaminate_all_and_reembed`,
`supervised_umap_decontam*`, `allcell_rebuild`, `allcell_annotate`, `finalize_annotation`).

### `NK-T cell clustering/`
*(GitHub folders can't contain `/`, so "NK/T" → "NK-T".)*
T/NK isolation, reference (pan-tissue T atlas) scHPF projection, and **T-cell subtype
definition** — CD8 TRM 1/2, CD8 TEMRA, CD4 Th/CTL/Tcm-mem/Treg, NK — with UMAPs and
dotplots (`hardrule_TNK`, `prep_reference`, `consensus_schpf`, `project_merscope`,
`Tcell_resubtype`, `define_TRM_subsets`, `final_Tcell_umap`, `rescue_tregs`, `CD4_umap` …).

### `T cell compartment/`
Vascular-compartment assignment of T cells (perivascular ≤30µm / vessel-adjacent /
parenchymal), compartment composition, CD8 DEGs in the perivascular space, and a
vessel-niche zoom figure (`Tcell_compartments`, `compartment_by_celltype`,
`distance_QC_and_niche`, `CD8_PVS_DEG`, `zoom_vessel_niche_hb1R3_30um`).

### `microglia phenotyping/`
Microglial state definition anchored to Green 2024 signatures and **microglia-specific
scHPF**, with QC, clustering, compartment-resolved states and DEGs
(`extract_green_mic_markers.R`, `clean_microglia_schpf_qc`, `within_cluster_qc`,
`clean_spatial`, `rawscore_heatmap_DEGbars`, `pairwise_compartment_DEG`,
`microglia_state_separability`, `BAM_control_supp`, `microglia_density_GM_WM`).

### `microglia-T cell interaction/`
Microglia DEGs in the T-cell niche, **ligand–receptor (co)expression** (gene-level and
spatial), microglial-state enrichment around each T subset, MHC-II machinery near T cells,
and cell–cell **neighborhood enrichment** by compartment (`niche_DEG`,
`microglia_DEG_by_Tsubset`, `TRM_effector_memory_and_Tdeg`, `microglia_T_LR_dotplot`,
`spatial_LR_dotplot`, `spatial_LR_coexpression`, `MHCII_around_Tcells`,
`states_around_Tsubsets`, `Tsubset_zheat_compartment`, `neighborhood_interactions`).

### `utils/`
Shared helpers (`stats_hardening` — multiple-testing / effect-size utilities).

## Inputs the scripts expect (produced by the clustering steps; not in this repo)
- `merged_qc_decontaminated.h5ad` — QC'd, ambient-decontaminated counts
  (`layers/counts`, `layers/counts_decontam`), `obs.cell_type_v2`.
- `Tcell_subset_final_labels.csv` — final T/NK subset labels.
- `QC data/*/cell_metadata.csv` — per-cell centroid coordinates.
- Reference: Green 2024 microglia signatures, pan-tissue T-cell atlas (for projection).

## Environment
Python (conda env `merscope`): `scanpy`, `anndata`, `schpf` (0.5.0), `scipy`, `numpy`,
`pandas`, `matplotlib`, `h5py`, `scikit-learn`, `igraph`/`leidenalg`. R (≥4.6) + `Seurat`
for `extract_green_mic_markers.R`.

## Notes
- Within each folder, scripts run roughly in the order described; most read the shared
  input files above and are otherwise independent.
- Panel blind spots: MHC-I (HLA-A/B/C, B2M, TAP), interferon-stimulated genes, and
  cell-cycle genes are absent from the 550-gene panel.
