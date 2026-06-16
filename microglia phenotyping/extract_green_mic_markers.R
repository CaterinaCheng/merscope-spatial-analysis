#!/usr/bin/env Rscript
# 330_extract_green_mic_markers.R
# Extract Green2024 microglial STATE annotations + per-state mean expression (on our panel
# genes) from microglia.seurat.rds, to build literature-anchored microglial-state signatures.
suppressPackageStartupMessages({ library(Seurat); library(SeuratObject) })
REF <- "<MERSCOPE_ROOT>/merged_analysis/Green2024/reference/microglia.seurat.rds"
OUT <- "<MERSCOPE_ROOT>/merged_analysis/scHPF/new"
cat("loading microglia reference...\n"); t0 <- Sys.time()
obj <- readRDS(REF); cat("loaded in", format(Sys.time()-t0), " cells:", ncol(obj), "genes:", nrow(obj), "\n")
md <- obj@meta.data
cat("\nmeta.data columns:\n"); print(colnames(md))
lab <- grep("state|subtype|cluster|annotation|celltype|cell_type|label", colnames(md), value=TRUE, ignore.case=TRUE)
cat("\nlabel-like columns + distributions:\n")
for (col in lab) { cat("--- ", col, " ---\n", sep=""); print(head(sort(table(md[[col]]), decreasing=TRUE), 20)) }

# choose the finest state column that looks like Mic.X
statecol <- NULL
for (col in lab) { if (any(grepl("^Mic\\.", as.character(md[[col]])))) { statecol <- col; break } }
cat("\nchosen state column:", statecol, "\n")
if (!is.null(statecol)) {
  Idents(obj) <- md[[statecol]]
  DefaultAssay(obj) <- names(obj@assays)[1]
  cat("assay used:", DefaultAssay(obj), "\n")
  # mean expression per state (data slot = log-normalized)
  ae <- AverageExpression(obj, assays=DefaultAssay(obj), slot="data")[[1]]
  ae <- as.data.frame(ae)
  write.csv(ae, file.path(OUT, "green_mic_state_mean_expr.csv"))
  cat("wrote green_mic_state_mean_expr.csv  dim:", nrow(ae), "x", ncol(ae), "\n")
  # also write the state sizes
  write.csv(as.data.frame(table(md[[statecol]])), file.path(OUT,"green_mic_state_sizes.csv"), row.names=FALSE)
}
cat("done\n")
