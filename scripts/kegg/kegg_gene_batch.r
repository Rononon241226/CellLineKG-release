# ============================================================
# KEGG Enrichment Analysis for Cell Line Gene Expressions
# ============================================================
# Input:
#   - data/processed/neo4j_import/nodes/nodes_cellline.csv: graph cell lines
#   - data/intermediate/schema/depmap/nodes_cellline.csv: cell line metadata
#   - data/intermediate/normalized/depmap/cell_line_lineages.csv: lineage metadata
#   - data/intermediate/normalized/depmap/gene_expressions_selected.csv: expression data
# Output:
#   - results/kegg/gene/{cell_name}_{level}_kegg.csv: enrichment results
#   - results/kegg/gene/{cell_name}_{level}_kegg_barplot.png: PNG bar plot
#   - results/kegg/gene/{cell_name}_{level}_kegg_barplot.pdf: PDF bar plot
#   - results/kegg/gene/_summary_all.csv: combined summary
# ============================================================


library(clusterProfiler)
library(org.Hs.eg.db)
library(dplyr)
library(ggplot2)

# ============================================================
# >>>>> CONFIGURATION — Modify this section as needed <<<<<
# ============================================================

# FDR threshold (Benjamini-Hochberg adjusted p-value)
fdr_cutoff <- 0.05

# Minimum high-expression gene-set size required for KEGG enrichment.
min_gene_count <- 100

# Minimum significant KEGG pathways required to save and plot a result.
min_sig_pathways <- 10

# Output switches.
# Input: logical flags controlling figure export.
# Output: PDF-only export by default, preserving existing PNG files.
write_png <- FALSE
write_pdf <- TRUE

# Cell lines to analyze
# Supports: depmap_model_id (e.g. "ACH-000019"),
#           cell line name (e.g. "MCF7"),
#           or RRID (e.g. "CVCL_0031")


# File paths (relative to project root)
project_root <- getwd()
cell_file <- file.path(project_root, "data", "processed", "neo4j_import", "nodes", "nodes_cellline.csv")
nodes_file <- file.path(project_root, "data", "intermediate", "schema", "depmap", "nodes_cellline.csv")
lineage_file <- file.path(project_root, "data", "intermediate", "normalized", "depmap", "cell_line_lineages.csv")
expr_file <- file.path(project_root, "data", "intermediate", "normalized", "depmap", "gene_expressions_selected.csv")
output_dir <- file.path(project_root, "results", "kegg", "gene")
pdf_output_dir <- output_dir


# ============================================================
# Load data
# ============================================================
message("=== Loading data ===")
if (!file.exists(nodes_file)) stop("nodes_cellline.csv not found: ", nodes_file)
if (!file.exists(lineage_file)) stop("cell_line_lineages.csv not found: ", lineage_file)
if (!file.exists(expr_file))  stop("gene_expressions_selected.csv not found: ", expr_file)

cells <- read.csv(cell_file, stringsAsFactors = FALSE)[, c("cellosaurus_id", "name")]
nodes <- read.csv(nodes_file, stringsAsFactors = FALSE)
lineages <- read.csv(lineage_file, stringsAsFactors = FALSE)
expr  <- read.csv(expr_file,  stringsAsFactors = FALSE)


message(sprintf("  Cell lines (nodes): %d", nrow(nodes)))
message(sprintf("  Gene records (expr): %d", nrow(expr)))
message(sprintf("  Unique cell lines in expr: %d", length(unique(expr$depmap_model_id))))

# ============================================================
# Resolve cell line identifiers → depmap_model_id
# ============================================================
# The aim of the function: map unified graph cell lines to DepMap IDs and lineage names.
# Arg: cell_df, nodes_df, and lineage_df are input cell-line metadata tables.
# Returns: matched cell lines with RRID, cell name, DepMap ID, and lineage name.
resolve_celllines <- function(cell_df, nodes_df, lineage_df) {
  
  nodes_small <- nodes_df[, c("rrid", "depmap_model_id")]
  
  merged <- merge(
    x = cell_df,
    y = nodes_small,
    by.x = "cellosaurus_id",
    by.y = "rrid",
    all.x = TRUE
  )
  merged <- merge(
    x = merged,
    y = lineage_df[, c("depmap_model_id", "lineage_name")],
    by = "depmap_model_id",
    all.x = TRUE
  )
  merged$lineage_name[is.na(merged$lineage_name)] <- ""
  
  matched <- merged[!is.na(merged$depmap_model_id), ]
  
  missing_rrid <- merged$cellosaurus_id[is.na(merged$depmap_model_id)]
  if (length(missing_rrid) > 0) {
    for (id in missing_rrid) {
      warning(sprintf("  [WARN] '%s' not found in nodes_cellline.csv", id))
    }
  }
  
  if (nrow(matched) == 0) {
    return(NULL)
  }
  
  return(matched)
}

selected <- resolve_celllines(cells, nodes, lineages)

if (is.null(selected)) {
  stop("No valid cell lines matched. Check your selected_cells.")
}

message(sprintf("\n=== %d cell line(s) matched ===", nrow(selected)))
for (i in seq_len(nrow(selected))) {
  message(sprintf("  %d. %s | %s | %s",
    i, selected$name[i], selected$depmap_model_id[i], selected$cellosaurus_id[i]))
}

# ============================================================
# Create output directory
# ============================================================
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(pdf_output_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# KEGG enrichment runner
# ============================================================
# The aim of the function: run KEGG enrichment for a high-expression gene set.
# Arg: entrez_ids, level, cell_name, and cell_id identify one enrichment input.
# Returns: significant KEGG pathways when at least min_sig_pathways are found.
run_kegg <- function(entrez_ids, level, cell_name, cell_id) {
  gene_count <- length(entrez_ids)

  if (gene_count == 0) {
    message(sprintf("    [%s] No genes, skipping.", level))
    return(NULL)
  }

  if (gene_count < min_gene_count) {
    message(sprintf(
      "    [%s] Only %d genes (< %d), skipping KEGG enrichment.",
      level, gene_count, min_gene_count
    ))
    return(NULL)
  }

  message(sprintf("    [%s] %d genes -> enrichKEGG...", level, gene_count))

  result <- tryCatch({
    enrichKEGG(
      gene          = entrez_ids,
      organism      = "hsa",
      pvalueCutoff  = 1.0,       # keep raw results, filter later
      pAdjustMethod = "BH",
      qvalueCutoff  = 1.0
    )
  }, error = function(e) {
    warning(sprintf("    [%s] enrichKEGG failed: %s", level, e$message))
    return(NULL)
  })

  if (is.null(result) || nrow(as.data.frame(result)) == 0) {
    message(sprintf("    [%s] No enriched KEGG pathways found.", level))
    return(NULL)
  }

  df <- as.data.frame(result)

  # Apply FDR filter
  df_sig <- df[df$p.adjust < fdr_cutoff, ]

  if (nrow(df_sig) < min_sig_pathways) {
    message(sprintf(
      "    [%s] %d significant pathways (FDR < %.2f) out of %d total; fewer than %d, skipping result output.",
      level, nrow(df_sig), fdr_cutoff, nrow(df), min_sig_pathways
    ))
    return(NULL)
  }

  message(sprintf(
    "    [%s] %d significant pathways (FDR < %.2f) out of %d total; saving result.",
    level, nrow(df_sig), fdr_cutoff, nrow(df)
  ))

  # Save CSV
  safe_name <- gsub("[/:\\*\\?\"<>| ]", "_", cell_name)
  out_csv <- file.path(output_dir,
    sprintf("%s_%s_%s_kegg.csv", cell_id, safe_name, level))
  write.csv(df_sig, out_csv, row.names = FALSE)
  message(sprintf("    Saved to: %s", out_csv))
  
  message("\n    Count gene count summary:")
  print(summary(df_sig$Count))

  # Bar plot
  if (nrow(df_sig) >= 3) {
    out_png <- file.path(output_dir,
      sprintf("%s_%s_%s_kegg_barplot.png", cell_id, safe_name, level))
    out_pdf <- file.path(pdf_output_dir,
      sprintf("%s_%s_%s_kegg_barplot.pdf", cell_id, safe_name, level))

    plot_data <- df_sig[order(df_sig$p.adjust), ]
    if (nrow(plot_data) > 10) plot_data <- plot_data[1:10, ]
    plot_data$Description <- factor(plot_data$Description, levels = rev(plot_data$Description))

    p <- ggplot(plot_data, aes(x = Count, y = Description, fill = p.adjust)) +
      geom_col(width = 0.72) +
      scale_fill_gradient(low = "#ff0303", high = "#0341fa") +
      labs(
        title = sprintf("%s: KEGG enrichment of highly expressed genes", cell_name),
        x = "Gene count",
        y = NULL,
        fill = "FDR"
      ) +
      theme_minimal(base_size = 12) +
      theme(
        plot.title = element_text(size = 14, face = "plain", colour = "black"),
        axis.text = element_text(size = 14, face = "plain", colour = "black"),
        axis.title = element_text(size = 14, face = "plain", colour = "black"),
        legend.text = element_text(size = 14, face = "plain", colour = "black"),
        legend.title = element_text(size = 14, face = "plain", colour = "black"),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank()
      )

    plot_width <- 10
    plot_height <- max(6, nrow(plot_data) * 0.55 + 1.5)

    if (write_png) {
      # Save a PNG image with stable dimensions for pathway labels.
      ggsave(out_png, p, width = plot_width, height = plot_height, dpi = 300, device = "png", limitsize = FALSE)
      message(sprintf("    Bar plot saved to: %s", out_png))
    }

    if (write_pdf) {
      # Save a compact PDF image for stacked figure layout.
      ggsave(out_pdf, p, width = plot_width, height = plot_height, device = "pdf", limitsize = FALSE)
      message(sprintf("    Compact PDF saved to: %s", out_pdf))
    }
  }

  return(df_sig)
}

# ============================================================
# Main: process each cell line
# ============================================================
all_summaries <- list()

for (i in seq_len(nrow(selected))) {
  cell_id   <- selected$depmap_model_id[i]
  cell_name <- selected$name[i]

  message(sprintf("\n========================================"))
  message(sprintf("=== %s (%s) [%d/%d] ===", cell_name, cell_id, i, nrow(selected)))
  message(sprintf("========================================"))

  cell_genes <- expr[expr$depmap_model_id == cell_id, ]

  if (nrow(cell_genes) == 0) {
    message(sprintf("  No expression data for %s, skipping.", cell_id))
    next
  }

  message(sprintf("  Total selected genes: %d", nrow(cell_genes)))

  # High expression genes
  high_entrez <- unique(cell_genes[cell_genes$level == "high", "entrez_id"])
  high_entrez <- high_entrez[!is.na(high_entrez)]

  message(sprintf("  High: %d genes", length(high_entrez)))

  res_high <- run_kegg(high_entrez, "high", cell_name, cell_id)

  # Collect summary
  for (level in c("high")) {
    res <- res_high
    if (!is.null(res) && nrow(res) > 0) {
      all_summaries[[length(all_summaries) + 1]] <- data.frame(
        lineage_name    = selected$lineage_name[i],
        rrid            = selected$cellosaurus_id[i],
        cell_name       = cell_name,
        depmap_model_id = cell_id,
        expression_level = level,
        pathway_id      = res$ID,
        pathway_desc    = res$Description,
        gene_ratio      = res$GeneRatio,
        bg_ratio        = res$BgRatio,
        pvalue          = res$pvalue,
        p_adjust        = res$p.adjust,
        qvalue          = res$qvalue,
        gene_ids        = res$geneID,
        count           = res$Count,
        stringsAsFactors = FALSE
      )
    }
  }
}

# ============================================================
# Combined summary
# ============================================================
if (length(all_summaries) > 0) {
  summary_all <- do.call(rbind, all_summaries)
  summary_csv <- file.path(output_dir, "_summary_all.csv")
  write.csv(summary_all, summary_csv, row.names = FALSE)
  message(sprintf("\n=== Combined summary: %d pathway records → %s ===",
    nrow(summary_all), summary_csv))
} else {
  message("\n=== No significant enrichment found for any cell line. ===")
}

message("\n=== All done! ===")


