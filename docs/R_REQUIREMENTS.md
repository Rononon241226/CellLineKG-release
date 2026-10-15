# R requirements

The KEGG scripts were run with R 4.4.2 and the following packages:

| Package | Version used |
| --- | --- |
| `clusterProfiler` | 4.14.6 |
| `org.Hs.eg.db` | 3.20.0 |
| `dplyr` | 1.1.4 |
| `ggplot2` | 4.0.1 |

Install the Bioconductor and CRAN packages in R:

```r
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

BiocManager::install(c("clusterProfiler", "org.Hs.eg.db"))
install.packages(c("dplyr", "ggplot2"))
```
