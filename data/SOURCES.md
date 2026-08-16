# Data freeze

`ipf_matrix.parquet` is 4,283 Open Targets IPF-associated genes (`EFO_0000768`)
with 18 public analogue scores, a binary clinical-stage mechanism label
(`y_clinical`, 139 positives), and the Open Targets overall association score
(comparator only; not in X).

STRING files are the induced subgraph on that gene list (combined score ≥ 700).

## Attribution

| Source | What is used | License (upstream) |
| --- | --- | --- |
| [Open Targets](https://platform.opentargets.org/) | Gene universe, overall score, clinical MoA label, several analogue channels | See Open Targets terms |
| [STRING v12](https://string-db.org/) | PPI edges and combined scores | CC BY 4.0 |
| GEO / GTEx / HPA / Reactome / LINCS / DepMap / GWAS Catalog / OmniPath | Public analogue score inputs (precomputed in the parquet) | Each resource's own terms |

This freeze is the induced gene matrix and STRING subgraph used in the ranking
experiment.
