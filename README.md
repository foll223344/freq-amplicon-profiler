# freq — per-position substitution frequency profiles from amplicon deep sequencing

`freq.py` computes and plots per-position substitution frequencies (mutations per
thousand reads) across a genomic window from a table of variant calls, as used for
APOBEC3A-induced C>T / G>A profiling of PCR amplicons.

## Input

A headerless TSV with six columns:

```
CHROM   POS   REF   ALT   SAMPLE   AD
```

where `AD` is the comma-separated per-allele read depth field emitted by
`bcftools mpileup -a FORMAT/AD` (reference depth first).

## Calculation

For every position, frequency = (ALT_DP / TOTAL_DP) x 1000, where TOTAL_DP is the
sum of all allelic depths at that position. Several alternative alleles at one
position are summed. Records below `--noise-threshold` are discarded; positions
with no retained record are set to zero. The per-amplicon mean rate is the
arithmetic mean over all positions of the window, including zeros, and is written
to `summary*.tsv`.

## Usage

```bash
python freq.py \
  --input group5.tsv \
  --ref-genome Homo_sapiens.GRCh38.dna.primary_assembly.fa \
  --region 17:7674963-7675135 \
  --output-dir A3A_Gr5 \
  --noise-threshold 0.1 \
  --multi-sample --plot-type area \
  --y_max 100 --svg-format
```

Key options:

| option | meaning |
| --- | --- |
| `--region chr:start-end` | window to profile (1-based, inclusive) |
| `--noise-threshold` | minimum mutations per thousand reads to keep a record |
| `--y_max` / `--y_min` | hard Y-axis limits, shared across panels of a group |
| `--plot-type bar\|area` | per-position bars or smoothed filled line |
| `--smooth-sigma` | Gaussian sigma for area plots (display only) |
| `--multi-sample` | overlay several samples on one panel |
| `--motif1` / `--motif2` | highlight IUPAC motifs, searched on both strands |
| `--log-scale`, `--dpi`, `--svg-format` | output scaling and format |

Outputs: PNG (300 dpi) and optional SVG plots, `summary*.tsv` with mean rates, and
`axis_labels*.tsv` with the per-position values behind each plot.

## Requirements

Python 3.12, numpy 1.26, pandas 2.2, scipy 1.12, matplotlib 3.10, biopython 1.85.

## Citation

If you use this script, please cite [paper reference to be added].

## License

MIT — see `LICENSE`.
