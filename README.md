# spliceai2-ism

In-silico saturation mutagenesis (ISM) of gene haplotypes with
[SpliceAI2](https://github.com/Illumina/SpliceAI2). For every haplotype of a
gene (any combination of SNVs, MNVs and indels), it predicts the haplotype
sequence, then scores every possible SNV in the haplotype's gene body
against it (three per base) and reports how each SNV changes the
splice-site and junction probabilities.

SNVs are scored incrementally: one full pass of the haplotype keeps every
block's activations, and each SNV then recomputes only the part of the
network it can change, a cone of ±56,480 nt for the released models (see
[How it works](#how-it-works)). Results equal full forward passes up to
FP16 rounding.

The model code is an independent inference-only PyTorch implementation
written from the reference repository. It keeps the reference module names so
the released checkpoints load with `strict=True`. Weights come from the gated
Hugging Face repo [`illumina-ai/SpliceAI2`](https://huggingface.co/illumina-ai/SpliceAI2),
under the SpliceAI2 Model Terms of Use (non-commercial). Both models of the
released ensemble are used, with FP16 autocast and `torch.compile`, as in the
reference scorer.

## Setup

```bash
uv sync
uv run hf auth login   # once; accept the model terms on Hugging Face first
```

The weights are downloaded into the Hugging Face cache on first use.
`torch.compile` needs a C compiler and a writable `$HOME`.

## Usage

```bash
uv run spliceai2-ism --variants haplotypes.tsv --regions genes.tsv \
    --fasta GRCh38.fa --out-dir results/ism-run1
sbatch scripts/ism.sbatch --variants ... --regions ... --fasta ... --out-dir ...
```

| Option | Default | |
|---|---|---|
| `--genes` | all in the variant TSV | optional comma-separated subset, e.g. to split a run over a SLURM job array |
| `--variant-batch-size` | `auto` | SNVs per incremental batch; `auto` picks the largest that fits in GPU memory (see below) |
| `--out-2d-len` | 1024 | candidate sites for the junction head (reference default) |
| `--min-delta` | 1e-3 | smallest \|Δ probability\| written to `deltas.parquet` |
| `--length-bucket` | 16,384 | gene lengths are rounded up to a multiple of this, so genes share compiled graphs |
| `--debug` | off | run eagerly, without `torch.compile` |

### Batch size

With `--variant-batch-size auto`:

- **Measurement:** before the run, SNV batches of 1 and 5 are scored eagerly
  on the longest gene, and their peak GPU memory gives a fixed part plus a
  cost per SNV. This takes seconds and needs no compilation.
- **Choice:** the largest batch whose estimated peak fits in 80% of the
  memory available to the process. It is rounded down to a multiple of 8 and
  capped at 256, beyond which throughput no longer improves. The choice and
  the measurements are printed and stored in `run.json`.
- **Fallback:** if a batch still runs out of memory, the batch size is
  halved for that batch and all later ones. Halving recompiles the SNV stage
  once. This also applies to a fixed `--variant-batch-size`.

### Inputs

**Regions** (`--regions`), one row per gene; coordinates 1-based, inclusive:

```
NAME	CHROM	STRAND	TX_START	TX_END
BRCA1	chr17	-	43044295	43125483
```

To make it for all protein-coding genes of a GENCODE GFF3 (plain or gzipped):

```bash
uv run spliceai2-regions gencode.v49.annotation.gff3.gz -o genes.tsv
```

It writes one row per `gene` feature with `gene_type=protein_coding`, using
the gene's span (all of its transcripts). Names are `gene_name`; a name that
occurs more than once (e.g. pseudoautosomal genes, which are on both chrX and
chrY) becomes `gene_name_gene_id` for each of those genes.

**Variants** (`--variants`), one row per variant of a haplotype; `POS`
1-based, REF and ALT as in VCF:

```
CHROM	POS	REF	ALT	group	gene
chr17	43045000	A	G	1	BRCA1
chr17	43071077	T	TGA	1	BRCA1
chr17	43045000	A	G	2	BRCA1
chr17	43090010	CAGT	C	2	BRCA1
```

- A haplotype is a `(gene, group)` pair: all of that group's variants in
  that gene, applied together.
- `gene` must match a `NAME` in the region file.
- Group ids per gene run 0, 1, 2, ... If they start at 1, group 0 is the
  haplotype without variants (the reference sequence). Other gaps are an
  error.
- The genes that run are the ones in this file's `gene` column. The region
  file is only looked up for their coordinates, so it can be a full
  annotation; genes without variants are not run.
- Variants can be SNVs, MNVs, insertions, deletions or complex changes:
  REF and ALT are any non-empty sequences of A, C, G, T, with indels
  written as in VCF, including the preceding base (insertion `T` → `TGA`,
  deletion `CAGT` → `C`). Symbolic alleles (`<DEL>`, `*`, `-`) are not
  accepted.
- How a variant is applied: ALT's bases pair with REF's from the left and
  replace them. Extra ALT bases are inserted after the last REF base, and
  extra REF bases are deleted.
- REF must match the FASTA over its whole length. Variants of one haplotype
  may not overlap, e.g. an SNV inside a deletion.
- Variants in the flanking context count too. Variants too far from the
  gene to affect it are ignored with a warning.
- Header names are case-insensitive and may start with `#`.

**FASTA** (`--fasta`): the GRCh38 reference, read with pyfaidx (it writes a
`.fai` index next to the FASTA if there is none). The model always gets the
human (GRCh38) species channel.

### Outputs

Four Parquet tables, plus `run.json`. All values are probabilities
(sigmoid of the model's logits).

**Coordinates.** Every base of a haplotype has two positions, both 1-based
and forward strand:

- `haplotype_pos`: its position on the haplotype's own chromosome, the
  reference chromosome with all of the haplotype's variants applied. It
  equals the reference position up to the first indel, and is shifted by
  every indel before it.
- `reference_pos`: its GRCh38 position; null for bases the haplotype
  inserts. Bases the haplotype deletes are absent: no predictions, no SNVs.

For `1000 A → ATG` with no indels before it, the bases are A `(1000, 1000)`,
T `(null, 1001)`, G `(null, 1002)`, and the next reference base
`(1001, 1003)` as `(reference_pos, haplotype_pos)`. Haplotypes line up on
`reference_pos`. A haplotype's gene body is its bases at
`TX_START`–`TX_END`, plus bases inserted after `TX_START`–`TX_END − 1`; an
insertion right after `TX_END` is flank.

**`predictions_sites.parquet`**: one row per gene-body base of each
haplotype.

| Column | |
|---|---|
| `gene`, `group` | the haplotype |
| `reference_pos`, `haplotype_pos` | the base |
| `donor_prob`, `acceptor_prob` | splice-site probabilities |

**`predictions_junctions.parquet`**: one row per donor-acceptor pair of the
haplotype's top-`--out-2d-len` candidate sites that lies inside the gene
(about 524k pairs at 1024).

| Column | |
|---|---|
| `gene`, `group` | the haplotype |
| `donor_reference_pos`, `donor_haplotype_pos` | the donor |
| `acceptor_reference_pos`, `acceptor_haplotype_pos` | the acceptor |
| `prob` | junction probability |

**`deltas_sites.parquet`** and **`deltas_junctions.parquet`**: per haplotype
and SNV, the signed change against the haplotype,
`delta = P(SNV) − P(haplotype)`. They have the same columns as the
prediction tables, with deltas instead of probabilities (`donor_delta`,
`acceptor_delta`; `delta`), plus the SNV:

| Column | |
|---|---|
| `snv_reference_pos`, `snv_haplotype_pos` | the SNV's base |
| `snv_ref`, `snv_alt` | `snv_ref` is the haplotype's base (at a haplotype variant that is its ALT, so one SNV reverts to REF) |

A site row is written when its donor or acceptor delta reaches
`--min-delta`; a junction row when its delta does. An SNV with no change
above the threshold has no rows. Every base of the haplotype's gene body
that is A, C, G or T is saturated, inserted bases included, with all three
other bases. Junctions are keyed by their donor and acceptor positions. An
SNV can change the top-k candidate sites, so a pair predicted for only one
of SNV and haplotype counts as probability 0 for the other.

**`run.json`**: configuration, environment and per-haplotype timings.

Example, with DuckDB:

```sql
SELECT snv_reference_pos, snv_alt, reference_pos, donor_delta, acceptor_delta
FROM 'results/ism-run1/deltas_sites.parquet'
WHERE gene = 'BRCA1' AND "group" = 1
ORDER BY greatest(abs(donor_delta), abs(acceptor_delta)) DESC LIMIT 20;
```

### Strand

Minus-strand genes are reverse-complemented for prediction, as SpliceAI-style
models score the transcribed strand. Outputs are mapped back to the forward
strand: both positions are forward strand, and `snv_ref` / `snv_alt` are
forward-strand bases. Donor and acceptor keep their biological meaning, so
on the minus strand a junction's donor lies at a higher coordinate than its
acceptor.

### Limits

- One forward pass per gene, no tiling: the gene plus 131,072 nt of context
  must fit in GPU memory (about 3 GiB at 150k nt input, growing roughly
  linearly).
- Genes shorter than `--out-2d-len` nt are scored too. Their candidate
  sites are chosen among the first `--out-2d-len` positions from the gene
  start (in transcript orientation), so they include sequence past the
  gene's 3' end. Only junctions with both sites in the gene are reported.
- A 100 kb gene has about 300,000 SNVs per haplotype. Throughput is logged
  per haplotype (SNVs/s); the first haplotype of each length bucket also
  pays for `torch.compile`.

## How it works

`spliceai2_ism.incremental.IncrementalEnsemble`:

- **The cone:** a substitution can only change activations inside a window
  that widens block by block. For the released models that is ±56,480 nt
  after the last block, within the 65,536 nt flank.
  - One full pass of the haplotype keeps every block's pooled output.
  - Each SNV then recomputes only the block outputs inside its cone (about
    1.3% of a full pass at 100k), reading the context around them from the
    cache.
- **Three rewrites, all exact in exact arithmetic:**
  - Splice-site heads are applied before the linear upsampling, and an SNV's
    track is only stitched together inside its cone.
  - Junction-head features are interpolated straight from the pooled block
    outputs.
  - Each block's input window carries one token-mixer radius of context per
    side; only the outputs that can change are computed from it, so the FFN
    never runs on that context.
- **Top-k per SNV:** the junction head picks each SNV's own top-k sites from
  its full splice-site track, as a full pass would.

## Tests

```bash
uv run pytest           # CPU, small random models
uv run pytest -m gpu    # released models on the GPU vs full forward passes
```

The CPU tests run the whole pipeline on a synthetic chromosome, with one gene
per strand, and check every reported probability against independent full
forward passes. The GPU test does the same with the released weights,
compiled and in debug mode, with a tolerance of 5e-4 for FP16 rounding. It
needs the Hugging Face login and takes several minutes, mostly compilation.

## Benchmarks

The runtime benchmarks this project started from are in
[`benchmarks/`](benchmarks/README.md) (`spliceai2-bench`,
`spliceai2-mutbench`, `spliceai2-suite`).
