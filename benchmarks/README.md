# Benchmarks

GPU runtime benchmarks for SpliceAI2 inference, using the model and
incremental code in `spliceai2_ism`. They are installed with the project
(`uv sync` at the repository root) and their tests run with `uv run pytest`.

```bash
uv run spliceai2-bench                                   # 100 × 100,000, 1D + 2D
uv run spliceai2-bench --no-compile                      # eager baseline
uv run spliceai2-bench --batch-size 4 --cudnn-benchmark  # tuning knobs
uv run spliceai2-bench --heads 1d --json-out results.json
uv run spliceai2-mutbench                                # reference + 99 substitutions
uv run spliceai2-suite                                   # all of the above
sbatch benchmarks/h100_suite.sbatch                      # on a SLURM cluster
```

If `--model-dir` is omitted, both model folders
(`spliceai2_multispecies_13M_01`, `_02`) are downloaded into the HF cache.

## spliceai2-bench: full forward passes

Every run times two workloads in the same process:

- **fixed:** `--n-seqs` sequences of exactly `--seq-len` output positions,
  with batch size `--batch-size`.
- **variable:** `--n-seqs` sequences whose output lengths are drawn from
  N(`--var-mean`, `--var-std`), by default N(100,000, 10,000) with seed
  `--seed + 1`. Lengths are rounded so the model accepts them. These run at
  batch size 1. The report adds a linear fit of per-sequence GPU time
  against real input length.

How the variable workload is compiled (`--var-mode`):

- `bucket` (default): each sequence is padded up to a multiple of
  `--bucket-size` output positions (default 8,192), and each bucket size
  gets its own static graph.
  - Padding goes after the right-hand flank. It is all-zero nucleotide
    channels, and it doesn't change the outputs at real positions.
  - The 2D stage ignores padded positions when choosing candidate sites,
    and padded outputs are dropped before the copy back to the CPU.
  - The report shows the padding overhead and the time per bucket.
- `dynamic`: one graph with a dynamic length dimension serves every length,
  with no padding. In testing it was about 24% slower than a static graph on
  identical inputs.

Each workload has its own compile cache, so the fixed workload always keeps
its own static graph.

Measurements are interleaved so that GPU heating cannot favour one workload:

- Within each pass, the two workloads' batches alternate, spread evenly
  across the pass.
- Each round reverses the batch order of the one before (A-B-B-A); the
  default `--repeats 4` balances it.
- GPU-timed and end-to-end passes swap order every round.

Each pass logs GPU temperature, SM clock and power at its start and end
(through NVML).

Before timing, every variable length runs once untimed. This triggers any
remaining compilation, and an out-of-memory error stops the run at that
point rather than during timing. During timing, a recompile raises an error
instead of silently skewing the numbers.

- **Input:** each sequence is `length + crop_len` nt (crop_len = flanking
  context, read from the checkpoint; 131,072 for the released models). The
  model predicts the central `length` positions in one forward pass, the
  same as the reference. `length + crop_len` must be divisible by the
  deepest pooling factor.
- **1D stage:** trunk plus splice-site heads for both models, logits averaged.
- **2D stage:** top-`out_2d_len` candidate sites (default 1024, as in the
  reference scorer), then the junction heads for both models, averaged. Its
  cost is O(out_2d_len²) and does not depend on sequence length.
- **First call:** wall-clock time for the first batch of each distinct batch
  shape, which includes `torch.compile` and cuDNN setup. Reported separately.
- **Steady-state GPU time:** CUDA events around each stage, summed over all
  sequences. Each batch is copied to the GPU before its start event, so the
  copy is not timed. Median, min and max over rounds.
- **End-to-end:** wall clock from Python strings to CPU probabilities. Covers
  one-hot encoding, pinned H2D copy, both stages and D2H copy.
- **Peak GPU memory:** the largest single batch in each workload.

Sequences are random ACGT with the GRCh38 species channel; runtime does not
depend on sequence content.

## spliceai2-mutbench: single-base substitutions

Times `IncrementalEnsemble` (see the main README) against full passes, in
two interleaved workloads:

- **full:** a complete forward pass for the reference and for each variant
  sequence, batch 1.
- **incremental:** one reference pass, then the variants in batches of
  `--variant-batch-size`.

Both workloads use the same code paths, so the comparison isolates the
recomputation that is saved. Before timing, it checks:

- each variant's incremental prediction against its full pass;
- both paths against the original `Ensemble` code, on a short sequence in
  eager mode.

## spliceai2-suite: everything in one command

By default every benchmark makes 500 predictions of 100,000 positions:

| Benchmark | Command used | Workload |
|---|---|---|
| fixed | `spliceai2-bench` | 500 sequences × 100k; once per `--fixed-batch-sizes` entry (default 1 and 8) |
| variable | `spliceai2-bench` | 500 sequences, lengths ~ N(100k, 10k), bucketed |
| random substitutions | `spliceai2-mutbench --layout random` | reference + 499 substitutions at random positions |
| consecutive substitutions | `spliceai2-mutbench --layout consecutive` | reference + 499 substitutions at adjacent positions |

- **Isolation:** each benchmark runs in its own process.
- **Output:** each benchmark's JSON and log, plus `summary.md` and
  `summary.json`, go to `results/suite-<timestamp>/` (or `--out-dir`).
- **Failures:** if one benchmark fails, it is marked in the summary and the
  suite carries on with the next.
- **Time on an H100:** roughly an hour, about a third of it first-time
  `torch.compile`, which the on-disk compile cache speeds up on later runs.

If the cluster's compute nodes have no internet, download the weights on a
login node first:

```bash
uv run hf auth login
uv run python -c "from spliceai2_ism.weights import download_models; print(download_models())"
```

Then either pass the printed folder as `--model-dir`, or share `HF_HOME`
with the compute nodes.
