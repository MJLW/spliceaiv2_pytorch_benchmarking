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
uv run spliceai2-sweep                                   # batch-size sweep, naive vs incremental
uv run spliceai2-ablation                                # each optimization on/off
sbatch benchmarks/h100_suite.sbatch                      # on a SLURM cluster
sbatch benchmarks/h200_sweep.sbatch                      # sweep with profiler traces
sbatch benchmarks/h200_ablation.sbatch                   # ablation on an H200
```

The sbatch scripts set `TORCHINDUCTOR_CACHE_DIR` to `.inductor_cache/` in
the submit directory (unless it is already set), so compiled graphs survive
between jobs instead of living in node-local `/tmp`.

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

Times `IncrementalEnsemble` (see the main README) against naive full passes,
in two interleaved workloads:

- **full (naive):** a complete forward pass (`Ensemble`, the plain
  full-pass code) for the reference and for each variant sequence, in
  batches of `--full-batch-size`. The last batch is padded by repeating its
  final sequence. The padding is timed, but per-prediction rates count only
  real predictions.
- **incremental:** one reference pass, then the variants in batches of
  `--variant-batch-size`.

`--full-batch-size` defaults to `auto` and `--variant-batch-size` to 32
(past which an H200 gains no throughput). `auto` works like
`--variant-batch-size auto` of `spliceai2-ism`:

- two small batches of each path run eagerly, with the incremental
  reference cache resident as it is during timing;
- their peak memory gives a fixed part plus a cost per item;
- `auto` is the largest batch within 80% of the available memory, rounded
  down to a multiple of 8 and capped at the number of items.

The report and JSON also give the *ceiling*: the largest batch the same
estimate allows in all of the memory. Pass an integer to fix a size.

If the second probe batch runs out of memory, the estimate uses the memory
held before probing (models and cache) as its second point. If even batch 1
runs out of memory eagerly, the estimate is 1, and the run still tries it
compiled, which usually needs less memory. The probe never stops the run
itself.

Before timing, it checks:

- each variant's incremental prediction against its full pass;
- both paths against the original model code, on a short sequence in
  eager mode.

With any rewrite enabled it also compares the incremental path with
`--opts` against the same path with none (eager, FP16, on the short
sequence: reference and 32 variants), so each rewrite's own effect on the
outputs is reported separately (`accuracy.vs_unoptimized`).

Other options:

- `--opts SPEC`: which exact rewrites of the incremental path to use (see
  [Optimizations](../README.md#optimizations)): `all` (default), `none`,
  names to enable (`stencil_resample,padded_ffn`), or `all,-name` to
  leave one out. The naive path never uses them.
- `--compile-mode default|reduce-overhead|max-autotune-no-cudagraphs` and
  `--freezing`: torch.compile settings. Under `reduce-overhead` (CUDA
  graphs) the reference cache is copied into static buffers after each
  reference pass, so the variant graphs read it in place.
- `--e2e-output full|deltas`: what the end-to-end passes produce. `full`
  copies probability tracks and junction matrices to the CPU, into reused
  pinned buffers (`--host-copy pinned`, default) or with plain `.cpu()`
  (`--host-copy pageable`). `deltas` computes sparse deltas against the
  reference on the GPU and copies only those, as `spliceai2-ism` does
  (`--min-delta`, default 1e-3; 1D deltas over the whole output region).
  The end-to-end time is also split into `load` (encode, upload), `run`
  (both stages) and `to_host` (output), in the report and in
  `end_to_end_phases_s`.
- `--warm-seconds N`: after compilation, run both workloads untimed for N
  seconds, so timing starts at steady clocks and temperature. Telemetry
  before and after is logged.
- `--only full|incremental`: time one workload only. The accuracy checks
  still run both.
- `--profile-dir DIR`: profile after timing (see below).
- Running out of GPU memory exits with code 3.

## Profiling

`--profile-dir DIR` (on `spliceai2-mutbench` and `spliceai2-bench`) runs
`torch.profiler` over `--profile-iters` warm iterations (default 3, after
one untimed one), once timing is done:

- **mutbench workloads:** `full` (the first three batches),
  `incremental_reference` (the reference pass) and `incremental_variants`
  (two variant batches);
- **bench workload:** `fixed` (its first three batches).

Each writes:

- `<name>.trace.json`: Chrome trace; open it in <https://ui.perfetto.dev>
  or `chrome://tracing`. Stages are named ranges such as `full/1d`,
  `full/2d`, `incremental_variants/var_1d`.
- `<name>.kernels.txt`: operators and kernels by self GPU time.

With `--profile-stacks` it also records Python call stacks and writes, for
GPU and for CPU self time:

- `<name>.flame_gpu.svg`, `<name>.flame_cpu.svg`: flamegraphs. Open them in
  a browser; hover a frame to see its time, click it to zoom, and click the
  root to reset. GPU time is attributed to the operator that launched each
  kernel.
- `<name>.stacks_gpu.txt`, `<name>.stacks_cpu.txt`: the same data as folded
  stacks, for <https://www.speedscope.app> or `flamegraph.pl`.

Under `torch.compile` the stacks end at the compiled graph. Profile once
with `--no-compile` to see time per module. Stack recording slows the
profiled iterations only, not the timed rounds.
`spliceai2-sweep --profile --profile-stacks` passes the flag to every run,
and `h200_sweep.sbatch` uses it.

### Comparing runs: differential flamegraphs

Diffs are written in addition to each run's own flamegraphs, which stay as
they are.

```bash
uv run spliceai2-flamediff A/full.stacks_gpu.txt B/full.stacks_gpu.txt -o diff.svg
```

- **Normalisation:** times are per prediction by default. Each profile
  writes `<name>.meta.json` with its iterations and predictions per
  iteration, so different batch sizes, or naive vs incremental, compare
  like for like. Use `--per iteration` or `--per raw` for the other units.
- **Widths and colours:** frame widths follow B. Red frames are slower in
  B, blue frames faster, and stronger colours mean larger changes; grey is
  unchanged. Hover a frame for A, B and Δ. `--width-from a` draws A's
  widths instead, which shows frames that B no longer has.
- **Which pairs work best:** the same workload at different batch sizes or
  layouts. Naive and incremental run different compiled graphs, so with
  `torch.compile` their diff is mostly frames found in only one of them;
  compare them with `--no-compile` runs.

`spliceai2-sweep --profile-stacks` writes these automatically to `diffs/`:

- on each axis, every run against the axis's first run, per workload;
- consecutive against random, per axis, at the largest batch size both
  layouts ran.

With `--profile`, `sweep.md` also gets a table of the top GPU kernels per
workload, in µs per prediction across the batch sizes, to show which
kernels scale and which don't.

The report and JSON (`profile`) give, per iteration:

- the summed GPU kernel time;
- the CPU operator time, which includes profiler overhead (CPU time close
  to or above GPU time means the workload is launch-bound);
- the number of kernel launches;
- the top kernels.

## spliceai2-sweep: batch sizes, naive vs incremental

Finds how throughput of each path scales with batch size, up to what fits
in GPU memory, on warm GPUs. For each layout in `--layouts` (default both
`random` and `consecutive`) there are two axes:

- **naive:** `--full-batch-size` 1, 2, 4, 8, …, with the incremental
  workload at `auto` in the same run, which also gives a same-run speedup.
- **incremental:** `--variant-batch-size` 1, 2, 4, …, with `--only
  incremental`, so the slow full passes are not timed.

How each axis proceeds:

- **Doubling:** the batch size doubles from `--start-batch` until a run
  runs out of GPU memory or reaches the number of items.
- **Ceiling run:** after an out-of-memory run, one more run uses the
  ceiling of the last successful run, if it lies between the last size that
  fit and the one that did not.
- **Isolation:** every run is its own `spliceai2-mutbench` process, with
  `--warm-seconds` (default 60) and `--repeats` (default 2).

The layouts are compared because they behave differently:

- the naive path costs the same for both, so it serves as a control;
- incremental batches of adjacent substitutions share nearly the same cone,
  while random ones are spread out, so throughput and the largest batch can
  differ.

Output in `results/sweep-<timestamp>/` (or `--out-dir`):

- per-run JSON and logs (and with `--profile`, traces in
  `<run>.profile/`);
- `sweep.json` and `sweep.md`, rewritten after every axis. They hold:
  - the best naive vs best incremental throughput per layout, and the
    speedup;
  - the largest batch that fit and the first that ran out of memory;
  - random vs consecutive;
  - one table per layout and axis.

`benchmarks/h200_sweep.sbatch` runs the sweep with `--profile` on one H200.
Most of the time goes to the naive axis: every run times 500 full forward
passes per round.

## spliceai2-ablation: each optimization on and off

Runs `spliceai2-mutbench` once per configuration and sequence length, each
in its own process: reference + `--n-mutations` (499) `--layout` (random)
substitutions of a sequence of each of `--seq-lens` (10,000, 20,000, 50,000
and 100,000 positions, as genes of those lengths), variant batch 32, only
the incremental workload timed. `--seq-lens N` runs one length.

- `none`, `all`: no rewrite, every rewrite.
- `only_<name>`, `all_minus_<name>`: each rewrite alone, and every rewrite
  but that one. The first gives its gain on its own (over `none`), the
  second its gain on top of the others (`all` over `all_minus_<name>`).
- `all_freezing`, `all_reduce_overhead`, `all_max_autotune`: compile
  settings on top of `all`.
- `all_pageable`, `all_deltas`, `all_no_output`: end-to-end output
  variants (plain `.cpu()` copies; sparse deltas as `spliceai2-ism` writes
  them; no output at all, so end-to-end is only loading and running). The
  difference between `all_no_output` and the others is the cost of
  producing outputs. The benchmark never writes files; the cost of writing
  Parquet in `spliceai2-ism` is not included.
- `naive`: `all` with the naive full passes timed in the same process,
  for the speedup over plain SpliceAI2.

`trim_windows` only trims SNVs near the ends of the sequence, or on short
sequences, and a batch takes the widest window any of its SNVs needs. With
random positions, every batch at 10,000-50,000 positions is trimmed (the
whole sequence is shorter than the last blocks' windows), but at 100,000
nearly every batch keeps the full width; in `spliceai2-ism`, where a batch
holds neighbouring SNVs, SNVs near a gene's ends are trimmed too
(`--layout consecutive` approaches that). Each width class compiles its own
variant graphs; their compile time is included.

`--configs a,b,...` runs a subset. `ablation.md` and `ablation.json` (in
`results/ablation-<timestamp>/` or `--out-dir`; each length's runs in a
subdirectory such as `10kb/`) list each rewrite's gain per length, side by
side, and, per run, GPU throughput, ms per variant (2D part), reference pass, end-to-end
throughput and phases, peak memory, compile time, and accuracy against the
naive full passes and against the unoptimized incremental path.
`benchmarks/h200_ablation.sbatch` runs all of it on one H200.

## spliceai2-suite: everything in one command

By default every benchmark makes 500 predictions of 100,000 positions:

| Benchmark | Command used | Workload |
|---|---|---|
| fixed | `spliceai2-bench` | 500 sequences × 100k; once per `--fixed-batch-sizes` entry (default 1 and 8) |
| variable | `spliceai2-bench` | 500 sequences, lengths ~ N(100k, 10k), bucketed |
| random substitutions | `spliceai2-mutbench --layout random` | reference + 499 substitutions at random positions |
| consecutive substitutions | `spliceai2-mutbench --layout consecutive` | reference + 499 substitutions at adjacent positions |

The full passes of the substitution benchmarks use `--full-batch-size auto`.

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
