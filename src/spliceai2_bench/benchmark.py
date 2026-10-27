"""Runtime benchmark for SpliceAI2 inference on long sequences.

Times the two-model ensemble under FP16 autocast and torch.compile, matching
the reference scorer (Lightning ``precision="16-mixed"`` + ``torch.compile``).
Each sequence runs as a single forward pass, as in the reference: input is
length + crop_len nt and predictions cover the central ``length`` positions.

Two workloads run in the same process:
  * fixed:    n sequences of exactly --seq-len output positions;
  * variable: n sequences with output lengths ~ N(--var-mean, --var-std),
              batch size 1. With --var-mode bucket (default) each sequence is
              padded up to a multiple of --bucket-size output positions and
              every bucket gets its own static graph; with --var-mode dynamic
              one graph with a dynamic length dimension serves all lengths.

Repetitions are interleaved so thermal drift cannot bias one workload: each
pass alternates batches of both workloads, the batch order reverses every
round (ABBA), and GPU-timed and end-to-end passes swap order every round.
"""
import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

from spliceai2_bench.encode import (
    one_hot_encode,
    random_sequences,
    random_sequences_of_lengths,
)
from spliceai2_bench.predict import Ensemble
from spliceai2_bench.weights import checkpoint_paths, download_models, load_model
from spliceai2_bench.workloads import bucket_lengths, interleave, normal_lengths

WORKLOADS = ("fixed", "variable")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SpliceAI2 inference runtime benchmark (FP16 + torch.compile)"
    )
    p.add_argument("--n-seqs", type=int, default=100,
                   help="sequences per workload")
    p.add_argument(
        "--seq-len", type=int, default=100_000,
        help="output positions per sequence (fixed workload); flanking "
             "context is added on top",
    )
    p.add_argument("--var-mean", type=float,
                   help="mean output length, variable workload "
                        "(default: --seq-len)")
    p.add_argument("--var-std", type=float, default=10_000,
                   help="std of output length, variable workload")
    p.add_argument("--var-mode", choices=["bucket", "dynamic"], default="bucket",
                   help="bucket: pad to --bucket-size multiples, one static "
                        "graph per bucket; dynamic: one dynamic-length graph")
    p.add_argument("--bucket-size", type=int, default=8192,
                   help="bucket granularity in output positions")
    p.add_argument("--batch-size", type=int, default=1,
                   help="fixed workload only; the variable one uses 1")
    p.add_argument(
        "--out-2d-len", type=int, default=1024,
        help="candidate sites per sequence for the junction head "
             "(reference scorer default)",
    )
    p.add_argument("--heads", default="1d,2d", choices=["1d", "1d,2d"])
    p.add_argument(
        "--model-dir", type=Path,
        help="folder containing the two model folders; downloaded from "
             "Hugging Face if omitted",
    )
    p.add_argument("--no-compile", action="store_true")
    p.add_argument(
        "--cudnn-benchmark", action="store_true",
        help="enable cuDNN autotuning (left off by the reference)",
    )
    p.add_argument("--repeats", type=int, default=4,
                   help="rounds; an even number balances the ABBA order")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=Path)
    return p.parse_args(argv)


class Runner:
    """Runs the 1D stage and (optionally) the 2D stage on one batch."""

    def __init__(self, stage_1d, stage_2d, do_2d, compile_, dynamic=None,
                 mark_length_dynamic=False):
        self.do_2d = do_2d
        self.mark_dynamic = compile_ and mark_length_dynamic
        if compile_:
            stage_1d = torch.compile(stage_1d, dynamic=dynamic)
            stage_2d = torch.compile(stage_2d, dynamic=dynamic)
        self.stage_1d = stage_1d
        self.stage_2d = stage_2d

    def __call__(self, x, between=None, valid_len=None):
        """Returns (out_ss, idxs, out_jxn); ``between`` runs after stage 1D."""
        if self.mark_dynamic:
            torch._dynamo.maybe_mark_dynamic(x, 2)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            if not self.do_2d:
                out_ss = self.stage_1d(x)
                if between is not None:
                    between()
                return out_ss, None, None
            ftrs, out_ss = self.stage_1d(x)
            if between is not None:
                between()
            if self.mark_dynamic:
                # ftrs holds one list of per-block features per model
                for t in (*(f for model_ftrs in ftrs for f in model_ftrs),
                          out_ss):
                    torch._dynamo.maybe_mark_dynamic(t, 2)
            idxs, out_jxn = self.stage_2d(ftrs, out_ss, valid_len)
            return out_ss, idxs, out_jxn


def make_runners(ensemble, do_2d, compile_, var_mode):
    """Runners for the fixed- and variable-length workloads.

    torch.compile caches graphs per code object, so each runner gets its own
    lambdas. Otherwise graphs compiled for the variable workload would also
    serve, and change, the fixed-length measurements.
    """
    stage_1d = ensemble.trunk_1d if do_2d else ensemble.predict_1d
    fixed = Runner(
        lambda x: stage_1d(x),
        lambda ftrs, out_ss, valid_len: ensemble.head_2d(ftrs, out_ss, valid_len),
        do_2d, compile_,
    )
    variable = Runner(
        lambda x: stage_1d(x),
        lambda ftrs, out_ss, valid_len: ensemble.head_2d(ftrs, out_ss, valid_len),
        do_2d, compile_,
        # bucket: a static graph per bucket, never generalised to dynamic
        dynamic=False if var_mode == "bucket" else None,
        mark_length_dynamic=var_mode == "dynamic",
    )
    return fixed, variable


def encode_batch(seqs, in_len):
    return torch.from_numpy(
        np.stack([one_hot_encode(s, length=in_len) for s in seqs])
    )


@dataclass
class Workload:
    name: str
    seqs: list
    lengths: list  # real output positions per sequence
    in_lens: list  # model input length per sequence (padded when bucketed)
    batch_size: int
    runner: Runner
    masked: bool = False  # pass real lengths so padding is ignored in 2D
    batches: list = field(init=False)  # pinned CPU one-hot batches
    valid_lens: list = field(init=False)
    peak_mem: int = 0

    def __post_init__(self):
        n = -(-len(self.seqs) // self.batch_size)
        self.batches = [self.encode(i).pin_memory() for i in range(n)]
        self.valid_lens = [
            torch.tensor(self._slice(self.lengths, i)) if self.masked else None
            for i in range(n)
        ]

    def _slice(self, values, i):
        return values[i * self.batch_size:(i + 1) * self.batch_size]

    def encode(self, i):
        return encode_batch(self._slice(self.seqs, i),
                            max(self._slice(self.in_lens, i)))

    def out_len(self, i):
        return max(self._slice(self.lengths, i))

    def valid(self, i, device):
        v = self.valid_lens[i]
        return None if v is None else v.to(device, non_blocking=True)

    def first_batch_per_shape(self):
        first = {}
        for i, x in enumerate(self.batches):
            first.setdefault(tuple(x.shape), i)
        return dict(sorted(first.items()))


def warmup(w, i, device):
    """Wall-clock seconds of the first call (compile + cuDNN setup) per stage."""
    x, v = w.batches[i].to(device), w.valid(i, device)
    marks = []

    def between():
        torch.cuda.synchronize()
        marks.append(time.perf_counter())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    w.runner(x, between=between, valid_len=v)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    w.runner(x, valid_len=v)  # settle allocator / autotuning
    torch.cuda.synchronize()
    return {"1d": marks[0] - t0,
            "2d": t1 - marks[0] if w.runner.do_2d else None}


current_batch = [None]  # batch being warmed, for the out-of-memory message


def warm_all(w, device):
    """Runs every batch once, untimed: triggers any remaining recompiles and
    cuDNN setup before timing starts."""
    for i, x in enumerate(w.batches):
        current_batch[0] = i
        w.runner(x.to(device), valid_len=w.valid(i, device))
        torch.cuda.synchronize()


def gpu_pass(workloads, schedule, device):
    """GPU seconds per batch, {name: [(t_1d, t_2d), ...]} in batch order.

    Inputs are copied to the GPU before each batch's start event, so the
    copy is not timed.
    """
    events = []
    for k, i in schedule:
        w = workloads[k]
        x, v = w.batches[i].to(device, non_blocking=True), w.valid(i, device)
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        torch.cuda.reset_peak_memory_stats()
        e0.record()
        w.runner(x, between=e1.record, valid_len=v)
        e2.record()
        w.peak_mem = max(w.peak_mem, torch.cuda.max_memory_allocated())
        events.append((k, i, e0, e1, e2))
        del x, v
    torch.cuda.synchronize()
    times = {w.name: [None] * len(w.batches) for w in workloads}
    for k, i, e0, e1, e2 in events:
        times[workloads[k].name][i] = (
            e0.elapsed_time(e1) / 1e3, e1.elapsed_time(e2) / 1e3
        )
    return times


def e2e_pass(workloads, schedule, device):
    """Wall-clock seconds per batch, strings to CPU probabilities."""
    times = {w.name: [None] * len(w.batches) for w in workloads}
    for k, i in schedule:
        w = workloads[k]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = w.encode(i).pin_memory().to(device, non_blocking=True)
        out_ss, _, out_jxn = w.runner(x, valid_len=w.valid(i, device))
        out_ss[..., :w.out_len(i)].sigmoid().float().cpu()  # drop padding
        if out_jxn is not None:
            out_jxn.sigmoid().float().cpu()
        times[w.name][i] = time.perf_counter() - t0
    return times


def gpu_telemetry():
    """Temperature (°C), SM clock (MHz) and power (W); None if unavailable."""
    out = {}
    for key, fn, scale in (
        ("temp_c", torch.cuda.temperature, 1),
        ("sm_clock_mhz", torch.cuda.clock_rate, 1),
        ("power_w", torch.cuda.power_draw, 1e-3),
    ):
        try:
            out[key] = fn() * scale
        except Exception:
            out[key] = None
    return out


def summarize(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "all": values,
    }


def summarize_workload(w, gpu_rounds, e2e_rounds, crop_len, do_2d):
    gpu_1d = [sum(t[0] for t in rnd) for rnd in gpu_rounds]
    gpu_2d = [sum(t[1] for t in rnd) for rnd in gpu_rounds]
    real_in = [n + crop_len for n in w.lengths]
    out = {
        "n_seqs": len(w.seqs),
        "batch_size": w.batch_size,
        "lengths": w.lengths,
        "in_lens": w.in_lens,
        "positions": sum(w.lengths),
        "padding_overhead": sum(w.in_lens) / sum(real_in) - 1,
        "gpu_s": {
            "1d": summarize(gpu_1d),
            "2d": summarize(gpu_2d) if do_2d else None,
            "total": summarize([a + b for a, b in zip(gpu_1d, gpu_2d)]),
        },
        "end_to_end_s": summarize([sum(rnd) for rnd in e2e_rounds]),
        "peak_gpu_mem_bytes": w.peak_mem,
    }
    if w.batch_size == 1 and len(set(w.lengths)) > 1:
        # median over rounds per sequence, then GPU time vs real input length
        per_seq = np.median(
            [[a + b for a, b in rnd] for rnd in gpu_rounds], axis=0
        )
        x = np.asarray(real_in)
        slope, intercept = np.polyfit(x, per_seq, 1)
        resid = per_seq - (slope * x + intercept)
        r2 = 1 - (resid ** 2).sum() / ((per_seq - per_seq.mean()) ** 2).sum()
        out["per_seq_gpu_s"] = per_seq.tolist()
        out["fit"] = {
            "intercept_s": float(intercept),
            "slope_s_per_10k_nt": float(slope * 1e4),
            "r2": float(r2),
        }
        if w.masked:
            out["buckets"] = [
                {
                    "in_len": b,
                    "out_len": b - crop_len,
                    "n_seqs": int(sum(1 for n in w.in_lens if n == b)),
                    "median_gpu_s_per_seq": float(np.median(
                        [t for t, n in zip(per_seq, w.in_lens) if n == b]
                    )),
                }
                for b in sorted(set(w.in_lens))
            ]
    return out


def environment_info():
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def fmt_telemetry(before, after):
    def pair(key, unit, spec):
        a, b = before[key], after[key]
        if a is None or b is None:
            return None
        return f"{a:{spec}}→{b:{spec}} {unit}"

    parts = [pair("temp_c", "°C", ".0f"), pair("sm_clock_mhz", "MHz", ".0f"),
             pair("power_w", "W", ".0f")]
    return ", ".join(p for p in parts if p) or "telemetry unavailable"


def raise_recompile_limit(n_graphs):
    cfg = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), n_graphs + 2))
            return


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("a CUDA GPU is required")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    do_2d = args.heads == "1d,2d"
    compile_ = not args.no_compile
    bucketed = args.var_mode == "bucket"

    t0 = time.perf_counter()
    model_dir = args.model_dir or download_models()
    models = [load_model(p) for p in checkpoint_paths(model_dir)]
    ensemble = Ensemble(models, args.out_2d_len).eval().to(device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    m = models[0]
    if (args.seq_len + m.crop_len) % m.max_pool:
        sys.exit(f"seq_len + crop_len = {args.seq_len + m.crop_len} must be "
                 f"divisible by {m.max_pool}")
    if bucketed and (args.bucket_size + m.crop_len) % m.max_pool:
        sys.exit(f"--bucket-size + crop_len must be divisible by {m.max_pool}")
    minimum = args.out_2d_len if do_2d else 1
    if args.seq_len < minimum:
        sys.exit("--out-2d-len must not exceed --seq-len")
    var_mean = args.seq_len if args.var_mean is None else args.var_mean
    var_lengths = normal_lengths(
        args.n_seqs, var_mean, args.var_std, m.crop_len, m.max_pool,
        minimum, seed=args.seed + 1,
    )
    var_out_lens = (bucket_lengths(var_lengths, args.bucket_size)
                    if bucketed else var_lengths)

    fixed_runner, var_runner = make_runners(
        ensemble, do_2d, compile_, args.var_mode
    )
    print("Generating and encoding sequences...", flush=True)
    workloads = [
        Workload(
            "fixed",
            random_sequences(args.n_seqs, args.seq_len + m.crop_len, args.seed),
            [args.seq_len] * args.n_seqs,
            [args.seq_len + m.crop_len] * args.n_seqs,
            args.batch_size, fixed_runner,
        ),
        Workload(
            "variable",
            random_sequences_of_lengths(
                [n + m.crop_len for n in var_lengths], args.seed + 1
            ),
            var_lengths,
            [n + m.crop_len for n in var_out_lens],
            1, var_runner, masked=bucketed,
        ),
    ]
    fixed, variable = workloads

    if compile_:
        if bucketed:
            raise_recompile_limit(len(set(variable.in_lens)))
        # never fall back to eager silently if a workload keeps recompiling
        for flag in ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit"):
            if hasattr(torch._dynamo.config, flag):
                setattr(torch._dynamo.config, flag, True)

    print("Compiling / warming up...", flush=True)
    warmup_s = {"fixed": {}, "variable": {}}
    for shape, i in fixed.first_batch_per_shape().items():
        warmup_s["fixed"][f"batch {shape[0]}"] = warmup(fixed, i, device)
    try:
        if bucketed:
            for shape, i in variable.first_batch_per_shape().items():
                current_batch[0] = i
                label = f"bucket {shape[2] - m.crop_len:,} positions"
                warmup_s["variable"][label] = warmup(variable, i, device)
        else:
            current_batch[0] = 0
            warmup_s["variable"]["dynamic length"] = warmup(variable, 0, device)
        warm_all(variable, device)
    except torch.cuda.OutOfMemoryError:
        i = current_batch[0]
        alloc_hint = (
            "" if "expandable_segments:True"
            in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            else "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, "
        )
        sys.exit(
            f"variable workload: out of GPU memory at "
            f"{variable.lengths[i]:,} output positions "
            f"({variable.in_lens[i]:,} nt model input). With 2D heads, "
            f"per-block features are kept for the whole input; try --heads "
            f"1d, {alloc_hint}or a smaller --var-std."
        )
    if compile_ and hasattr(torch._dynamo.config, "error_on_recompile"):
        torch._dynamo.config.error_on_recompile = True

    schedule = interleave([len(w.batches) for w in workloads])
    gpu_rounds = {w.name: [] for w in workloads}
    e2e_rounds = {w.name: [] for w in workloads}
    passes = []
    for r in range(args.repeats):
        order = "forward" if r % 2 == 0 else "reverse"
        sched = schedule if r % 2 == 0 else schedule[::-1]
        modes = ("gpu", "e2e") if r % 2 == 0 else ("e2e", "gpu")
        for mode in modes:
            before = gpu_telemetry()
            t_start = time.perf_counter()
            if mode == "gpu":
                times = gpu_pass(workloads, sched, device)
                totals = {n: sum(a + b for a, b in t) for n, t in times.items()}
                rounds = gpu_rounds
            else:
                times = e2e_pass(workloads, sched, device)
                totals = {n: sum(t) for n, t in times.items()}
                rounds = e2e_rounds
            elapsed = time.perf_counter() - t_start
            after = gpu_telemetry()
            for name, t in times.items():
                rounds[name].append(t)
            passes.append({
                "round": r + 1, "mode": mode, "order": order,
                "elapsed_s": elapsed, "totals_s": totals,
                "telemetry_start": before, "telemetry_end": after,
            })
            print(
                f"  round {r + 1}/{args.repeats} {mode:<3}  "
                f"fixed {totals['fixed']:7.2f} s  "
                f"variable {totals['variable']:7.2f} s  | "
                f"{fmt_telemetry(before, after)}",
                flush=True,
            )

    result = {
        "env": environment_info(),
        "config": {
            "n_seqs": args.n_seqs,
            "seq_len": args.seq_len,
            "var_mean": var_mean,
            "var_std": args.var_std,
            "var_mode": args.var_mode,
            "bucket_size": args.bucket_size if bucketed else None,
            "batch_size": args.batch_size,
            "heads": args.heads,
            "out_2d_len": args.out_2d_len if do_2d else None,
            "compile": compile_,
            "autocast": "float16",
            "cudnn_benchmark": args.cudnn_benchmark,
            "repeats": args.repeats,
            "seed": args.seed,
            "num_models": len(models),
            "num_blocks": m.num_blocks,
            "num_features": m.num_features,
            "crop_len": m.crop_len,
        },
        "load_s": load_s,
        "warmup_s": warmup_s,
        "workloads": {
            w.name: summarize_workload(
                w, gpu_rounds[w.name], e2e_rounds[w.name], m.crop_len, do_2d
            )
            for w in workloads
        },
        "passes": passes,
    }
    print()
    print_report(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json_out}")


def print_report(r):
    c, env = r["config"], r["env"]

    def fmt(s):
        return f"{s['median']:9.3f} s  [{s['min']:.3f}–{s['max']:.3f}]"

    def rate(s, w):
        sec = s["median"]
        return (f"{sec / w['n_seqs']:.4f} s/seq, "
                f"{w['positions'] / sec / 1e6:.3f} M positions/s")

    def warm(wu):
        line = f"1D {wu['1d']:7.3f} s"
        return line + (f", 2D {wu['2d']:7.3f} s" if wu["2d"] is not None else "")

    print("SpliceAI2 runtime benchmark")
    print(f"  GPU                {env['gpu']}")
    print(f"  torch/CUDA/cuDNN   {env['torch']} / {env['cuda']} / {env['cudnn']}")
    print(f"  models             {c['num_models']} × {c['num_blocks']} blocks, "
          f"{c['num_features']} features, crop_len {c['crop_len']:,}")
    print(f"  mode               fp16 autocast, "
          f"torch.compile {'on' if c['compile'] else 'off'}, "
          f"cudnn.benchmark {'on' if c['cudnn_benchmark'] else 'off'}, "
          f"heads {c['heads']}"
          + (f", out_2d_len {c['out_2d_len']}" if c["out_2d_len"] else ""))
    print(f"  timing             {c['repeats']} interleaved rounds (ABBA), "
          f"median [min–max]")
    print()
    print(f"Model load                       {r['load_s']:9.3f} s")
    print("First call (compile + cuDNN setup)")
    for name in WORKLOADS:
        for label, wu in r["warmup_s"][name].items():
            print(f"  {name + ', ' + label:<34} {warm(wu)}")

    for name in WORKLOADS:
        w = r["workloads"][name]
        lens = w["lengths"]
        print()
        if name == "fixed":
            print(f"Fixed workload: {w['n_seqs']} seqs × {c['seq_len']:,} "
                  f"positions ({c['seq_len'] + c['crop_len']:,} nt input), "
                  f"batch {w['batch_size']}")
        else:
            print(f"Variable workload: {w['n_seqs']} seqs, lengths ~ "
                  f"N({c['var_mean']:,.0f}, {c['var_std']:,.0f}): "
                  f"min {min(lens):,}, median {int(statistics.median(lens)):,}, "
                  f"max {max(lens):,}, total {w['positions']:,} positions, "
                  f"batch 1")
            if c["var_mode"] == "bucket":
                n_graphs = len(set(w["in_lens"]))
                print(f"  bucketed to multiples of {c['bucket_size']:,} "
                      f"positions ({n_graphs} static "
                      f"graph{'s' if n_graphs != 1 else ''}); "
                      f"padding adds {100 * w['padding_overhead']:.2f}% "
                      f"input nt")
            else:
                print("  one dynamic-length graph, no padding")
        g = w["gpu_s"]
        print("  Steady-state GPU time, all sequences")
        print(f"    1D  trunk + splice-site heads  {fmt(g['1d'])}")
        if g["2d"] is not None:
            print(f"    2D  top-k + junction heads     {fmt(g['2d'])}")
        print(f"    total                          {fmt(g['total'])}")
        print(f"                                   {rate(g['total'], w)}")
        print("  End-to-end wall clock (encode + H2D + GPU + D2H)")
        print(f"    total                          {fmt(w['end_to_end_s'])}")
        print(f"                                   {rate(w['end_to_end_s'], w)}")
        if "fit" in w:
            f = w["fit"]
            print(f"  Per-sequence GPU time ≈ {f['intercept_s']:.4f} s + "
                  f"{f['slope_s_per_10k_nt']:.4f} s per 10k nt of real input "
                  f"(R² = {f['r2']:.3f})")
        if "buckets" in w:
            print("  Per bucket (median GPU time per sequence)")
            for b in w["buckets"]:
                print(f"    {b['out_len']:>9,} positions  {b['n_seqs']:3d} seqs"
                      f"  {b['median_gpu_s_per_seq']:.4f} s")
        print(f"  Peak GPU memory                  "
              f"{w['peak_gpu_mem_bytes'] / 2**30:9.2f} GiB")

    print()
    print("Pass log")
    for p in r["passes"]:
        t = p["totals_s"]
        print(f"  round {p['round']} {p['mode']:<3} {p['order']:<7}  "
              f"fixed {t['fixed']:7.2f} s  variable {t['variable']:7.2f} s  | "
              f"{fmt_telemetry(p['telemetry_start'], p['telemetry_end'])}")


if __name__ == "__main__":
    main()
