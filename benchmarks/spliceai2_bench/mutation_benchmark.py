"""Benchmark: one reference sequence plus single-base substitutions.

By default 100 predictions: the reference and 99 random substitutions in its
output region, each scored with the 1D and 2D heads under FP16 autocast and
torch.compile. They are produced two ways in the same process:
  * full:        a complete forward pass per sequence, batch 1;
  * incremental: one pass of the reference that keeps per-block activations,
                 then batches of variants that recompute only what each
                 substitution can change (see spliceai2_ism.incremental).
Both use IncrementalEnsemble's code, so the comparison isolates the
recomputation saved. Before timing, each variant's incremental prediction is
compared with its full pass, and both paths are checked against the original
model on a short sequence.

Measurements are interleaved as in spliceai2-bench: each pass alternates the
two workloads' items, the order reverses every round (ABBA), and GPU-timed
and end-to-end passes swap order every round. In reversed passes the
incremental variants run before the reference pass; they use the cache of
the previous pass, which holds identical values.
"""
import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

from spliceai2_bench.benchmark import fmt_telemetry, gpu_telemetry, summarize
from spliceai2_bench.workloads import interleave
from spliceai2_ism.encode import one_hot_encode, random_sequences
from spliceai2_ism.incremental import IncrementalEnsemble
from spliceai2_ism.predict import predict_full
from spliceai2_ism.runtime import environment_info, forbid_eager_fallback
from spliceai2_ism.weights import checkpoint_paths, download_models, load_model

BASES = "ACGT"
WORKLOADS = ("full", "incremental")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SpliceAI2 substitution benchmark: incremental vs full "
                    "recomputation (FP16 + torch.compile)"
    )
    p.add_argument(
        "--seq-len", type=int, default=100_000,
        help="output positions of the reference; flanking context is added "
             "on top",
    )
    p.add_argument("--n-mutations", type=int, default=99,
                   help="substitutions scored besides the reference")
    p.add_argument("--layout", choices=["random", "consecutive"],
                   default="random",
                   help="random positions, or adjacent positions starting at "
                        "a random offset")
    p.add_argument("--variant-batch-size", type=int, default=9,
                   help="substitutions per incremental batch")
    p.add_argument("--out-2d-len", type=int, default=1024,
                   help="candidate sites per prediction for the junction head")
    p.add_argument(
        "--check-len", type=int, default=8192,
        help="output positions of the short sequence used to check both "
             "paths against the original model",
    )
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


def random_substitutions(seq, crop_len, seq_len, n, seed, layout="random"):
    """n distinct output positions, each with one of the three other bases.

    layout "random": positions drawn uniformly; "consecutive": n adjacent
    positions starting at a random offset.
    """
    rng = np.random.default_rng(seed)
    if layout == "random":
        pos = rng.choice(seq_len, n, replace=False)
    else:
        pos = rng.integers(0, seq_len - n + 1) + np.arange(n)
    ref = np.array([BASES.index(seq[crop_len // 2 + p]) for p in pos])
    alt = (ref + rng.integers(1, 4, n)) % 4
    return pos.tolist(), alt.tolist()


def apply_substitution(seq, crop_len, pos, alt):
    i = crop_len // 2 + pos
    return seq[:i] + BASES[alt] + seq[i + 1:]


def encode(seq):
    return torch.from_numpy(one_hot_encode(seq))[None]


@contextlib.contextmanager
def inference():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        yield


class Stages:
    """Compiled stages. Full and incremental reference passes share theirs:
    same shapes, so the same kernels."""

    def __init__(self, inc, compile_):
        self.ref_1d = lambda x: inc.reference_1d(x)
        self.ref_2d = lambda z, out_ss: inc.head_2d(z, out_ss)
        self.var_1d = lambda cache, ref_ss, pos, alt: inc.variants_track(
            cache, ref_ss, pos, alt
        )
        self.var_2d = lambda z, out_ss, pos, z_var: inc.head_2d(
            z, out_ss, pos, z_var
        )
        if compile_:
            for name in ("ref_1d", "ref_2d", "var_1d", "var_2d"):
                setattr(self, name,
                        torch.compile(getattr(self, name), dynamic=False))


class FullWorkload:
    """A complete forward pass per sequence: the reference, then each
    variant sequence."""

    name = "full"

    def __init__(self, seqs, stages, device):
        self.seqs, self.stages, self.device = seqs, stages, device
        self.inputs = [encode(s).pin_memory() for s in seqs]
        self.peak_mem = 0

    def __len__(self):
        return len(self.seqs)

    def predictions(self, i):
        return 1

    def load(self, i, from_string=False):
        x = encode(self.seqs[i]).pin_memory() if from_string else self.inputs[i]
        return x.to(self.device, non_blocking=True)

    def run(self, x, between=None):
        """Returns (out_ss, idxs, out_jxn); ``between`` runs after stage 1D."""
        with inference():
            cache, out_ss = self.stages.ref_1d(x)
            if between is not None:
                between()
            idxs, out_jxn = self.stages.ref_2d(cache[1], out_ss)
        return out_ss, idxs, out_jxn


class IncrementalWorkload:
    """Item 0: reference pass that (re)builds the cache. Items 1..: variant
    batches; the last one is padded by repeating its final substitution so
    every batch has the same shape."""

    name = "incremental"

    def __init__(self, ref_seq, positions, alts, batch_size, stages, device):
        self.ref_seq, self.stages, self.device = ref_seq, stages, device
        self.ref_input = encode(ref_seq).pin_memory()
        self.batches = []
        for s in range(0, len(positions), batch_size):
            pos, alt = positions[s:s + batch_size], alts[s:s + batch_size]
            pad = batch_size - len(pos)
            self.batches.append(
                (pos + pos[-1:] * pad, alt + alt[-1:] * pad, len(pos))
            )
        self.pinned = [tuple(torch.tensor(v).pin_memory() for v in b[:2])
                       for b in self.batches]
        self.cache = self.ref_ss = None
        self.peak_mem = 0

    def __len__(self):
        return 1 + len(self.batches)

    def predictions(self, i):
        return 1 if i == 0 else self.batches[i - 1][2]

    def load(self, i, from_string=False):
        if i == 0:
            x = encode(self.ref_seq).pin_memory() if from_string else self.ref_input
            return x.to(self.device, non_blocking=True)
        if from_string:
            pos, alt = (torch.tensor(v).pin_memory()
                        for v in self.batches[i - 1][:2])
        else:
            pos, alt = self.pinned[i - 1]
        return (pos.to(self.device, non_blocking=True),
                alt.to(self.device, non_blocking=True))

    def run(self, inp, between=None):
        with inference():
            if isinstance(inp, torch.Tensor):
                # free the old cache before building anew
                self.cache = self.ref_ss = None
                self.cache, self.ref_ss = self.stages.ref_1d(inp)
                if between is not None:
                    between()
                out_ss = self.ref_ss
                idxs, out_jxn = self.stages.ref_2d(self.cache[1], out_ss)
            else:
                pos, alt = inp
                z_var, out_ss = self.stages.var_1d(self.cache, self.ref_ss,
                                                   pos, alt)
                if between is not None:
                    between()
                idxs, out_jxn = self.stages.var_2d(
                    self.cache[1], out_ss, pos, z_var
                )
        return out_ss, idxs, out_jxn


def to_host(w, i, out):
    """CPU probabilities of the real (unpadded) predictions of item i."""
    n = w.predictions(i)
    out_ss, idxs, out_jxn = (t[:n] for t in out)
    return (out_ss.sigmoid().float().cpu(), idxs.cpu(),
            out_jxn.sigmoid().float().cpu())


def warmup(w, i):
    """Wall-clock seconds of the first call (compile + cuDNN setup) per stage."""
    inp = w.load(i)
    marks = []

    def between():
        torch.cuda.synchronize()
        marks.append(time.perf_counter())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    w.run(inp, between)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    w.run(w.load(i))  # settle allocator / autotuning
    torch.cuda.synchronize()
    return {"1d": marks[0] - t0, "2d": t1 - marks[0]}


def gpu_pass(workloads, schedule):
    """GPU seconds per item, {name: [(t_1d, t_2d), ...]} in item order."""
    events = []
    for k, i in schedule:
        w = workloads[k]
        inp = w.load(i)
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        torch.cuda.reset_peak_memory_stats()
        e0.record()
        w.run(inp, between=e1.record)
        e2.record()
        w.peak_mem = max(w.peak_mem, torch.cuda.max_memory_allocated())
        events.append((k, i, e0, e1, e2))
        del inp
    torch.cuda.synchronize()
    times = {w.name: [None] * len(w) for w in workloads}
    for k, i, e0, e1, e2 in events:
        times[workloads[k].name][i] = (
            e0.elapsed_time(e1) / 1e3, e1.elapsed_time(e2) / 1e3
        )
    return times


def e2e_pass(workloads, schedule):
    """Wall-clock seconds per item: strings (or substitution lists) to CPU
    probabilities."""
    times = {w.name: [None] * len(w) for w in workloads}
    for k, i in schedule:
        w = workloads[k]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        to_host(w, i, w.run(w.load(i, from_string=True)))
        times[w.name][i] = time.perf_counter() - t0
    return times


def compare(ours, theirs):
    """Max |Δ probability| (1D; 2D where both picked the same top-k sites)
    between two lists of (out_ss, idxs, out_jxn), one prediction each."""
    d1, d2, same, overlap = 0.0, None, 0, []
    for (ss, idx, jxn), (ss_t, idx_t, jxn_t) in zip(ours, theirs):
        d1 = max(d1, (ss.float().sigmoid() - ss_t.float().sigmoid())
                 .abs().max().item())
        overlap.append(torch.isin(idx, idx_t).float().mean().item())
        if torch.equal(idx, idx_t):
            same += 1
            d = (jxn.float().sigmoid() - jxn_t.float().sigmoid()).abs().max()
            d2 = max(d2 or 0.0, d.item())
    return {
        "n": len(ours),
        "max_dp_1d": d1,
        "topk_identical": same,
        "mean_topk_overlap": statistics.mean(overlap),
        "max_dp_2d": d2,
    }


def check_against_original(models, inc, args, device):
    """Eager reference pass and a few variants of a short sequence through
    IncrementalEnsemble, vs full passes of the original model code."""
    crop_len = inc.crop_len
    seq = random_sequences(1, args.check_len + crop_len, args.seed + 2)[0]
    positions, alts = random_substitutions(seq, crop_len, args.check_len, 3,
                                           args.seed + 3)
    pos = torch.tensor(positions, device=device)
    alt = torch.tensor(alts, device=device)
    with inference():
        cache, ref_ss = inc.reference_1d(encode(seq).to(device))
        idxs, jxn = inc.head_2d(cache[1], ref_ss)
        ours = [(ref_ss[0], idxs[0], jxn[0])]
        z_var, var_ss = inc.variants_track(cache, ref_ss, pos, alt)
        idxs, jxn = inc.head_2d(cache[1], var_ss, pos, z_var)
        ours += [(var_ss[j], idxs[j], jxn[j]) for j in range(len(positions))]
        del cache, z_var
        torch.cuda.empty_cache()

        theirs = []
        for s in [seq] + [apply_substitution(seq, crop_len, p, a)
                          for p, a in zip(positions, alts)]:
            ss, idx, jx = predict_full(models, encode(s).to(device),
                                       args.out_2d_len)
            theirs.append((ss[0], idx[0], jx[0]))
    return {"reference": compare(ours[:1], theirs[:1]),
            "variants": compare(ours[1:], theirs[1:])}


def check_against_full(full, incr):
    """Every variant's incremental prediction vs its full pass (untimed)."""
    ours, theirs = [], []
    for i in range(1, len(incr)):
        out = incr.run(incr.load(i))
        start = sum(incr.predictions(j) for j in range(1, i))
        for j in range(incr.predictions(i)):
            ours.append(tuple(t[j] for t in out))
            theirs.append(tuple(t[0] for t in full.run(full.load(1 + start + j))))
    return compare(ours, theirs)


def stage_summary(gpu_rounds, e2e_rounds, items):
    g1 = [sum(rnd[i][0] for i in items) for rnd in gpu_rounds]
    g2 = [sum(rnd[i][1] for i in items) for rnd in gpu_rounds]
    return {
        "gpu_s": {
            "1d": summarize(g1),
            "2d": summarize(g2),
            "total": summarize([a + b for a, b in zip(g1, g2)]),
        },
        "end_to_end_s": summarize([sum(rnd[i] for i in items)
                                   for rnd in e2e_rounds]),
    }


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("a CUDA GPU is required")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    compile_ = not args.no_compile

    t0 = time.perf_counter()
    model_dir = args.model_dir or download_models()
    models = [load_model(p) for p in checkpoint_paths(model_dir)]
    inc = IncrementalEnsemble(models, args.out_2d_len).eval().to(device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0

    m = models[0]
    for flag, n in (("--seq-len", args.seq_len), ("--check-len", args.check_len)):
        if (n + m.crop_len) % m.max_pool:
            sys.exit(f"{flag} + crop_len = {n + m.crop_len} must be divisible "
                     f"by {m.max_pool}")
        if n < args.out_2d_len:
            sys.exit(f"--out-2d-len must not exceed {flag}")
    if not 1 <= args.n_mutations <= args.seq_len:
        sys.exit("--n-mutations must be between 1 and --seq-len")

    print("Checking both paths against the original model "
          f"(eager, {args.check_len:,} positions)...", flush=True)
    original_check = check_against_original(models, inc, args, device)
    torch.cuda.empty_cache()

    print("Generating sequences...", flush=True)
    seq = random_sequences(1, args.seq_len + m.crop_len, args.seed)[0]
    positions, alts = random_substitutions(seq, m.crop_len, args.seq_len,
                                           args.n_mutations, args.seed + 1,
                                           args.layout)
    seqs = [seq] + [apply_substitution(seq, m.crop_len, p, a)
                    for p, a in zip(positions, alts)]
    stages = Stages(inc, compile_)
    full = FullWorkload(seqs, stages, device)
    incr = IncrementalWorkload(seq, positions, alts, args.variant_batch_size,
                               stages, device)
    workloads = [full, incr]

    if compile_:
        forbid_eager_fallback()

    print("Compiling / warming up...", flush=True)
    warmup_s = {"reference pass": warmup(full, 0)}
    warmup(incr, 0)  # same compiled stages; builds the cache
    warmup_s[f"variant batch of {args.variant_batch_size}"] = warmup(incr, 1)
    for w in workloads:
        for i in range(len(w)):
            w.run(w.load(i))
    torch.cuda.synchronize()

    print("Checking incremental predictions against full passes...",
          flush=True)
    full_check = check_against_full(full, incr)
    if compile_ and hasattr(torch._dynamo.config, "error_on_recompile"):
        torch._dynamo.config.error_on_recompile = True

    schedule = interleave([len(w) for w in workloads])
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
                times = gpu_pass(workloads, sched)
                totals = {n: sum(a + b for a, b in t) for n, t in times.items()}
                rounds = gpu_rounds
            else:
                times = e2e_pass(workloads, sched)
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
                f"full {totals['full']:7.2f} s  "
                f"incremental {totals['incremental']:7.2f} s  | "
                f"{fmt_telemetry(before, after)}",
                flush=True,
            )

    n_in = args.seq_len + m.crop_len
    result = {
        "env": environment_info(),
        "config": {
            "seq_len": args.seq_len,
            "n_mutations": args.n_mutations,
            "layout": args.layout,
            "variant_batch_size": args.variant_batch_size,
            "out_2d_len": args.out_2d_len,
            "check_len": args.check_len,
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
        "substitutions": {"positions": positions,
                          "alt": [BASES[a] for a in alts]},
        "geometry": {
            "reach_nt": inc.reach,
            "cache_margin_nt": inc.margin,
            "recompute_fraction": inc.recompute_fraction(n_in),
            "blocks": [vars(p) for p in inc.plans],
        },
        "accuracy": {"original_model": original_check, "full_pass": full_check},
        "load_s": load_s,
        "warmup_s": warmup_s,
        "workloads": {
            "full": {
                **stage_summary(gpu_rounds["full"], e2e_rounds["full"],
                                range(len(full))),
                "peak_gpu_mem_bytes": full.peak_mem,
            },
            "incremental": {
                "reference": stage_summary(gpu_rounds["incremental"],
                                           e2e_rounds["incremental"], [0]),
                "variants": stage_summary(gpu_rounds["incremental"],
                                          e2e_rounds["incremental"],
                                          range(1, len(incr))),
                "total": stage_summary(gpu_rounds["incremental"],
                                       e2e_rounds["incremental"],
                                       range(len(incr))),
                "peak_gpu_mem_bytes": incr.peak_mem,
            },
        },
        "speedup": {
            mode: summarize([p["totals_s"]["full"] / p["totals_s"]["incremental"]
                             for p in passes if p["mode"] == mode])
            for mode in ("gpu", "e2e")
        },
        "passes": passes,
    }
    print()
    print_report(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json_out}")


def print_report(r):
    c, env, g = r["config"], r["env"], r["geometry"]
    n_pred = c["n_mutations"] + 1

    def fmt(s):
        return f"{s['median']:9.3f} s  [{s['min']:.3f}–{s['max']:.3f}]"

    def per(s, n, what):
        return f"{s['median'] / n * 1e3:.2f} ms/{what}"

    def accuracy(a):
        d2 = "n/a" if a["max_dp_2d"] is None else f"{a['max_dp_2d']:.1e}"
        return (f"1D {a['max_dp_1d']:.1e}, 2D {d2}, top-k identical "
                f"{a['topk_identical']}/{a['n']} "
                f"(mean overlap {a['mean_topk_overlap']:.4f})")

    def stages(s, n, what, indent="    "):
        gs = s["gpu_s"]
        print(f"{indent}GPU 1D  trunk + splice-site heads  {fmt(gs['1d'])}")
        print(f"{indent}GPU 2D  top-k + junction heads     {fmt(gs['2d'])}")
        print(f"{indent}GPU total                          {fmt(gs['total'])}"
              f"  {per(gs['total'], n, what)}")
        print(f"{indent}end-to-end                         "
              f"{fmt(s['end_to_end_s'])}  {per(s['end_to_end_s'], n, what)}")

    print("SpliceAI2 substitution benchmark")
    print(f"  GPU                {env['gpu']}")
    print(f"  torch/CUDA/cuDNN   {env['torch']} / {env['cuda']} / {env['cudnn']}")
    print(f"  models             {c['num_models']} × {c['num_blocks']} blocks, "
          f"{c['num_features']} features, crop_len {c['crop_len']:,}")
    print(f"  mode               fp16 autocast, "
          f"torch.compile {'on' if c['compile'] else 'off'}, "
          f"cudnn.benchmark {'on' if c['cudnn_benchmark'] else 'off'}, "
          f"heads 1d,2d, out_2d_len {c['out_2d_len']}")
    print(f"  sequence           {c['seq_len']:,} positions "
          f"({c['seq_len'] + c['crop_len']:,} nt input): reference + "
          f"{c['n_mutations']} {c['layout']} substitutions = {n_pred} "
          f"predictions")
    if c["layout"] == "consecutive":
        ps = r["substitutions"]["positions"]
        print(f"                     output positions {min(ps):,}–{max(ps):,}")
    print(f"  incremental        variant batch {c['variant_batch_size']}; a "
          f"substitution reaches ±{g['reach_nt']:,} nt; "
          f"{100 * g['recompute_fraction']:.2f}% of a full pass's block "
          f"outputs recomputed per variant")
    print(f"  timing             {c['repeats']} interleaved rounds (ABBA), "
          f"median [min–max]")
    print()
    print("Accuracy, max |Δ probability| (2D: predictions with identical top-k)")
    a = r["accuracy"]
    print(f"  original model, {c['check_len']:,} positions, eager")
    print(f"    reference pass   {accuracy(a['original_model']['reference'])}")
    print(f"    variants         {accuracy(a['original_model']['variants'])}")
    print(f"  incremental vs full pass, compiled")
    print(f"    all variants     {accuracy(a['full_pass'])}")
    print()
    print(f"Model load                               {r['load_s']:9.3f} s")
    print("First call (compile + cuDNN setup)")
    for label, wu in r["warmup_s"].items():
        print(f"  {label:<28} 1D {wu['1d']:7.3f} s, 2D {wu['2d']:7.3f} s")

    w = r["workloads"]
    print()
    print(f"Full recompute: {n_pred} forward passes, batch 1")
    stages(w["full"], n_pred, "prediction")
    print(f"  Peak GPU memory                    "
          f"{w['full']['peak_gpu_mem_bytes'] / 2**30:9.2f} GiB")
    print()
    print(f"Incremental: reference pass + {c['n_mutations']} variants")
    print("  reference pass (keeps per-block activations)")
    stages(w["incremental"]["reference"], 1, "pass")
    print("  variants")
    stages(w["incremental"]["variants"], c["n_mutations"], "variant")
    print("  total")
    stages(w["incremental"]["total"], n_pred, "prediction")
    print(f"  Peak GPU memory                    "
          f"{w['incremental']['peak_gpu_mem_bytes'] / 2**30:9.2f} GiB")
    print()
    s = r["speedup"]
    print("Speedup, full / incremental (median [min–max] of per-pass ratios)")
    print(f"  GPU time     {s['gpu']['median']:6.1f}×  "
          f"[{s['gpu']['min']:.1f}–{s['gpu']['max']:.1f}]")
    print(f"  end-to-end   {s['e2e']['median']:6.1f}×  "
          f"[{s['e2e']['min']:.1f}–{s['e2e']['max']:.1f}]")

    print()
    print("Pass log")
    for p in r["passes"]:
        t = p["totals_s"]
        print(f"  round {p['round']} {p['mode']:<3} {p['order']:<7}  "
              f"full {t['full']:7.2f} s  incremental {t['incremental']:7.2f} s"
              f"  | {fmt_telemetry(p['telemetry_start'], p['telemetry_end'])}")


if __name__ == "__main__":
    main()
