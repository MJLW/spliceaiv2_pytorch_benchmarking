"""Benchmark: one reference sequence plus single-base substitutions.

By default 100 predictions: the reference and 99 random substitutions in its
output region, each scored with the 1D and 2D heads under FP16 autocast and
torch.compile. They are produced two ways in the same process:
  * full:        naive, a complete forward pass per sequence (``Ensemble``,
                 the plain full-pass code), in batches of --full-batch-size;
  * incremental: one pass of the reference that keeps per-block activations,
                 then batches of variants that recompute only what each
                 substitution can change (see spliceai2_ism.incremental).
Both batch sizes default to ``auto``: the largest batch whose peak memory,
measured eagerly on two small batches, fits in 80% of the GPU. Before
timing, each variant's incremental prediction is compared with its full
pass, and both paths are checked against the original model on a short
sequence.

Measurements are interleaved as in spliceai2-bench: each pass alternates the
two workloads' items, the order reverses every round (ABBA), and GPU-timed
and end-to-end passes swap order every round. In reversed passes the
incremental variants run before the reference pass; they use the cache of
the previous pass, which holds identical values. --warm-seconds runs both
workloads untimed first, so timing starts at steady clocks and temperature;
--profile-dir writes torch.profiler traces of the warm state afterwards.

Running out of GPU memory exits with code 3 (OOM_EXIT), so spliceai2-sweep
can tell it from other failures.
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
import torch.utils._pytree as pytree

from spliceai2_bench.benchmark import fmt_telemetry, gpu_telemetry, summarize
from spliceai2_bench.profiling import label, profile_workload
from spliceai2_bench.workloads import interleave
from spliceai2_ism.batching import MEMORY_FRACTION, MULTIPLE, fit_batch_size
from spliceai2_ism.deltas import delta_2d, haplotype_rows
from spliceai2_ism.encode import one_hot_encode, random_sequences
from spliceai2_ism.incremental import IncrementalEnsemble, Optimizations
from spliceai2_ism.predict import Ensemble, predict_full
from spliceai2_ism.runtime import (
    environment_info,
    forbid_eager_fallback,
    raise_recompile_limit,
)
from spliceai2_ism.weights import checkpoint_paths, download_models, load_model

BASES = "ACGT"
WORKLOADS = ("full", "incremental")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune-no-cudagraphs")
OOM_EXIT = 3
FULL_PROBE_SIZES = (1, 2)
VARIANT_PROBE_SIZES = (1, 5)


def _opts(spec):
    try:
        return Optimizations.parse(spec)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _batch_size(value):
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError("a positive integer or 'auto'")
    return n


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
    p.add_argument("--full-batch-size", type=_batch_size, default="auto",
                   help="sequences per full forward pass, or 'auto': the "
                        "largest that fits in GPU memory")
    p.add_argument("--variant-batch-size", type=_batch_size, default=32,
                   help="substitutions per incremental batch, or 'auto' "
                        "(throughput no longer improves past about 32)")
    p.add_argument("--only", choices=WORKLOADS,
                   help="time only this workload (the accuracy checks still "
                        "run both)")
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
        "--compile-mode", choices=COMPILE_MODES, default="default",
        help="torch.compile mode; reduce-overhead adds CUDA graphs",
    )
    p.add_argument("--freezing", action="store_true",
                   help="Inductor freezing: weights become constants")
    p.add_argument(
        "--opts", type=_opts, default=Optimizations(),
        help="incremental rewrites (spliceai2_ism.incremental.Optimizations): "
             "'all' (default), 'none', names to enable, or 'all,-name' to "
             f"leave one out; names: {', '.join(Optimizations.names())}. "
             "The naive path never uses them",
    )
    p.add_argument(
        "--cudnn-benchmark", action="store_true",
        help="enable cuDNN autotuning (left off by the reference)",
    )
    p.add_argument(
        "--e2e-output", choices=["full", "deltas", "none"], default="full",
        help="end-to-end output: full probability tracks and junction "
             "matrices, sparse deltas against the reference computed on "
             "the GPU (as spliceai2-ism writes them), or none (outputs stay "
             "on the GPU; the cost of producing outputs is the difference)",
    )
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="with --e2e-output deltas: smallest |Δ probability| "
                        "kept (spliceai2-ism's default)")
    p.add_argument(
        "--host-copy", choices=["pinned", "pageable"], default="pinned",
        help="with --e2e-output full: copy outputs to the CPU into reused "
             "pinned buffers, or into newly allocated pageable memory (plain "
             ".cpu(), as before)",
    )
    p.add_argument("--warm-seconds", type=float, default=0,
                   help="run both workloads untimed for this long before "
                        "timing, to reach steady GPU clocks and temperature")
    p.add_argument("--repeats", type=int, default=4,
                   help="rounds; an even number balances the ABBA order")
    p.add_argument("--profile-dir", type=Path,
                   help="write torch.profiler traces of each workload here, "
                        "after timing")
    p.add_argument("--profile-iters", type=int, default=3,
                   help="profiled iterations per workload")
    p.add_argument("--profile-stacks", action="store_true",
                   help="also record Python stacks and write GPU and CPU "
                        "flamegraphs (SVG) next to the traces")
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


def encode_many(seqs):
    return torch.from_numpy(np.stack([one_hot_encode(s) for s in seqs]))


@contextlib.contextmanager
def inference():
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        yield


class Stages:
    """Compiled stages: batched full passes (``Ensemble``), and the
    incremental reference pass and variant batches."""

    def __init__(self, inc, ens, compile_, mode="default"):
        # CUDA graphs replay into fixed output buffers: callers mark each
        # step, and keep what must outlive the next call in their own memory
        self.cudagraphs = compile_ and mode == "reduce-overhead"
        self.full_1d = lambda x: ens.trunk_1d(x)
        self.full_2d = lambda ftrs, out_ss: ens.head_2d(ftrs, out_ss)
        self.ref_1d = lambda x: inc.reference_1d(x)
        self.ref_2d = lambda z, out_ss: inc.head_2d(z, out_ss)
        # widths: the batch's window width class (trim_windows), a
        # compile-time constant
        self.var_1d = lambda cache, ref_ss, pos, alt, widths=None: \
            inc.variants_track(cache, ref_ss, pos, alt, widths)
        self.var_2d = lambda z, out_ss, pos, z_var, widths=None: inc.head_2d(
            z, out_ss, pos, z_var, widths=widths
        )
        if compile_:
            kw = {"dynamic": False}
            if mode != "default":
                kw["mode"] = mode
            for name in ("full_1d", "full_2d", "ref_1d", "ref_2d", "var_1d",
                         "var_2d"):
                setattr(self, name, torch.compile(getattr(self, name), **kw))

    def step(self):
        """Call before each item: starts a new CUDA-graph step."""
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()


class StaticCopy:
    """Copies a tree of tensors into buffers allocated on first use and
    reused afterwards, marked as static inputs, so CUDA graphs read them in
    place instead of copying them on every replay."""

    def __init__(self):
        self.bufs = None

    def __call__(self, tree):
        flat, spec = pytree.tree_flatten(tree)
        if self.bufs is None:
            self.bufs = [t.clone() for t in flat]
            for t in self.bufs:
                torch._dynamo.mark_static_address(t)
        else:
            for dst, src in zip(self.bufs, flat):
                dst.copy_(src)
        return pytree.tree_unflatten(self.bufs, spec)


def full_batches(n, batch_size):
    """Sequence indices per batch, the last one padded by repeating its final
    sequence so every batch has the same shape, and the real count per
    batch."""
    out = []
    for s in range(0, n, batch_size):
        idx = list(range(s, min(s + batch_size, n)))
        out.append((idx + idx[-1:] * (batch_size - len(idx)), len(idx)))
    return out


class FullWorkload:
    """Complete forward passes of the reference and each variant sequence,
    ``batch_size`` sequences at a time."""

    name = "full"

    def __init__(self, seqs, batch_size, stages, device):
        self.seqs, self.stages, self.device = seqs, stages, device
        self.batches = full_batches(len(seqs), batch_size)
        self.inputs = [self._encode(i).pin_memory()
                       for i in range(len(self.batches))]
        self.peak_mem = 0
        self.host_buffers = {}  # pinned output buffers, see to_host
        self.ref_out = None  # reference prediction, see remember_reference

    def __len__(self):
        return len(self.batches)

    def predictions(self, i):
        return self.batches[i][1]

    def reference_rows(self, i):
        """Rows of item i that hold the reference sequence (sequence 0)."""
        idx, n = self.batches[i]
        return [r for r in range(n) if idx[r] == 0]

    def _encode(self, i):
        return encode_many([self.seqs[j] for j in self.batches[i][0]])

    def load(self, i, from_string=False):
        x = self._encode(i).pin_memory() if from_string else self.inputs[i]
        return x.to(self.device, non_blocking=True)

    def run(self, x, between=None):
        """Returns (out_ss, idxs, out_jxn); ``between`` runs after stage 1D."""
        self.stages.step()
        with inference():
            with label("1d"):
                ftrs, out_ss = self.stages.full_1d(x)
            if between is not None:
                between()
            with label("2d"):
                idxs, out_jxn = self.stages.full_2d(ftrs, out_ss)
        return out_ss, idxs, out_jxn


class IncrementalWorkload:
    """Item 0: reference pass that (re)builds the cache. Items 1..: variant
    batches; the last one is padded by repeating its final substitution so
    every batch has the same shape."""

    name = "incremental"

    def __init__(self, ref_seq, positions, alts, batch_size, stages, device,
                 widths=lambda pos: None):
        """``widths(pos)``: the window width class of a batch (e.g.
        ``inc.window_widths(pos, seq_len)``), chosen once per batch."""
        self.ref_seq, self.stages, self.device = ref_seq, stages, device
        self.ref_input = encode(ref_seq).pin_memory()
        self.batches = []
        for s in range(0, len(positions), batch_size):
            pos, alt = positions[s:s + batch_size], alts[s:s + batch_size]
            pad = batch_size - len(pos)
            self.batches.append(
                (pos + pos[-1:] * pad, alt + alt[-1:] * pad, len(pos))
            )
        self.widths = [widths(b[0]) for b in self.batches]
        self.pinned = [tuple(torch.tensor(v).pin_memory() for v in b[:2])
                       for b in self.batches]
        self.cache = self.ref_ss = None
        self.static = StaticCopy() if stages.cudagraphs else None
        self.peak_mem = 0
        self.host_buffers = {}  # pinned output buffers, see to_host
        self.ref_out = None  # reference prediction, see remember_reference

    def __len__(self):
        return 1 + len(self.batches)

    def predictions(self, i):
        return 1 if i == 0 else self.batches[i - 1][2]

    def reference_rows(self, i):
        return [0] if i == 0 else []

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
                alt.to(self.device, non_blocking=True), self.widths[i - 1])

    def run(self, inp, between=None):
        self.stages.step()
        with inference():
            if isinstance(inp, torch.Tensor):
                if self.static is None:
                    # free the old cache before building anew
                    self.cache = self.ref_ss = None
                with label("ref_1d"):
                    self.cache, self.ref_ss = self.stages.ref_1d(inp)
                    if self.static is not None:
                        # CUDA graphs: the cache must outlive later replays
                        self.cache, self.ref_ss = self.static(
                            (self.cache, self.ref_ss)
                        )
                if between is not None:
                    between()
                out_ss = self.ref_ss
                with label("ref_2d"):
                    idxs, out_jxn = self.stages.ref_2d(self.cache[1], out_ss)
            else:
                pos, alt, widths = inp
                with label("var_1d"):
                    z_var, out_ss = self.stages.var_1d(self.cache, self.ref_ss,
                                                       pos, alt, widths)
                if between is not None:
                    between()
                with label("var_2d"):
                    idxs, out_jxn = self.stages.var_2d(
                        self.cache[1], out_ss, pos, z_var, widths
                    )
        return out_ss, idxs, out_jxn


def to_host(w, i, out, mode="pinned"):
    """CPU probabilities of the real (unpadded) predictions of item i.

    mode "pageable": plain ``.cpu()``, which allocates new host memory
    every call. "pinned": copies into page-locked buffers that are
    allocated once per output shape and reused (``w.host_buffers``), so the
    returned tensors are overwritten by the next call with the same shape.
    """
    n = w.predictions(i)
    out_ss, idxs, out_jxn = (t[:n] for t in out)
    probs = (out_ss.sigmoid().float(), idxs, out_jxn.sigmoid().float())
    if mode == "pageable":
        return tuple(t.cpu() for t in probs)
    # keyed on the padded batch shape, so the tail batch (fewer real
    # predictions) reuses the full-size buffers
    key = tuple((tuple(t.shape), t.dtype) for t in out)
    bufs = w.host_buffers.get(key)
    if bufs is None:
        bufs = w.host_buffers[key] = tuple(
            torch.empty((t.shape[0], *p.shape[1:]), dtype=p.dtype,
                        pin_memory=True)
            for t, p in zip(out, probs)
        )
    host = tuple(b[:n] for b in bufs)
    for h, p in zip(host, probs):
        h.copy_(p, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return host


def remember_reference(w, i, out):
    """Keeps the reference prediction (logits) if item i holds it, as the
    baseline of ``deltas_to_host``. Cloned: under CUDA graphs ``out`` is
    overwritten by the next replay."""
    for r in w.reference_rows(i):
        out_ss, idxs, out_jxn = out
        w.ref_out = (out_ss[r:r + 1].clone(), idxs[r].clone(),
                     out_jxn[r].clone())


def deltas_to_host(w, i, out, min_delta):
    """Like spliceai2-ism: sparse deltas of item i's predictions against the
    reference, computed on the GPU, and only those copied to the CPU; the
    reference itself as full probabilities (``haplotype_rows``). 1D deltas
    cover the whole output region (spliceai2-ism only looks inside each
    SNV's cone, where every non-zero delta lies)."""
    n = w.predictions(i)
    out_ss, idxs, out_jxn = (t[:n] for t in out)
    ref_ss, ref_idx, ref_jxn = w.ref_out
    host = [tuple(t.cpu() for t in haplotype_rows(
        out_ss[r:r + 1], idxs[r], out_jxn[r], out_ss.shape[2]))
        for r in w.reference_rows(i)]
    d = out_ss.float().sigmoid() - ref_ss.float().sigmoid()
    keep = (d.abs() >= min_delta).any(dim=1)
    row, pos = keep.nonzero(as_tuple=True)
    r1 = (row, pos, d[row, 0, pos], d[row, 1, pos])
    r2 = delta_2d(ref_idx, ref_jxn, idxs, out_jxn, min_delta)
    host.append(tuple(t.cpu() for t in (*r1, *r2)))
    return host


def make_emit(args):
    """The end-to-end output step, ``emit(w, i, out)``."""
    if args.e2e_output == "deltas":
        return lambda w, i, out: deltas_to_host(w, i, out, args.min_delta)
    if args.e2e_output == "none":
        return lambda w, i, out: None
    return lambda w, i, out: to_host(w, i, out, args.host_copy)


def probe_peaks(fn, sizes):
    """Peak allocated bytes of ``fn(b)`` per batch size b, smallest first,
    stopping at the first size that runs out of memory."""
    peaks = {}
    for b in sizes:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        try:
            out = fn(b)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            out = None
        if out is None:  # outside the except block, so its tensors are freed
            torch.cuda.empty_cache()
            break
        peaks[b] = torch.cuda.max_memory_allocated()
        del out
    return peaks


def fit_probe(peaks, baseline, capacity, fraction, multiple):
    """(batch size, fixed bytes, bytes per item) from probe peaks. With one
    peak, the memory held before the probe (``baseline``: models and
    cache) stands in for the second point; with none, the estimate is
    unknown and the size is 1, so the run still tries it (compiled stages
    usually need less memory than the eager probe)."""
    if not peaks:
        return 1, None, None
    if len(peaks) == 1:
        peaks = {0: baseline, **peaks}
    return fit_batch_size(peaks, capacity, fraction, cap=None,
                          multiple=multiple)


def measure_batch_sizes(inc, ens, seq, seq_len, n_full, n_var, device):
    """Batch sizes from eager peak memory of small batches, measured with
    the incremental reference cache resident, as during timing.

    Returns, per path ("full", "variant"), the size ``auto`` picks (largest
    within MEMORY_FRACTION of the memory, a multiple of 8) and the
    ``ceiling`` (largest within all of it), both capped at the number of
    items, plus the measurements.
    """
    x = encode(seq).to(device)
    with inference():
        cache, ref_ss = inc.reference_1d(x)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()

        def full(b):
            ftrs, out_ss = ens.trunk_1d(x.expand(b, -1, -1))
            return ens.head_2d(ftrs, out_ss)

        def variants(b):
            pos = torch.linspace(0, seq_len - 1, b, device=device).long()
            alt = torch.zeros(b, dtype=torch.long, device=device)
            z_var, var_ss = inc.variants_track(cache, ref_ss, pos, alt)
            return inc.head_2d(cache[1], var_ss, pos, z_var)

        peaks = {"full": probe_peaks(full, FULL_PROBE_SIZES),
                 "variant": probe_peaks(variants, VARIANT_PROBE_SIZES)}
    del cache, ref_ss, x
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    capacity = free + torch.cuda.memory_reserved()
    limits = {"full": n_full, "variant": n_var}
    sizes = {"full": FULL_PROBE_SIZES, "variant": VARIANT_PROBE_SIZES}

    def fit(path, fraction, multiple):
        n, fixed, per_item = fit_probe(peaks[path], baseline, capacity,
                                       fraction, multiple)
        return min(n, limits[path]), fixed, per_item

    out = {"auto": {}, "ceiling": {}, "fixed_bytes": {},
           "bytes_per_item": {}, "probe_oom_at": {}}
    for path in peaks:
        out["auto"][path], fixed, per_item = fit(path, MEMORY_FRACTION,
                                                 MULTIPLE)
        out["ceiling"][path] = fit(path, 1.0, 1)[0]
        out["fixed_bytes"][path] = fixed
        out["bytes_per_item"][path] = per_item
        missing = [b for b in sizes[path] if b not in peaks[path]]
        out["probe_oom_at"][path] = missing[0] if missing else None
    out.update({
        "peak_bytes": {path: {str(b): p for b, p in pk.items()}
                       for path, pk in peaks.items()},
        "baseline_bytes": baseline,
        "capacity_bytes": capacity,
        "total_bytes": total,
        "fraction": MEMORY_FRACTION,
    })
    return out


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


def warm_for(workloads, schedule, seconds):
    """Runs the schedule untimed, over and over, for ``seconds``; returns
    the number of items run."""
    t_end, n = time.perf_counter() + seconds, 0
    while True:
        for k, i in schedule:
            if time.perf_counter() >= t_end:
                return n
            w = workloads[k]
            w.run(w.load(i))
            torch.cuda.synchronize()
            n += 1


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


E2E_PHASES = ("load", "run", "to_host")


def e2e_pass(workloads, schedule, emit):
    """Wall-clock seconds per item: strings (or substitution lists) to CPU
    probabilities. Returns ({name: [total, ...]}, {name: [(load, run,
    to_host), ...]}) in item order:
      load     encode on the CPU, pin, launch the copy to the GPU;
      run      both stages, until the GPU has finished;
      to_host  ``emit``: probabilities or sparse deltas, copied to CPU
               memory (see make_emit).
    The synchronize that ends ``run`` replaces the one the copy would do
    anyway, so the total matches an unsplit pass."""
    times = {w.name: [None] * len(w) for w in workloads}
    phases = {w.name: [None] * len(w) for w in workloads}
    for k, i in schedule:
        w = workloads[k]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        inp = w.load(i, from_string=True)
        t1 = time.perf_counter()
        out = w.run(inp)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        emit(w, i, out)
        t3 = time.perf_counter()
        times[w.name][i] = t3 - t0
        phases[w.name][i] = (t1 - t0, t2 - t1, t3 - t2)
    return times, phases


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
        widths = inc.window_widths(positions, args.check_len)
        z_var, var_ss = inc.variants_track(cache, ref_ss, pos, alt, widths)
        idxs, jxn = inc.head_2d(cache[1], var_ss, pos, z_var, widths=widths)
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


def check_against_unoptimized(models, inc, args, device, n_variants=32):
    """The incremental path with ``inc.opts`` vs the same path without any
    rewrite (Optimizations.none()), eager under FP16 autocast, on the short
    check sequence: the rewrites' own effect on the outputs, separate from
    the incremental scheme's."""
    base = IncrementalEnsemble(list(inc.models), args.out_2d_len,
                               Optimizations.none())
    crop_len = inc.crop_len
    seq = random_sequences(1, args.check_len + crop_len, args.seed + 4)[0]
    positions, alts = random_substitutions(seq, crop_len, args.check_len,
                                           n_variants, args.seed + 5)
    x = encode(seq).to(device)
    pos = torch.tensor(positions, device=device)
    alt = torch.tensor(alts, device=device)
    outs = []
    with inference():
        for e in (inc, base):
            cache, ref_ss = e.reference_1d(x)
            idxs, jxn = e.head_2d(cache[1], ref_ss)
            preds = [(ref_ss[0], idxs[0], jxn[0])]
            widths = e.window_widths(positions, args.check_len)
            z_var, var_ss = e.variants_track(cache, ref_ss, pos, alt, widths)
            idxs, jxn = e.head_2d(cache[1], var_ss, pos, z_var,
                                  widths=widths)
            preds += [(var_ss[j], idxs[j], jxn[j]) for j in range(len(pos))]
            outs.append([tuple(t.cpu() for t in p) for p in preds])
            del cache, z_var
            torch.cuda.empty_cache()
    return {"reference": compare(outs[0][:1], outs[1][:1]),
            "variants": compare(outs[0][1:], outs[1][1:])}


def check_against_full(full, incr):
    """Every variant's incremental prediction vs its full pass (untimed).
    Sequence 0 of the full workload is the reference, sequence 1 + j the
    j-th substitution."""
    ours, start = {}, 1
    for i in range(1, len(incr)):
        out = incr.run(incr.load(i))
        for j in range(incr.predictions(i)):
            ours[start + j] = tuple(t[j].cpu() for t in out)
        start += incr.predictions(i)
    mine, theirs = [], []
    for i, (idx, n) in enumerate(full.batches):
        out = full.run(full.load(i))
        for row in range(n):
            if idx[row] in ours:
                mine.append(ours[idx[row]])
                theirs.append(tuple(t[row].cpu() for t in out))
    return compare(mine, theirs)


def stage_summary(gpu_rounds, e2e_rounds, items, n_predictions,
                  phase_rounds=()):
    g1 = [sum(rnd[i][0] for i in items) for rnd in gpu_rounds]
    g2 = [sum(rnd[i][1] for i in items) for rnd in gpu_rounds]
    gpu = [a + b for a, b in zip(g1, g2)]
    e2e = [sum(rnd[i] for i in items) for rnd in e2e_rounds]
    phases = {
        name: summarize([sum(rnd[i][j] for i in items)
                         for rnd in phase_rounds])
        for j, name in enumerate(E2E_PHASES)
    } if phase_rounds else None
    return {
        "n_predictions": n_predictions,
        "gpu_s": {
            "1d": summarize(g1),
            "2d": summarize(g2),
            "total": summarize(gpu),
        },
        "end_to_end_s": summarize(e2e),
        "end_to_end_phases_s": phases,
        "predictions_per_s": {
            "gpu": n_predictions / statistics.median(gpu),
            "e2e": n_predictions / statistics.median(e2e),
        },
    }


def main(argv=None):
    args = parse_args(argv)
    chosen = {}
    try:
        run(args, chosen)
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError included
        if not (isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(e).lower()):
            raise
        print(f"\nout of GPU memory (full batch "
              f"{chosen.get('full', args.full_batch_size)}, variant batch "
              f"{chosen.get('variant', args.variant_batch_size)}): {e}",
              file=sys.stderr, flush=True)
        sys.exit(OOM_EXIT)


def run(args, chosen):
    if not torch.cuda.is_available():
        sys.exit("a CUDA GPU is required")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    compile_ = not args.no_compile
    if args.freezing:
        import torch._inductor.config as inductor_config
        inductor_config.freezing = True

    t0 = time.perf_counter()
    model_dir = args.model_dir or download_models()
    models = [load_model(p) for p in checkpoint_paths(model_dir)]
    inc = IncrementalEnsemble(models, args.out_2d_len,
                              args.opts).eval().to(device)
    ens = Ensemble(list(inc.models), args.out_2d_len).eval()  # same modules
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
    if args.profile_stacks and not args.profile_dir:
        sys.exit("--profile-stacks needs --profile-dir")

    print("Checking both paths against the original model "
          f"(eager, {args.check_len:,} positions)...", flush=True)
    original_check = check_against_original(models, inc, args, device)
    torch.cuda.empty_cache()
    unoptimized_check = None
    if args.opts != Optimizations.none():
        print(f"Checking the rewrites ({args.opts.spec()}) against the "
              "unoptimized incremental path (eager)...", flush=True)
        unoptimized_check = check_against_unoptimized(models, inc, args,
                                                      device)
        torch.cuda.empty_cache()

    print("Generating sequences...", flush=True)
    seq = random_sequences(1, args.seq_len + m.crop_len, args.seed)[0]
    positions, alts = random_substitutions(seq, m.crop_len, args.seq_len,
                                           args.n_mutations, args.seed + 1,
                                           args.layout)
    seqs = [seq] + [apply_substitution(seq, m.crop_len, p, a)
                    for p, a in zip(positions, alts)]

    print("Measuring memory per batch item (eager)...", flush=True)
    sizing = measure_batch_sizes(inc, ens, seq, args.seq_len, len(seqs),
                                 args.n_mutations, device)
    requested = {"full": args.full_batch_size,
                 "variant": args.variant_batch_size}
    for path, b in requested.items():
        chosen[path] = sizing["auto"][path] if b == "auto" else b
    for path, what, unit, div in (("full", "sequence", "GiB", 2**30),
                                  ("variant", "variant", "MiB", 2**20)):
        per_item = sizing["bytes_per_item"][path]
        cost = ("did not fit even at batch 1 (eager); trying anyway"
                if per_item is None else f"{per_item / div:,.2f} {unit} per "
                f"{what}, ceiling {sizing['ceiling'][path]}")
        oom = sizing["probe_oom_at"][path]
        if per_item is not None and oom is not None:
            cost += f"; probe ran out of memory at batch {oom}"
        mode = "auto" if requested[path] == "auto" else "fixed"
        print(f"  {path} batch {chosen[path]} ({mode}; {cost})", flush=True)
    print(f"  {sizing['capacity_bytes'] / 2**30:.1f} GiB available",
          flush=True)

    stages = Stages(inc, ens, compile_, args.compile_mode)
    emit = make_emit(args)
    full = FullWorkload(seqs, chosen["full"], stages, device)
    incr = IncrementalWorkload(
        seq, positions, alts, chosen["variant"], stages, device,
        widths=lambda pos: inc.window_widths(pos, args.seq_len),
    )
    workloads = [w for w in (full, incr)
                 if args.only is None or w.name == args.only]

    # one variant graph per window width class (trim_windows)
    classes = list(dict.fromkeys(incr.widths))
    if compile_:
        raise_recompile_limit(len(classes))
        forbid_eager_fallback()

    print("Compiling / warming up...", flush=True)
    warmup_s = {f"full batch of {chosen['full']}": warmup(full, 0),
                "reference pass": warmup(incr, 0)}  # builds the cache
    warmup_s[f"variant batch of {chosen['variant']}"] = warmup(incr, 1)
    for widths in classes[1:]:  # first batch of every other width class
        i = 1 + incr.widths.index(widths)
        warmup_s[f"variant batch of {chosen['variant']}, widths {widths}"] = \
            warmup(incr, i)
    for w in (full, incr):
        for i in range(len(w)):
            out = w.run(w.load(i))
            remember_reference(w, i, out)  # item 0 holds it, so it comes first
            # also allocates the pinned output buffers, outside the timing
            emit(w, i, out)
    torch.cuda.synchronize()

    print("Checking incremental predictions against full passes...",
          flush=True)
    full_check = check_against_full(full, incr)
    if compile_ and hasattr(torch._dynamo.config, "error_on_recompile"):
        torch._dynamo.config.error_on_recompile = True

    schedule = interleave([len(w) for w in workloads])
    warm = None
    if args.warm_seconds > 0:
        print(f"Warming the GPU for {args.warm_seconds:.0f} s...", flush=True)
        before = gpu_telemetry()
        t_start = time.perf_counter()
        items = warm_for(workloads, schedule, args.warm_seconds)
        warm = {"seconds": time.perf_counter() - t_start, "items": items,
                "telemetry_start": before, "telemetry_end": gpu_telemetry()}
        print(f"  {items} items | "
              f"{fmt_telemetry(warm['telemetry_start'], warm['telemetry_end'])}",
              flush=True)

    gpu_rounds = {w.name: [] for w in workloads}
    e2e_rounds = {w.name: [] for w in workloads}
    phase_rounds = {w.name: [] for w in workloads}
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
                times, phases = e2e_pass(workloads, sched, emit)
                totals = {n: sum(t) for n, t in times.items()}
                rounds = e2e_rounds
                for name, p in phases.items():
                    phase_rounds[name].append(p)
            elapsed = time.perf_counter() - t_start
            after = gpu_telemetry()
            for name, t in times.items():
                rounds[name].append(t)
            passes.append({
                "round": r + 1, "mode": mode, "order": order,
                "elapsed_s": elapsed, "totals_s": totals,
                "telemetry_start": before, "telemetry_end": after,
            })
            print(f"  round {r + 1}/{args.repeats} {mode:<3}  "
                  f"{fmt_totals(totals)}  | {fmt_telemetry(before, after)}",
                  flush=True)

    profile = None
    if args.profile_dir:
        print(f"Profiling ({args.profile_iters} iterations per workload)...",
              flush=True)
        profile = {}
        targets = []
        if full in workloads:
            targets.append(("full", full, range(min(3, len(full)))))
        if incr in workloads:
            targets += [("incremental_reference", incr, [0]),
                        ("incremental_variants", incr,
                         range(1, min(3, len(incr))))]
        for name, w, items in targets:
            def step(w=w, items=items):
                for i in items:
                    w.run(w.load(i))
            n = sum(w.predictions(i) for i in items)
            profile[name] = {
                "predictions_per_iter": n,
                **profile_workload(name, step, args.profile_dir,
                                   args.profile_iters,
                                   stacks=args.profile_stacks,
                                   predictions_per_iter=n),
            }

    n_in = args.seq_len + m.crop_len
    n_pred = len(seqs)
    summaries = {}
    if full in workloads:
        summaries["full"] = {
            **stage_summary(gpu_rounds["full"], e2e_rounds["full"],
                            range(len(full)), n_pred, phase_rounds["full"]),
            "batch_size": chosen["full"],
            "batches": len(full),
            "peak_gpu_mem_bytes": full.peak_mem,
        }
    if incr in workloads:
        g, e = gpu_rounds["incremental"], e2e_rounds["incremental"]
        ph = phase_rounds["incremental"]
        summaries["incremental"] = {
            "reference": stage_summary(g, e, [0], 1, ph),
            "variants": stage_summary(g, e, range(1, len(incr)),
                                      args.n_mutations, ph),
            "total": stage_summary(g, e, range(len(incr)), n_pred, ph),
            "batch_size": chosen["variant"],
            "peak_gpu_mem_bytes": incr.peak_mem,
        }
    speedup = None
    if len(workloads) == 2:
        speedup = {
            mode: summarize([p["totals_s"]["full"]
                             / p["totals_s"]["incremental"]
                             for p in passes if p["mode"] == mode])
            for mode in ("gpu", "e2e")
        }
    result = {
        "env": environment_info(),
        "config": {
            "seq_len": args.seq_len,
            "n_mutations": args.n_mutations,
            "layout": args.layout,
            "full_batch_size": chosen["full"],
            "variant_batch_size": chosen["variant"],
            "batch_size_mode": {p: "auto" if b == "auto" else "fixed"
                                for p, b in requested.items()},
            "only": args.only,
            "out_2d_len": args.out_2d_len,
            "check_len": args.check_len,
            "compile": compile_,
            "autocast": "float16",
            "cudnn_benchmark": args.cudnn_benchmark,
            "compile_mode": args.compile_mode,
            "freezing": args.freezing,
            "opts": args.opts.spec(),
            "e2e_output": args.e2e_output,
            "min_delta": args.min_delta,
            "host_copy": args.host_copy,
            "warm_seconds": args.warm_seconds,
            "repeats": args.repeats,
            "seed": args.seed,
            "num_models": len(models),
            "num_blocks": m.num_blocks,
            "num_features": m.num_features,
            "crop_len": m.crop_len,
        },
        "batch_sizing": sizing,
        "substitutions": {"positions": positions,
                          "alt": [BASES[a] for a in alts]},
        "geometry": {
            "reach_nt": inc.reach,
            "cache_margin_nt": inc.margin,
            "recompute_fraction": inc.recompute_fraction(n_in),
            "blocks": [vars(p) for p in inc.plans],
        },
        "accuracy": {"original_model": original_check, "full_pass": full_check,
                     "vs_unoptimized": unoptimized_check},
        "load_s": load_s,
        "warmup_s": warmup_s,
        "warm": warm,
        "workloads": summaries,
        "speedup": speedup,
        "profile": profile,
        "passes": passes,
    }
    print()
    print_report(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json_out}")


def fmt_totals(totals):
    return "  ".join(f"{name} {t:7.2f} s" for name, t in totals.items())


def print_report(r):
    c, env, g = r["config"], r["env"], r["geometry"]
    n_pred = c["n_mutations"] + 1
    sizing = r["batch_sizing"]

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
        phases = s.get("end_to_end_phases_s")
        if phases:
            print(f"{indent}  " + ",  ".join(
                f"{name} {phases[name]['median']:.3f} s"
                for name in E2E_PHASES))
        rate = s["predictions_per_s"]
        plural = what + ("es" if what.endswith("s") else "s")
        print(f"{indent}throughput                         "
              f"{rate['gpu']:9.1f} {plural}/s GPU, {rate['e2e']:.1f} "
              f"end-to-end")

    def batch(path):
        mode = c["batch_size_mode"][path]
        return (f"{c[path + '_batch_size']} ({mode}; ceiling "
                f"{sizing['ceiling'][path]})")

    print("SpliceAI2 substitution benchmark")
    print(f"  GPU                {env['gpu']}")
    print(f"  torch/CUDA/cuDNN   {env['torch']} / {env['cuda']} / {env['cudnn']}")
    print(f"  models             {c['num_models']} × {c['num_blocks']} blocks, "
          f"{c['num_features']} features, crop_len {c['crop_len']:,}")
    print(f"  mode               fp16 autocast, "
          f"torch.compile {'on' if c['compile'] else 'off'}, "
          f"cudnn.benchmark {'on' if c['cudnn_benchmark'] else 'off'}, "
          f"heads 1d,2d, out_2d_len {c['out_2d_len']}")
    print(f"  compile            mode {c.get('compile_mode', 'default')}, "
          f"freezing {'on' if c.get('freezing') else 'off'}")
    print(f"  incremental opts   {c.get('opts', 'none')}")
    output = c.get("e2e_output", "full")
    print(f"  end-to-end output  " + (
        f"sparse deltas (|Δp| ≥ {c.get('min_delta')})" if output == "deltas"
        else "none (outputs stay on the GPU)" if output == "none"
        else f"full outputs, {c.get('host_copy', 'pageable')} host copies"))
    print(f"  sequence           {c['seq_len']:,} positions "
          f"({c['seq_len'] + c['crop_len']:,} nt input): reference + "
          f"{c['n_mutations']} {c['layout']} substitutions = {n_pred} "
          f"predictions")
    if c["layout"] == "consecutive":
        ps = r["substitutions"]["positions"]
        print(f"                     output positions {min(ps):,}–{max(ps):,}")
    print(f"  batch sizes        full {batch('full')}, variant "
          f"{batch('variant')}; "
          f"{sizing['capacity_bytes'] / 2**30:.1f} GiB available")
    print(f"  incremental        a substitution reaches ±{g['reach_nt']:,} nt; "
          f"{100 * g['recompute_fraction']:.2f}% of a full pass's block "
          f"outputs recomputed per variant")
    warm = r["warm"]
    warm_line = ("none" if warm is None else
                 f"{warm['seconds']:.0f} s untimed, {warm['items']} items; "
                 f"{fmt_telemetry(warm['telemetry_start'], warm['telemetry_end'])}")
    print(f"  GPU warm-up        {warm_line}")
    print(f"  timing             {c['repeats']} interleaved rounds (ABBA), "
          f"median [min–max]"
          + (f"; only {c['only']} timed" if c["only"] else ""))
    print()
    print("Accuracy, max |Δ probability| (2D: predictions with identical top-k)")
    a = r["accuracy"]
    print(f"  original model, {c['check_len']:,} positions, eager")
    print(f"    reference pass   {accuracy(a['original_model']['reference'])}")
    print(f"    variants         {accuracy(a['original_model']['variants'])}")
    if a.get("vs_unoptimized"):
        print(f"  rewrites vs unoptimized incremental, {c['check_len']:,} "
              f"positions, eager")
        print(f"    reference pass   "
              f"{accuracy(a['vs_unoptimized']['reference'])}")
        print(f"    variants         "
              f"{accuracy(a['vs_unoptimized']['variants'])}")
    print(f"  incremental vs full pass, compiled")
    print(f"    all variants     {accuracy(a['full_pass'])}")
    print()
    print(f"Model load                               {r['load_s']:9.3f} s")
    print("First call (compile + cuDNN setup)")
    for label_, wu in r["warmup_s"].items():
        print(f"  {label_:<28} 1D {wu['1d']:7.3f} s, 2D {wu['2d']:7.3f} s")

    w = r["workloads"]
    if "full" in w:
        print()
        print(f"Full recompute (naive): {n_pred} forward passes in "
              f"{w['full']['batches']} batches of {w['full']['batch_size']}")
        stages(w["full"], n_pred, "prediction")
        print(f"  Peak GPU memory                    "
              f"{w['full']['peak_gpu_mem_bytes'] / 2**30:9.2f} GiB")
    if "incremental" in w:
        print()
        print(f"Incremental: reference pass + {c['n_mutations']} variants in "
              f"batches of {w['incremental']['batch_size']}")
        print("  reference pass (keeps per-block activations)")
        stages(w["incremental"]["reference"], 1, "pass")
        print("  variants")
        stages(w["incremental"]["variants"], c["n_mutations"], "variant")
        print("  total")
        stages(w["incremental"]["total"], n_pred, "prediction")
        print(f"  Peak GPU memory                    "
              f"{w['incremental']['peak_gpu_mem_bytes'] / 2**30:9.2f} GiB")
    s = r["speedup"]
    if s is not None:
        print()
        print("Speedup, full / incremental (median [min–max] of per-pass ratios)")
        print(f"  GPU time     {s['gpu']['median']:6.1f}×  "
              f"[{s['gpu']['min']:.1f}–{s['gpu']['max']:.1f}]")
        print(f"  end-to-end   {s['e2e']['median']:6.1f}×  "
              f"[{s['e2e']['min']:.1f}–{s['e2e']['max']:.1f}]")

    if r["profile"]:
        print()
        print("Profile (per iteration; GPU = summed kernel time)")
        for name, p in r["profile"].items():
            print(f"  {name:<22} {p['predictions_per_iter']:4d} predictions, "
                  f"GPU {p['device_ms_per_iter']:9.1f} ms, CPU ops "
                  f"{p['cpu_ms_per_iter']:9.1f} ms, "
                  f"{p['kernel_launches_per_iter']:,.0f} kernel launches")
            for k in p["top_kernels"][:5]:
                print(f"    {100 * (k['share'] or 0):5.1f}%  "
                      f"{k['device_ms_per_iter']:8.2f} ms  {k['name'][:70]}")
            print(f"    trace {p['trace']}")
            if "flamegraph_gpu" in p:
                print(f"    flamegraphs {p['flamegraph_gpu']}, "
                      f"{p['flamegraph_cpu']}")

    print()
    print("Pass log")
    for p in r["passes"]:
        print(f"  round {p['round']} {p['mode']:<3} {p['order']:<7}  "
              f"{fmt_totals(p['totals_s'])}"
              f"  | {fmt_telemetry(p['telemetry_start'], p['telemetry_end'])}")


if __name__ == "__main__":
    main()
