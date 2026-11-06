"""Ablation of the incremental path's optimizations with spliceai2-mutbench.

Every configuration is its own mutbench process on the same workload, once
per sequence length (default: reference + 499 random substitutions of a
10,000-, 20,000-, 50,000- and 100,000-position sequence, as genes of those
lengths; variant batch 32, only the incremental workload timed):

  none, all              no rewrite / every rewrite
  only_<name>            one rewrite on its own: its gain over ``none``
  all_minus_<name>       every rewrite but one: its gain on top of the others
  all_freezing           all + Inductor freezing
  all_reduce_overhead    all + torch.compile mode reduce-overhead (CUDA graphs)
  all_max_autotune       all + mode max-autotune-no-cudagraphs
  all_pageable           all, end-to-end outputs copied with plain .cpu()
  all_deltas             all, end-to-end output as sparse deltas (like
                         spliceai2-ism) instead of full tracks
  all_no_output          all, end-to-end without any output (nothing is
                         copied to the CPU): the cost of producing outputs
  naive                  all, with the naive full passes (plain SpliceAI2,
                         no rewrites) timed in the same process

ablation.md and ablation.json hold, per length and configuration: GPU
throughput, time per variant and per reference pass, end-to-end throughput
and its phases, peak memory, compile time and accuracy (vs the naive full
passes, and vs the unoptimized incremental path), plus the gain of each
rewrite per length. Each length's runs are in a subdirectory (``10kb/``).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from spliceai2_bench.mutation_benchmark import OOM_EXIT
from spliceai2_bench.suite import run
from spliceai2_ism.incremental import Optimizations

MODULE = "spliceai2_bench.mutation_benchmark"
DEFAULT_SEQ_LENS = (10_000, 20_000, 50_000, 100_000)


def configs():
    """{name: extra mutbench args}, in run order."""
    out = {"none": ["--opts", "none"], "all": ["--opts", "all"]}
    for name in Optimizations.names():
        out[f"only_{name}"] = ["--opts", name]
    for name in Optimizations.names():
        out[f"all_minus_{name}"] = ["--opts", f"all,-{name}"]
    out.update({
        "all_freezing": ["--opts", "all", "--freezing"],
        "all_reduce_overhead": ["--opts", "all",
                                "--compile-mode", "reduce-overhead"],
        "all_max_autotune": ["--opts", "all", "--compile-mode",
                             "max-autotune-no-cudagraphs"],
        "all_pageable": ["--opts", "all", "--host-copy", "pageable"],
        "all_deltas": ["--opts", "all", "--e2e-output", "deltas"],
        "all_no_output": ["--opts", "all", "--e2e-output", "none"],
        "naive": ["--opts", "all", "--full-batch-size", "auto"],
    })
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ablation of the incremental optimizations "
                    "(spliceai2-mutbench per configuration and length)",
        allow_abbrev=False,
    )
    p.add_argument("--seq-lens", type=_lengths,
                   default=list(DEFAULT_SEQ_LENS),
                   help="comma-separated output lengths, one workload each "
                        "(default: %(default)s)")
    p.add_argument("--n-mutations", type=int, default=499)
    p.add_argument("--layout", choices=["random", "consecutive"],
                   default="random")
    p.add_argument("--variant-batch-size", type=int, default=32)
    p.add_argument("--warm-seconds", type=float, default=60)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--configs",
                   help="comma-separated subset of: " + ", ".join(configs()))
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--out-dir", type=Path,
                   help="default: results/ablation-<timestamp>")
    args = p.parse_args(argv)
    known = configs()
    if args.configs:
        names = [c for c in args.configs.split(",") if c]
        bad = [c for c in names if c not in known]
        if bad or not names:
            p.error(f"--configs: unknown {', '.join(bad) or '(empty)'}")
        args.configs = names
    else:
        args.configs = list(known)
    if args.n_mutations > min(args.seq_lens):
        p.error(f"--n-mutations {args.n_mutations} exceeds the shortest "
                f"--seq-lens ({min(args.seq_lens):,})")
    return args


def _lengths(value):
    try:
        out = [int(v) for v in value.split(",") if v.strip()]
    except ValueError:
        out = []
    if not out or any(v <= 0 for v in out) or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError(
            f"{value!r}: expected distinct positive lengths, e.g. "
            "10000,20000,50000,100000")
    return out


def length_label(seq_len):
    """Short name of a sequence length: 10kb, or 12345nt."""
    return f"{seq_len // 1000}kb" if seq_len % 1000 == 0 else f"{seq_len}nt"


def bench_args(args, name, seq_len=None):
    seq_len = args.seq_lens[0] if seq_len is None else seq_len
    out = ["--seq-len", str(seq_len),
           "--n-mutations", str(args.n_mutations),
           "--layout", args.layout,
           "--variant-batch-size", str(args.variant_batch_size),
           "--warm-seconds", str(args.warm_seconds),
           "--repeats", str(args.repeats),
           "--seed", str(args.seed)]
    if name != "naive":
        out += ["--only", "incremental"]
    if args.model_dir:
        out += ["--model-dir", str(args.model_dir)]
    return out + configs()[name]


def row(r):
    """Headline numbers of one mutbench result."""
    w = r["workloads"]["incremental"]
    n_var = r["config"]["n_mutations"]
    acc = r["accuracy"]
    unopt = (acc.get("vs_unoptimized") or {}).get("variants")
    phases = w["total"].get("end_to_end_phases_s") or {}
    compile_s = sum(v["1d"] + v["2d"] for k, v in r["warmup_s"].items()
                    if not k.startswith("full"))
    out = {
        "opts": r["config"].get("opts"),
        "predictions_per_s": w["total"]["predictions_per_s"]["gpu"],
        "e2e_predictions_per_s": w["total"]["predictions_per_s"]["e2e"],
        "per_variant_ms":
            w["variants"]["gpu_s"]["total"]["median"] / n_var * 1e3,
        "variant_2d_ms":
            w["variants"]["gpu_s"]["2d"]["median"] / n_var * 1e3,
        "reference_ms": w["reference"]["gpu_s"]["total"]["median"] * 1e3,
        "e2e_phases_s": {k: v["median"] for k, v in phases.items()},
        "peak_gpu_gib": w["peak_gpu_mem_bytes"] / 2**30,
        "compile_s": compile_s,
        "vs_full": {k: acc["full_pass"][k] for k in
                    ("max_dp_1d", "max_dp_2d", "topk_identical", "n",
                     "mean_topk_overlap")},
        "vs_unoptimized": None if unopt is None else {
            k: unopt[k] for k in ("max_dp_1d", "max_dp_2d", "topk_identical",
                                  "n", "mean_topk_overlap")},
    }
    if "full" in r["workloads"]:
        out["naive_predictions_per_s"] = \
            r["workloads"]["full"]["predictions_per_s"]["gpu"]
        out["naive_speedup"] = r["speedup"]["gpu"]["median"]
    return out


def gains(rows):
    """Per rewrite: GPU time per variant of ``none`` / ``only_<name>`` (its
    gain on its own) and of ``all_minus_<name>`` / ``all`` (its gain on top
    of every other rewrite); None where a run is missing."""
    def ratio(a, b):
        if a in rows and b in rows:
            return rows[a]["per_variant_ms"] / rows[b]["per_variant_ms"]
        return None

    out = {name: {"alone": ratio("none", f"only_{name}"),
                  "on_top": ratio(f"all_minus_{name}", "all")}
           for name in Optimizations.names()}
    out["all"] = {"alone": ratio("none", "all"), "on_top": None}
    for cfg in ("all_freezing", "all_reduce_overhead", "all_max_autotune"):
        out[cfg] = {"alone": None, "on_top": ratio("all", cfg)}
    return out


def _x(v):
    return "–" if v is None else f"{v:.2f}×"


def _dp(a):
    if a is None:
        return "–"
    d2 = "n/a" if a["max_dp_2d"] is None else f"{a['max_dp_2d']:.1e}"
    return (f"{a['max_dp_1d']:.1e} / {d2} / "
            f"{a['topk_identical']}/{a['n']}")


def _runs_table(rows, status, configs):
    lines = [
        "| Config | Status | pred/s GPU | ms/variant (2D) | Reference pass | "
        "e2e pred/s | e2e load / run / to_host | Peak | Compile | "
        "vs naive full passes | vs unoptimized incremental |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in configs:
        r = rows.get(name)
        if r is None:
            lines.append(f"| {name} | {status.get(name, 'not run')} |"
                         + " |" * 9)
            continue
        ph = r["e2e_phases_s"]
        phases = " / ".join(f"{ph[k]:.2f}" for k in ("load", "run", "to_host")
                            if k in ph) or "–"
        naive = ""
        if "naive_predictions_per_s" in r:
            naive = (f" (naive {r['naive_predictions_per_s']:.1f}, "
                     f"{r['naive_speedup']:.1f}×)")
        lines.append(
            f"| {name} | ok | {r['predictions_per_s']:.1f}{naive} | "
            f"{r['per_variant_ms']:.2f} ({r['variant_2d_ms']:.2f}) | "
            f"{r['reference_ms']:.1f} ms | {r['e2e_predictions_per_s']:.1f} | "
            f"{phases} s | {r['peak_gpu_gib']:.1f} GiB | "
            f"{r['compile_s']:.0f} s | {_dp(r['vs_full'])} | "
            f"{_dp(r['vs_unoptimized'])} |"
        )
    return lines


def _gain_cell(name, g):
    """Rewrites: alone / on top; ``all``: alone; compile settings: on top."""
    if name in Optimizations.names():
        return f"{_x(g['alone'])} / {_x(g['on_top'])}"
    return _x(g["alone"] if name == "all" else g["on_top"])


# end-to-end output modes, all on top of every rewrite
OUTPUT_MODES = (
    ("no output", "all_no_output"),
    ("sparse deltas", "all_deltas"),
    ("full, pinned", "all"),
    ("full, pageable", "all_pageable"),
)

# all vs all_no_output: (title, row key or e2e phase, format)
OUTPUT_METRICS = (
    ("GPU pred/s", "predictions_per_s", "{:.1f}"),
    ("ms/variant (GPU)", "per_variant_ms", "{:.3f}"),
    ("ms/variant, 2D part (GPU)", "variant_2d_ms", "{:.3f}"),
    ("reference pass, ms (GPU)", "reference_ms", "{:.1f}"),
    ("e2e pred/s", "e2e_predictions_per_s", "{:.1f}"),
    ("e2e load, s", "load", "{:.3f}"),
    ("e2e run, s", "run", "{:.3f}"),
    ("e2e to_host, s", "to_host", "{:.3f}"),
    ("peak GPU memory, GiB", "peak_gpu_gib", "{:.2f}"),
    ("compile, s", "compile_s", "{:.0f}"),
)


def _metric(r, key):
    if r is None:
        return None
    if key in ("load", "run", "to_host"):
        return r["e2e_phases_s"].get(key)
    return r.get(key)


def output_cost(rows):
    """Per output mode (config name): end-to-end throughput, relative to
    ``all_no_output`` (below 1: slower), the time spent producing outputs
    (to_host, s per pass of all items) and its share of the end-to-end
    time; None where a run is missing. Plus ``all_vs_no_output``: every
    metric of ``all`` and ``all_no_output`` and their ratio."""
    base = rows.get("all_no_output")
    modes = {}
    for _, cfg in OUTPUT_MODES:
        r = rows.get(cfg)
        if r is None:
            modes[cfg] = None
            continue
        ph = r["e2e_phases_s"]
        total = sum(ph.values())
        modes[cfg] = {
            "e2e_predictions_per_s": r["e2e_predictions_per_s"],
            "relative_to_no_output": None if base is None else
                r["e2e_predictions_per_s"] / base["e2e_predictions_per_s"],
            "to_host_s": ph.get("to_host"),
            "to_host_share": None if not total or "to_host" not in ph
                else ph["to_host"] / total,
        }
    compare = {}
    for _, key, _ in OUTPUT_METRICS:
        a, b = _metric(rows.get("all"), key), _metric(base, key)
        compare[key] = {"all": a, "all_no_output": b,
                        "ratio": a / b if a is not None and b else None}
    return {"modes": modes, "all_vs_no_output": compare}


def _output_section(results, labels):
    cost = {k: output_cost(results[k]["rows"]) for k in labels}
    head = ["| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    lines = [
        "", "## Output cost", "",
        "End-to-end, every rewrite on, by output mode: `all_no_output` "
        "(nothing leaves the GPU), `all_deltas` (sparse deltas, as "
        "`spliceai2-ism` produces them), `all` (full tracks and junction "
        "matrices into reused pinned buffers), `all_pageable` (the same "
        "with plain `.cpu()`). Relative: end-to-end throughput vs no "
        "output (below 1: slower). to_host: seconds per pass over all "
        "items, and its share of the end-to-end time. The benchmark never "
        "writes files; Parquet writing in `spliceai2-ism` is not included.",
        "", *head,
    ]
    for title, cfg in OUTPUT_MODES:
        cells = []
        for k in labels:
            m = cost[k]["modes"][cfg]
            if m is None:
                cells.append("–")
                continue
            rel = m["relative_to_no_output"]
            cells.append(f"{m['e2e_predictions_per_s']:.1f}"
                         + ("" if rel is None or cfg == "all_no_output"
                            else f" ({rel:.2f}×)"))
        lines.append(f"| e2e pred/s, {title} | " + " | ".join(cells) + " |")
    for title, cfg in OUTPUT_MODES[1:]:
        cells = []
        for k in labels:
            m = cost[k]["modes"][cfg]
            if m is None or m["to_host_s"] is None:
                cells.append("–")
                continue
            share = m["to_host_share"]
            cells.append(f"{m['to_host_s']:.3f} s"
                         + ("" if share is None else f" ({share:.0%})"))
        lines.append(f"| to_host, {title} | " + " | ".join(cells) + " |")

    lines += [
        "", "### `all` vs `all_no_output`", "",
        "Every metric of the two runs, as `all` / `all_no_output` (ratio). "
        "They run the same GPU work, so the GPU rows differ only by "
        "run-to-run noise; the output cost is in the end-to-end rows.",
        "", *head,
    ]
    for title, key, fmt in OUTPUT_METRICS:
        cells = []
        for k in labels:
            c = cost[k]["all_vs_no_output"][key]
            if c["all"] is None or c["all_no_output"] is None:
                cells.append("–")
                continue
            cells.append(f"{fmt.format(c['all'])} / "
                         f"{fmt.format(c['all_no_output'])}"
                         + ("" if c["ratio"] is None
                            else f" ({c['ratio']:.2f}×)"))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return lines



def markdown(results, args, env):
    """``results``: {length label: {"seq_len", "rows", "status"}}, in run
    order."""
    labels = list(results)
    lines = [
        "# SpliceAI2 incremental: optimization ablation", "",
        f"- GPU: {env.get('gpu', '?')}; torch {env.get('torch', '?')}",
        f"- reference + {args.n_mutations} {args.layout} substitutions of a "
        "sequence of each length ("
        + ", ".join(f"{results[k]['seq_len']:,}" for k in labels)
        + f" positions); variant batch {args.variant_batch_size}; "
        f"{args.warm_seconds:.0f} s warm-up, {args.repeats} rounds; GPU time "
        "medians",
        "- gain *alone*: `none` → `only_<name>`; *on top*: "
        "`all_minus_<name>` → `all` (GPU time per variant); cells are "
        "alone / on top",
        "",
        "## Gains by sequence length", "",
        "| Optimization | " + " | ".join(labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    per_len = {k: gains(results[k]["rows"]) for k in labels}
    for name in per_len[labels[0]] if labels else ():
        lines.append(f"| {name} | " + " | ".join(
            _gain_cell(name, per_len[k][name]) for k in labels) + " |")

    def headline(title, key, fmt, cfg):
        cells = []
        for k in labels:
            r = results[k]["rows"].get(cfg)
            cells.append("–" if r is None or key not in r
                         else fmt.format(r[key]))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")

    headline("ms/variant, `none`", "per_variant_ms", "{:.2f}", "none")
    headline("ms/variant, `all`", "per_variant_ms", "{:.2f}", "all")
    headline("pred/s GPU, `all`", "predictions_per_s", "{:.0f}", "all")
    headline("speedup over naive (`naive`)", "naive_speedup", "{:.1f}×",
             "naive")

    lines += _output_section(results, labels)

    for k in labels:
        res = results[k]
        lines += [
            "", f"## {k}: {res['seq_len']:,} positions", "",
            "Accuracy columns: max \\|Δp\\| 1D / 2D / variants with "
            "identical top-k sites.", "",
            *_runs_table(res["rows"], res["status"], args.configs),
        ]
    return "\n".join(lines) + "\n"


def write(out_dir, args, env, results, runs):
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    lengths = {k: {**v, "gains": gains(v["rows"]),
                   "output_cost": output_cost(v["rows"])}
               for k, v in results.items()}
    (out_dir / "ablation.json").write_text(json.dumps(
        {"config": config, "env": env, "lengths": lengths, "runs": runs},
        indent=2))
    md = markdown(results, args, env)
    (out_dir / "ablation.md").write_text(md)
    return md


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or Path("results") / (
        "ablation-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    results, runs, env_info = {}, {}, {}
    for seq_len in args.seq_lens:
        label = length_label(seq_len)
        sub = out_dir / label
        sub.mkdir(exist_ok=True)
        rows, status = {}, {}
        results[label] = {"seq_len": seq_len, "rows": rows, "status": status}
        for name in args.configs:
            res = run(f"{label}/{name}", MODULE,
                      bench_args(args, name, seq_len), out_dir, env)
            code = res["exit_code"]
            status[name] = {0: "ok", OOM_EXIT: "oom"}.get(code, "failed")
            runs[f"{label}/{name}"] = {**res, "status": status[name]}
            path = sub / f"{name}.json"
            if code == 0 and path.exists():
                r = json.loads(path.read_text())
                env_info = env_info or r["env"]
                rows[name] = row(r)
            elif code == 0:
                status[name] = "failed"
            write(out_dir, args, env_info, results, runs)  # as we go

    md = write(out_dir, args, env_info, results, runs)
    print("\n" + md)
    print(f"results in {out_dir}/")
    failed = any(s != "ok" for v in results.values()
                 for s in v["status"].values())
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
