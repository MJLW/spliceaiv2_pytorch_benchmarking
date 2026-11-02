"""Ablation of the incremental path's optimizations with spliceai2-mutbench.

Every configuration is its own mutbench process on the same workload
(default: reference + 499 random substitutions of a 100,000-position
sequence, variant batch 32, only the incremental workload timed):

  none, all              no rewrite / every rewrite
  only_<name>            one rewrite on its own: its gain over ``none``
  all_minus_<name>       every rewrite but one: its gain on top of the others
  all_freezing           all + Inductor freezing
  all_reduce_overhead    all + torch.compile mode reduce-overhead (CUDA graphs)
  all_max_autotune       all + mode max-autotune-no-cudagraphs
  all_pageable           all, end-to-end outputs copied with plain .cpu()
  all_deltas             all, end-to-end output as sparse deltas (like
                         spliceai2-ism) instead of full tracks
  naive                  all, with the naive full passes (plain SpliceAI2,
                         no rewrites) timed in the same process

ablation.md and ablation.json hold, per configuration: GPU throughput, time
per variant and per reference pass, end-to-end throughput and its phases,
peak memory, compile time and accuracy (vs the naive full passes, and vs
the unoptimized incremental path), plus the gain of each rewrite.
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
        "naive": ["--opts", "all", "--full-batch-size", "auto"],
    })
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Ablation of the incremental optimizations "
                    "(spliceai2-mutbench per configuration)"
    )
    p.add_argument("--seq-len", type=int, default=100_000)
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
    return args


def bench_args(args, name):
    out = ["--seq-len", str(args.seq_len),
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


def markdown(rows, status, args, env):
    lines = [
        "# SpliceAI2 incremental: optimization ablation", "",
        f"- GPU: {env.get('gpu', '?')}; torch {env.get('torch', '?')}",
        f"- reference + {args.n_mutations} {args.layout} substitutions of a "
        f"{args.seq_len:,}-position sequence; variant batch "
        f"{args.variant_batch_size}; {args.warm_seconds:.0f} s warm-up, "
        f"{args.repeats} rounds; GPU time medians",
        "- gain *alone*: `none` → `only_<name>`; *on top*: "
        "`all_minus_<name>` → `all` (GPU time per variant)",
        "",
        "## Gains", "",
        "| Optimization | Gain alone | Gain on top of the others |",
        "|---|---|---|",
    ]
    for name, g in gains(rows).items():
        lines.append(f"| {name} | {_x(g['alone'])} | {_x(g['on_top'])} |")
    lines += [
        "", "## Runs", "",
        "Accuracy columns: max \\|Δp\\| 1D / 2D / variants with identical "
        "top-k sites.", "",
        "| Config | Status | pred/s GPU | ms/variant (2D) | Reference pass | "
        "e2e pred/s | e2e load / run / to_host | Peak | Compile | "
        "vs naive full passes | vs unoptimized incremental |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in args.configs:
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
    return "\n".join(lines) + "\n"


def write(out_dir, args, env, rows, status, runs):
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    (out_dir / "ablation.json").write_text(json.dumps(
        {"config": config, "env": env, "rows": rows, "gains": gains(rows),
         "runs": runs}, indent=2))
    md = markdown(rows, status, args, env)
    (out_dir / "ablation.md").write_text(md)
    return md


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or Path("results") / (
        "ablation-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    rows, status, runs, env_info = {}, {}, {}, {}
    for name in args.configs:
        res = run(name, MODULE, bench_args(args, name), out_dir, env)
        code = res["exit_code"]
        status[name] = {0: "ok", OOM_EXIT: "oom"}.get(code, "failed")
        runs[name] = {**res, "status": status[name]}
        path = out_dir / f"{name}.json"
        if code == 0 and path.exists():
            r = json.loads(path.read_text())
            env_info = env_info or r["env"]
            rows[name] = row(r)
        elif code == 0:
            status[name] = "failed"
        write(out_dir, args, env_info, rows, status, runs)  # as we go

    md = write(out_dir, args, env_info, rows, status, runs)
    print("\n" + md)
    print(f"results in {out_dir}/")
    sys.exit(1 if any(s != "ok" for s in status.values()) else 0)


if __name__ == "__main__":
    main()
