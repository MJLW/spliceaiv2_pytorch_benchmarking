"""Runs the complete benchmark suite with one command and writes a summary.

  1. fixed:        --n-seqs sequences × --seq-len positions        (spliceai2-bench)
  2. variable:     --n-seqs sequences, lengths ~ N(--seq-len, --var-std)  (same run)
  3. random:       reference + (--n-seqs - 1) substitutions at random positions
  4. consecutive:  reference + (--n-seqs - 1) substitutions at adjacent positions
                   (3 and 4: spliceai2-mutbench, incremental vs full passes)

spliceai2-bench runs once per --fixed-batch-sizes entry; its variable
workload is always batch 1, and the summary takes it from the first run.
Every benchmark runs in its own process, so GPU memory and torch.compile
settings start fresh, and writes <name>.json and <name>.log to the output
directory. summary.md and summary.json collect the headline numbers. A
failing benchmark is recorded and the suite moves on to the next one.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SpliceAI2 benchmark suite")
    p.add_argument("--n-seqs", type=int, default=500,
                   help="predictions per benchmark")
    p.add_argument("--seq-len", type=int, default=100_000)
    p.add_argument("--var-std", type=float, default=10_000)
    p.add_argument("--fixed-batch-sizes", default="1,8",
                   help="comma-separated batch sizes for the fixed workload")
    p.add_argument("--variant-batch-size", type=int, default=100,
                   help="substitutions per incremental batch")
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--out-dir", type=Path,
                   help="default: results/suite-<timestamp>")
    return p.parse_args(argv)


def benchmarks(args):
    """(name, module, arguments) per benchmark, in run order."""
    common = ["--seq-len", str(args.seq_len), "--repeats", str(args.repeats),
              "--seed", str(args.seed)]
    if args.cudnn_benchmark:
        common.append("--cudnn-benchmark")
    if args.model_dir:
        common += ["--model-dir", str(args.model_dir)]
    runs = []
    for bs in (int(b) for b in args.fixed_batch_sizes.split(",")):
        runs.append((f"bench_batch{bs}", "spliceai2_bench.benchmark", [
            *common, "--n-seqs", str(args.n_seqs), "--batch-size", str(bs),
            "--var-std", str(args.var_std),
        ]))
    for layout in ("random", "consecutive"):
        runs.append((f"mutations_{layout}", "spliceai2_bench.mutation_benchmark", [
            *common, "--n-mutations", str(args.n_seqs - 1),
            "--layout", layout,
            "--variant-batch-size", str(args.variant_batch_size),
        ]))
    return runs


def run(name, module, bench_args, out_dir, env):
    """Runs one benchmark, streaming its output to the console and a log."""
    cmd = [sys.executable, "-m", module, *bench_args,
           "--json-out", str(out_dir / f"{name}.json")]
    print(f"\n=== {name}\n    {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    with open(out_dir / f"{name}.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    return {"exit_code": code, "wall_s": time.perf_counter() - t0}


def telemetry_range(passes):
    """Min–max SM clock and max temperature over all timed passes."""
    clocks = [p[k]["sm_clock_mhz"] for p in passes
              for k in ("telemetry_start", "telemetry_end")
              if p[k]["sm_clock_mhz"] is not None]
    temps = [p[k]["temp_c"] for p in passes
             for k in ("telemetry_start", "telemetry_end")
             if p[k]["temp_c"] is not None]
    return {"sm_clock_mhz": [min(clocks), max(clocks)] if clocks else None,
            "max_temp_c": max(temps) if temps else None}


def summarize(name, r):
    """Headline numbers of one benchmark's JSON."""
    out = {"telemetry": telemetry_range(r["passes"])}
    if name.startswith("bench"):
        for wname, w in r["workloads"].items():
            out[wname] = {
                "n": w["n_seqs"],
                "batch_size": w["batch_size"],
                "gpu_s": w["gpu_s"]["total"]["median"],
                "gpu_1d_s": w["gpu_s"]["1d"]["median"],
                "gpu_2d_s": (w["gpu_s"]["2d"] or {}).get("median"),
                "e2e_s": w["end_to_end_s"]["median"],
                "padding_overhead": w["padding_overhead"],
                "peak_gpu_gib": w["peak_gpu_mem_bytes"] / 2**30,
            }
    else:
        full, inc = r["workloads"]["full"], r["workloads"]["incremental"]
        n_var = r["config"]["n_mutations"]
        acc = r["accuracy"]["full_pass"]
        out.update({
            "n": n_var + 1,
            "layout": r["config"]["layout"],
            "variant_batch_size": r["config"]["variant_batch_size"],
            "full_gpu_s": full["gpu_s"]["total"]["median"],
            "full_e2e_s": full["end_to_end_s"]["median"],
            "incremental_gpu_s": inc["total"]["gpu_s"]["total"]["median"],
            "incremental_e2e_s": inc["total"]["end_to_end_s"]["median"],
            "reference_pass_gpu_s": inc["reference"]["gpu_s"]["total"]["median"],
            "per_variant_gpu_ms":
                inc["variants"]["gpu_s"]["total"]["median"] / n_var * 1e3,
            "speedup_gpu": r["speedup"]["gpu"]["median"],
            "speedup_e2e": r["speedup"]["e2e"]["median"],
            "max_dp_1d": acc["max_dp_1d"],
            "max_dp_2d": acc["max_dp_2d"],
            "topk_overlap": acc["mean_topk_overlap"],
            "peak_gpu_gib": max(full["peak_gpu_mem_bytes"],
                                inc["peak_gpu_mem_bytes"]) / 2**30,
        })
    return out


def markdown(results, args, env_info):
    lines = [
        "# SpliceAI2 benchmark suite", "",
        f"- GPU: {env_info.get('gpu', '?')}; torch {env_info.get('torch', '?')}, "
        f"CUDA {env_info.get('cuda', '?')}, cuDNN {env_info.get('cudnn', '?')}",
        f"- {args.n_seqs} predictions per benchmark, {args.seq_len:,} output "
        f"positions (+131,072 nt context), FP16 autocast + torch.compile, "
        f"{args.repeats} interleaved rounds; times are medians over rounds",
        "",
        "## Full forward passes", "",
        "| Benchmark | Batch | GPU total | GPU per seq | End-to-end | "
        "Peak memory | SM clock (MHz) |",
        "|---|---|---|---|---|---|---|",
    ]
    variable_done = False
    for name, res in results.items():
        if not name.startswith("bench") or "summary" not in res:
            continue
        s = res["summary"]
        clk = s["telemetry"]["sm_clock_mhz"]
        clk = f"{clk[0]:.0f}–{clk[1]:.0f}" if clk else "n/a"
        rows = [("fixed", "Fixed 100k")]
        if not variable_done:
            rows.append(("variable", f"Variable N(100k, {args.var_std:,.0f})"))
            variable_done = True
        for key, label in rows:
            w = s[key]
            pad = (f" (+{100 * w['padding_overhead']:.1f}% padding)"
                   if w["padding_overhead"] > 0 else "")
            lines.append(
                f"| {label}{pad} | {w['batch_size']} | {w['gpu_s']:.2f} s | "
                f"{w['gpu_s'] / w['n'] * 1e3:.1f} ms | {w['e2e_s']:.2f} s | "
                f"{w['peak_gpu_gib']:.2f} GiB | {clk} |"
            )
    lines += [
        "", "## Single-base substitutions (reference + variants)", "",
        "| Layout | Full GPU | Incremental GPU | Reference pass | "
        "Per variant | Speedup GPU / e2e | max \\|Δp\\| 1D / 2D | "
        "Top-k overlap | Peak memory |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        if not name.startswith("mutations") or "summary" not in res:
            continue
        s = res["summary"]
        d2 = "n/a" if s["max_dp_2d"] is None else f"{s['max_dp_2d']:.1e}"
        lines.append(
            f"| {s['layout']} | {s['full_gpu_s']:.2f} s | "
            f"{s['incremental_gpu_s']:.2f} s | "
            f"{s['reference_pass_gpu_s'] * 1e3:.0f} ms | "
            f"{s['per_variant_gpu_ms']:.2f} ms | "
            f"{s['speedup_gpu']:.1f}× / {s['speedup_e2e']:.1f}× | "
            f"{s['max_dp_1d']:.1e} / {d2} | {s['topk_overlap']:.4f} | "
            f"{s['peak_gpu_gib']:.2f} GiB |"
        )
    failed = [n for n, res in results.items() if res["exit_code"] != 0]
    lines += ["", "## Runs", ""]
    for name, res in results.items():
        status = "ok" if res["exit_code"] == 0 else f"FAILED (exit {res['exit_code']})"
        lines.append(f"- {name}: {status}, {res['wall_s'] / 60:.1f} min "
                     f"wall clock; log {name}.log")
    if failed:
        lines += ["", f"**{len(failed)} benchmark(s) failed:** "
                  + ", ".join(failed)]
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    if args.n_seqs < 2:
        sys.exit("--n-seqs must be at least 2")
    out_dir = args.out_dir or Path("results") / (
        "suite-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    results, env_info = {}, {}
    for name, module, bench_args in benchmarks(args):
        res = run(name, module, bench_args, out_dir, env)
        path = out_dir / f"{name}.json"
        if res["exit_code"] == 0 and path.exists():
            r = json.loads(path.read_text())
            env_info = env_info or r["env"]
            res["summary"] = summarize(name, r)
        results[name] = res

    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    (out_dir / "summary.json").write_text(json.dumps(
        {"config": config, "env": env_info, "runs": results}, indent=2
    ))
    md = markdown(results, args, env_info)
    (out_dir / "summary.md").write_text(md)
    print("\n" + md)
    print(f"results in {out_dir}/")
    sys.exit(1 if any(r["exit_code"] != 0 for r in results.values()) else 0)


if __name__ == "__main__":
    main()
