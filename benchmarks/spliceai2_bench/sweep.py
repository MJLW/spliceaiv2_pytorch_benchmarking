"""Batch-size sweeps of spliceai2-mutbench: naive full passes vs reference +
incremental, on warm GPUs.

For each layout (random, consecutive) there are two axes:
  naive:        --full-batch-size b. The incremental workload runs in the same
                process at its auto batch size, so each run also gives a
                same-run speedup;
  incremental:  --variant-batch-size v with --only incremental. The full
                passes, the slow part, are not timed; the accuracy check
                still runs them once.
On each axis the batch size starts at --start-batch and doubles until a run
runs out of GPU memory (mutbench exit code 3) or reaches the number of
items. After an out-of-memory run, one more run uses the largest size that
the last successful run's memory measurement says fits (its ceiling), if
that lies between the last size that fit and the one that did not. Each run
is its own process, so an out-of-memory error leaves the next run
unaffected.

Every run writes <layout>_<axis>_b<size>.json and .log, with traces in
<name>.profile/ under --profile. sweep.json and sweep.md collect the tables:
throughput per batch size, best naive vs best incremental per layout, random
vs consecutive and, with --profile, the top GPU kernels per prediction
across batch sizes. With --profile-stacks, each run keeps its own
flamegraphs, and diffs/ adds differential flamegraphs (see diff_jobs).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from spliceai2_bench.flamediff import write_diff
from spliceai2_bench.mutation_benchmark import OOM_EXIT
from spliceai2_bench.suite import run, telemetry_range

AXES = ("naive", "incremental")
LAYOUTS = ("random", "consecutive")
MODULE = "spliceai2_bench.mutation_benchmark"
PATH = {"naive": "full", "incremental": "variant"}  # batch_sizing key


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SpliceAI2 batch-size sweep: naive full passes vs "
                    "reference + incremental"
    )
    p.add_argument("--seq-len", type=int, default=100_000)
    p.add_argument("--n-mutations", type=int, default=499,
                   help="substitutions besides the reference, per run")
    p.add_argument("--layouts", default=",".join(LAYOUTS),
                   help="comma-separated: random, consecutive")
    p.add_argument("--axes", default=",".join(AXES),
                   help="comma-separated: naive, incremental")
    p.add_argument("--start-batch", type=int, default=1,
                   help="first batch size of each axis; doubles from there")
    p.add_argument("--warm-seconds", type=float, default=60,
                   help="untimed GPU warm-up per run")
    p.add_argument("--repeats", type=int, default=2,
                   help="interleaved rounds per run (even: balanced)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="torch.profiler traces for every run")
    p.add_argument("--profile-stacks", action="store_true",
                   help="with --profile: also Python stacks and flamegraphs")
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--out-dir", type=Path,
                   help="default: results/sweep-<timestamp>")
    args = p.parse_args(argv)
    args.layouts = _choices(p, "--layouts", args.layouts, LAYOUTS)
    args.axes = _choices(p, "--axes", args.axes, AXES)
    if args.start_batch < 1:
        p.error("--start-batch must be at least 1")
    if args.profile_stacks and not args.profile:
        p.error("--profile-stacks needs --profile")
    return args


def _choices(p, flag, value, allowed):
    items = [v for v in value.split(",") if v]
    bad = [v for v in items if v not in allowed]
    if bad or not items:
        p.error(f"{flag}: choose from {', '.join(allowed)}")
    return items


def common_args(args, layout):
    out = ["--seq-len", str(args.seq_len),
           "--n-mutations", str(args.n_mutations),
           "--layout", layout,
           "--warm-seconds", str(args.warm_seconds),
           "--repeats", str(args.repeats),
           "--seed", str(args.seed)]
    if args.cudnn_benchmark:
        out.append("--cudnn-benchmark")
    if args.model_dir:
        out += ["--model-dir", str(args.model_dir)]
    return out


def axis_args(axis, b):
    if axis == "naive":
        return ["--full-batch-size", str(b), "--variant-batch-size", "auto"]
    return ["--variant-batch-size", str(b), "--full-batch-size", "auto",
            "--only", "incremental"]


def axis_limit(axis, n_mutations):
    """Items on the axis: sequences for full passes, variants otherwise."""
    return n_mutations + 1 if axis == "naive" else n_mutations


def status_of(exit_code):
    return {0: "ok", OOM_EXIT: "oom"}.get(exit_code, "failed")


def sweep_axis(run_at, path, start, limit):
    """Doubling sweep. ``run_at(b)`` returns (status, result JSON or None),
    status "ok", "oom" or "failed". Returns [(b, status, result), ...] in
    run order."""
    runs, last_ok, b = [], None, min(start, limit)
    while True:
        status, res = run_at(b)
        runs.append((b, status, res))
        if status != "ok":
            break
        last_ok = b
        if b >= limit:
            return runs
        b = min(2 * b, limit)
    if status == "oom" and last_ok is not None:
        ceiling = runs[-2][2]["batch_sizing"]["ceiling"][path]
        b = min(ceiling, runs[-1][0] - 1)
        if b > last_ok:
            runs.append((b, *run_at(b)))
    return runs


def row(axis, b, status, r):
    """Headline numbers of one run."""
    out = {"batch_size": b, "status": status}
    if r is None:
        return out
    if axis == "naive":
        w = r["workloads"]["full"]
        inc = r["workloads"]["incremental"]
        out.update({
            "gpu_s": w["gpu_s"]["total"]["median"],
            "predictions_per_s": w["predictions_per_s"]["gpu"],
            "e2e_predictions_per_s": w["predictions_per_s"]["e2e"],
            "speedup_gpu": r["speedup"]["gpu"]["median"],
            "incremental_batch_size": inc["batch_size"],
            "incremental_predictions_per_s":
                inc["total"]["predictions_per_s"]["gpu"],
        })
    else:
        w = r["workloads"]["incremental"]
        n_var = r["config"]["n_mutations"]
        out.update({
            "gpu_s": w["total"]["gpu_s"]["total"]["median"],
            "predictions_per_s": w["total"]["predictions_per_s"]["gpu"],
            "e2e_predictions_per_s": w["total"]["predictions_per_s"]["e2e"],
            "reference_pass_s": w["reference"]["gpu_s"]["total"]["median"],
            "per_variant_ms":
                w["variants"]["gpu_s"]["total"]["median"] / n_var * 1e3,
            "variants_per_s": w["variants"]["predictions_per_s"]["gpu"],
        })
    out["ms_per_prediction"] = 1e3 / out["predictions_per_s"]
    out["peak_gpu_gib"] = w["peak_gpu_mem_bytes"] / 2**30
    out["ceiling"] = r["batch_sizing"]["ceiling"][PATH[axis]]
    out["telemetry"] = telemetry_range(r["passes"])
    if r.get("profile"):
        out["profile"] = {name: {k: p[k] for k in (
            "trace", "device_ms_per_iter", "cpu_ms_per_iter",
            "kernel_launches_per_iter", "predictions_per_iter",
            "top_kernels", "flamegraph_gpu", "flamegraph_cpu", "stacks_gpu",
            "stacks_cpu") if k in p}
            for name, p in r["profile"].items()}
    return out


def axis_summary(rows):
    """Best throughput, largest batch that fit and smallest that did not."""
    ok = [r for r in rows if r["status"] == "ok"]
    oom = [r["batch_size"] for r in rows if r["status"] == "oom"]
    best = max(ok, key=lambda r: r["predictions_per_s"]) if ok else None
    return {
        "best": best,
        "max_fit": max((r["batch_size"] for r in ok), default=None),
        "first_oom": min(oom, default=None),
        "failed": [r["batch_size"] for r in rows if r["status"] == "failed"],
    }


def diff_jobs(sweep):
    """(baseline stacks, compared stacks, output name) for the automatic
    differential flamegraphs, GPU time per prediction:
      * on each axis, every run vs the axis's first successful run, per
        workload;
      * consecutive vs random, per axis and workload, at the largest batch
        size that both layouts ran."""
    def stacks(r):
        return {name: p["stacks_gpu"] for name, p in r.get("profile", {}).items()
                if "stacks_gpu" in p}

    jobs = []
    for layout, axes in sweep.items():
        for axis, rows in axes.items():
            ok = [r for r in rows if r["status"] == "ok" and stacks(r)]
            if not ok:
                continue
            base = ok[0]
            for r in ok[1:]:
                for name, path in stacks(r).items():
                    if name in stacks(base):
                        jobs.append((
                            stacks(base)[name], path,
                            f"{layout}_{axis}_b{r['batch_size']}_vs_"
                            f"b{base['batch_size']}.{name}.diff_gpu.svg",
                        ))
    if {"random", "consecutive"} <= sweep.keys():
        for axis in AXES:
            by_layout = {
                layout: {r["batch_size"]: r
                         for r in sweep[layout].get(axis, [])
                         if r["status"] == "ok" and stacks(r)}
                for layout in ("random", "consecutive")
            }
            common = by_layout["random"].keys() & by_layout["consecutive"].keys()
            if not common:
                continue
            b = max(common)
            sr = stacks(by_layout["random"][b])
            sc = stacks(by_layout["consecutive"][b])
            for name in sr.keys() & sc.keys():
                jobs.append((sr[name], sc[name],
                             f"consecutive_vs_random_{axis}_b{b}.{name}"
                             f".diff_gpu.svg"))
    return jobs


def kernel_table(rows, top=10):
    """Per workload: the ``top`` kernels by their largest time per
    prediction over the runs, as {workload: (kernels, {batch: {kernel: µs}})}."""
    per = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        for name, p in r.get("profile", {}).items():
            if not p.get("top_kernels") or not p.get("predictions_per_iter"):
                continue
            per.setdefault(name, {})[r["batch_size"]] = {
                k["name"]: k["device_ms_per_iter"] * 1e3
                / p["predictions_per_iter"]
                for k in p["top_kernels"]
            }
    out = {}
    for name, runs in per.items():
        worst = {}
        for us in runs.values():
            for k, v in us.items():
                worst[k] = max(worst.get(k, 0.0), v)
        kernels = sorted(worst, key=lambda k: -worst[k])[:top]
        out[name] = (kernels, runs)
    return out


def _fmt_clock(t):
    clk = t["sm_clock_mhz"]
    return f"{clk[0]:.0f}–{clk[1]:.0f}" if clk else "n/a"


def markdown(sweep, args, env_info, diffs=(), out_root=None):
    lines = [
        "# SpliceAI2 batch-size sweep: naive vs reference + incremental", "",
        f"- GPU: {env_info.get('gpu', '?')}; torch {env_info.get('torch', '?')}, "
        f"CUDA {env_info.get('cuda', '?')}, cuDNN {env_info.get('cudnn', '?')}",
        f"- reference + {args.n_mutations} substitutions of a "
        f"{args.seq_len:,}-position sequence per run; "
        f"{args.warm_seconds:.0f} s untimed GPU warm-up, then "
        f"{args.repeats} interleaved rounds; GPU time medians",
        "- batch sizes double until out of GPU memory; *ceiling* is the "
        "largest size that the run's memory measurement says fits",
        "",
        "## Best per layout", "",
        "| Layout | Best naive (batch) | Best incremental (batch) | "
        "Speedup | Naive max fit / OOM | Incremental max fit / OOM |",
        "|---|---|---|---|---|---|",
    ]

    def best(s):
        b = s["best"]
        return ("n/a" if b is None else
                f"{b['predictions_per_s']:.1f} pred/s ({b['batch_size']})")

    def bounds(s):
        oom = s["first_oom"]
        return f"{s['max_fit'] or 'none'} / {oom if oom else '–'}"

    for layout, axes in sweep.items():
        s = {a: axis_summary(rows) for a, rows in axes.items()}
        naive, inc = s.get("naive"), s.get("incremental")
        speed = "n/a"
        if naive and inc and naive["best"] and inc["best"]:
            speed = (f"{inc['best']['predictions_per_s'] / naive['best']['predictions_per_s']:.1f}×")
        lines.append(
            f"| {layout} | {best(naive) if naive else '–'} | "
            f"{best(inc) if inc else '–'} | {speed} | "
            f"{bounds(naive) if naive else '–'} | "
            f"{bounds(inc) if inc else '–'} |"
        )

    inc_layouts = {l: axis_summary(a["incremental"])
                   for l, a in sweep.items() if "incremental" in a}
    if len(inc_layouts) > 1:
        lines += [
            "", "## Random vs consecutive (incremental)", "",
            "| Layout | Best pred/s | Per variant | Batch | Max fit | "
            "First OOM |",
            "|---|---|---|---|---|---|",
        ]
        for layout, s in inc_layouts.items():
            b = s["best"]
            if b is None:
                lines.append(f"| {layout} | n/a | n/a | n/a | none | "
                             f"{s['first_oom'] or '–'} |")
                continue
            lines.append(
                f"| {layout} | {b['predictions_per_s']:.1f} | "
                f"{b['per_variant_ms']:.2f} ms | {b['batch_size']} | "
                f"{s['max_fit']} | {s['first_oom'] or '–'} |"
            )

    for layout, axes in sweep.items():
        for axis, rows in axes.items():
            lines += ["", f"## {layout}, {axis}", ""]
            if axis == "naive":
                lines += [
                    "| Full batch | Status | GPU total | ms/pred | pred/s | "
                    "e2e pred/s | Same-run speedup (inc. batch) | Peak | "
                    "Ceiling | SM clock (MHz) |",
                    "|---|---|---|---|---|---|---|---|---|---|",
                ]
            else:
                lines += [
                    "| Variant batch | Status | GPU total | ms/pred | pred/s "
                    "| e2e pred/s | Per variant | Reference pass | Peak | "
                    "Ceiling | SM clock (MHz) |",
                    "|---|---|---|---|---|---|---|---|---|---|---|",
                ]
            for r in rows:
                if r["status"] != "ok":
                    cols = 8 if axis == "naive" else 9
                    lines.append(f"| {r['batch_size']} | "
                                 f"{r['status'].upper()} |" + " |" * cols)
                    continue
                head = (f"| {r['batch_size']} | ok | {r['gpu_s']:.2f} s | "
                        f"{r['ms_per_prediction']:.2f} | "
                        f"{r['predictions_per_s']:.1f} | "
                        f"{r['e2e_predictions_per_s']:.1f} | ")
                tail = (f"{r['peak_gpu_gib']:.1f} GiB | {r['ceiling']} | "
                        f"{_fmt_clock(r['telemetry'])} |")
                if axis == "naive":
                    mid = (f"{r['speedup_gpu']:.1f}× "
                           f"({r['incremental_batch_size']}) | ")
                else:
                    mid = (f"{r['per_variant_ms']:.2f} ms | "
                           f"{r['reference_pass_s'] * 1e3:.0f} ms | ")
                lines.append(head + mid + tail)

    for layout, axes in sweep.items():
        for axis, rows in axes.items():
            for name, (kernels, runs) in kernel_table(rows).items():
                sizes = sorted(runs)
                lines += [
                    "", f"### Top GPU kernels, {layout}, {axis}: {name} "
                        f"(µs per prediction)", "",
                    "| Kernel | " + " | ".join(f"b{b}" for b in sizes) + " |",
                    "|---|" + "---|" * len(sizes),
                ]
                for k in kernels:
                    cells = [f"{runs[b][k]:.1f}" if k in runs[b] else "–"
                             for b in sizes]
                    short = k if len(k) <= 60 else k[:57] + "..."
                    short = short.replace("|", "\\|")
                    lines.append(f"| `{short}` | " + " | ".join(cells) + " |")

    if diffs:
        lines += ["", "## Differential flamegraphs (GPU time per prediction)",
                  "", "Red: slower than the baseline, blue: faster. The "
                  "per-run flamegraphs are in each run's `.profile/` "
                  "folder.", ""]
        lines += [f"- `{Path(d).relative_to(out_root) if out_root else d}`"
                  for d in diffs]
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.out_dir or Path("results") / (
        "sweep-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    env_info, sweep, runs = {}, {}, {}
    for layout in args.layouts:
        sweep[layout] = {}
        for axis in args.axes:
            def run_at(b, layout=layout, axis=axis):
                nonlocal env_info
                name = f"{layout}_{axis}_b{b}"
                bench_args = common_args(args, layout) + axis_args(axis, b)
                if args.profile:
                    bench_args += ["--profile-dir",
                                   str(out_dir / f"{name}.profile")]
                    if args.profile_stacks:
                        bench_args.append("--profile-stacks")
                res = run(name, MODULE, bench_args, out_dir, env)
                status = status_of(res["exit_code"])
                runs[name] = {**res, "status": status}
                path = out_dir / f"{name}.json"
                if status == "ok" and path.exists():
                    r = json.loads(path.read_text())
                    env_info = env_info or r["env"]
                    return status, r
                return ("failed" if status == "ok" else status), None

            done = sweep_axis(run_at, PATH[axis], args.start_batch,
                              axis_limit(axis, args.n_mutations))
            sweep[layout][axis] = [row(axis, *d) for d in done]
            # write as we go: a long sweep can be inspected before it ends
            write(out_dir, args, env_info, sweep, runs)

    diffs = []
    if args.profile_stacks:
        jobs = diff_jobs(sweep)
        print(f"\nWriting {len(jobs)} differential flamegraphs...",
              flush=True)
        for a, b, name in jobs:
            try:
                diffs.append(str(write_diff(a, b, out_dir / "diffs" / name)))
            except (OSError, ValueError) as e:
                print(f"  skipped {name}: {e}", flush=True)
    md = write(out_dir, args, env_info, sweep, runs, diffs)
    print("\n" + md)
    print(f"results in {out_dir}/")
    sys.exit(1 if any(r["status"] == "failed" for r in runs.values()) else 0)


def write(out_dir, args, env_info, sweep, runs, diffs=()):
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    summary = {layout: {axis: axis_summary(rows)
                        for axis, rows in axes.items()}
               for layout, axes in sweep.items()}
    (out_dir / "sweep.json").write_text(json.dumps(
        {"config": config, "env": env_info, "sweep": sweep,
         "summary": summary, "runs": runs, "diffs": list(diffs)}, indent=2
    ))
    md = markdown(sweep, args, env_info, diffs, out_dir)
    (out_dir / "sweep.md").write_text(md)
    return md


if __name__ == "__main__":
    main()
