"""Differential flamegraph of two profiled workloads (spliceai2-flamediff).

  spliceai2-flamediff A.stacks_gpu.txt B.stacks_gpu.txt -o diff.svg

Inputs are the folded stacks that --profile-stacks writes. By default both
are normalised per prediction, from the <name>.meta.json beside each file,
so runs with different batch sizes, or naive vs incremental, compare like
for like; --per iteration or --per raw instead. Frame widths follow B
(--width-from a shows frames that B no longer has); red frames are slower
in B, blue faster. The original flamegraphs of both runs are left as they
are.
"""
import argparse
import json
import sys
from pathlib import Path

from spliceai2_bench.profiling import diff_flamegraph_svg, meta_path

PER = ("prediction", "iteration", "raw")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Differential flamegraph of two folded-stack files"
    )
    p.add_argument("a", type=Path, help="baseline <name>.stacks_<kind>.txt")
    p.add_argument("b", type=Path, help="compared <name>.stacks_<kind>.txt")
    p.add_argument("-o", "--out", type=Path, required=True,
                   help="output SVG")
    p.add_argument("--per", choices=PER, default="prediction",
                   help="normalise time per prediction (default), per "
                        "profiled iteration, or not at all")
    p.add_argument("--width-from", choices=["a", "b"], default="b")
    p.add_argument("--label-a", help="default: A's run and workload")
    p.add_argument("--label-b", help="default: B's run and workload")
    return p.parse_args(argv)


def load_folded(path):
    folded = {}
    for line in Path(path).read_text().splitlines():
        stack, _, value = line.rpartition(" ")
        if stack:
            folded[stack] = folded.get(stack, 0.0) + float(value)
    return folded


def scale(path, per):
    """Factor that turns totals over all profiled iterations into µs per
    ``per``."""
    if per == "raw":
        return 1.0
    mp = meta_path(path)
    if not mp.exists():
        raise ValueError(f"{mp} not found; use --per raw")
    meta = json.loads(mp.read_text())
    n = meta["iterations"]
    if per == "prediction":
        if not meta.get("predictions_per_iter"):
            raise ValueError(f"{mp} has no predictions per iteration; use "
                             f"--per iteration")
        n *= meta["predictions_per_iter"]
    return 1.0 / n


def default_label(path):
    """e.g. random_naive_b8/full, from <run>.profile/full.stacks_gpu.txt."""
    p = Path(path)
    run = p.parent.name.removesuffix(".profile")
    return f"{run}/{p.name.split('.stacks_')[0]}"


def write_diff(a, b, out, per="prediction", width_from="b", label_a=None,
               label_b=None):
    label_a = label_a or default_label(a)
    label_b = label_b or default_label(b)
    fa, fb = scale(a, per), scale(b, per)
    folded_a = {s: v * fa for s, v in load_folded(a).items()}
    folded_b = {s: v * fb for s, v in load_folded(b).items()}
    kind = "GPU" if "stacks_gpu" in Path(b).name else "CPU"
    unit = {"prediction": "per prediction", "iteration": "per iteration",
            "raw": "total"}[per]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(diff_flamegraph_svg(
        folded_a, folded_b,
        f"{kind} self time {unit}: {label_a} → {label_b}",
        label_a, label_b, width_from,
    ))
    return out


def main(argv=None):
    args = parse_args(argv)
    try:
        out = write_diff(args.a, args.b, args.out, args.per, args.width_from,
                         args.label_a, args.label_b)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
