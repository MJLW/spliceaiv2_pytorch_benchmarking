from types import SimpleNamespace

import pytest
from torch.autograd import DeviceType

from spliceai2_bench.mutation_benchmark import (
    OOM_EXIT,
    full_batches,
    parse_args as mutbench_args,
)
from spliceai2_bench.profiling import (
    diff_flamegraph_svg,
    flamegraph_svg,
    fold_stacks,
    summarize_events,
    write_folded,
)
from spliceai2_bench.sweep import (
    axis_args,
    axis_limit,
    axis_summary,
    markdown,
    parse_args,
    row,
    status_of,
    sweep_axis,
)
from spliceai2_ism.batching import fit_batch_size


def test_full_batches_pad_last_batch():
    assert full_batches(5, 2) == [([0, 1], 2), ([2, 3], 2), ([4, 4], 1)]
    assert full_batches(4, 4) == [([0, 1, 2, 3], 4)]
    assert full_batches(3, 8) == [([0, 1, 2, 2, 2, 2, 2, 2], 3)]


def test_mutbench_batch_size_defaults():
    a = mutbench_args([])
    assert a.full_batch_size == "auto" and a.variant_batch_size == 32
    a = mutbench_args(["--full-batch-size", "16", "--variant-batch-size", "64"])
    assert a.full_batch_size == 16 and a.variant_batch_size == 64
    assert mutbench_args(["--variant-batch-size", "auto"]).variant_batch_size \
        == "auto"
    with pytest.raises(SystemExit):
        mutbench_args(["--full-batch-size", "0"])


def test_mutbench_opts_and_compile_flags():
    from spliceai2_ism.incremental import Optimizations
    a = mutbench_args([])
    assert a.opts == Optimizations() and a.compile_mode == "default"
    assert not a.freezing and a.e2e_output == "full"
    a = mutbench_args(["--opts", "all,-stencil_resample",
                       "--compile-mode", "reduce-overhead", "--freezing",
                       "--e2e-output", "deltas"])
    assert not a.opts.stencil_resample and a.opts.padded_ffn
    assert a.compile_mode == "reduce-overhead" and a.freezing
    assert a.e2e_output == "deltas"
    with pytest.raises(SystemExit):
        mutbench_args(["--opts", "all,-nonsense"])


def test_fit_batch_size_without_cap():
    # 1 GiB fixed, 1 GiB per item, 100 GiB available
    peaks = {1: 2 * 2**30, 2: 3 * 2**30}
    assert fit_batch_size(peaks, 100 * 2**30, 1.0, cap=None, multiple=1)[0] == 99
    assert fit_batch_size(peaks, 100 * 2**30, 0.8, cap=None)[0] == 72
    assert fit_batch_size(peaks, 100 * 2**30, 1.0, cap=16)[0] == 16


def test_fit_probe_with_missing_peaks():
    from spliceai2_bench.mutation_benchmark import fit_probe
    gib = 2**30
    # both probe sizes fit: the usual two-point fit
    assert fit_probe({1: 2 * gib, 2: 3 * gib}, gib, 10 * gib, 1.0, 1)[0] == 9
    # batch 2 ran out of memory: models + cache stand in for batch 0
    n, fixed, per = fit_probe({1: 4 * gib}, 2 * gib, 5.5 * gib, 1.0, 1)
    assert (n, fixed, per) == (1, 2 * gib, 2 * gib)
    # nothing fit: still try batch 1
    assert fit_probe({}, 2 * gib, 5.5 * gib, 1.0, 1) == (1, None, None)


def fake_runner(oom_from, ceiling):
    calls = []

    def run_at(b):
        calls.append(b)
        if b >= oom_from:
            return "oom", None
        return "ok", {"batch_sizing": {"ceiling": {"full": ceiling}}}
    return run_at, calls


def test_sweep_doubles_until_oom_then_tries_ceiling():
    run_at, calls = fake_runner(oom_from=26, ceiling=25)
    runs = sweep_axis(run_at, "full", 1, 500)
    assert calls == [1, 2, 4, 8, 16, 32, 25]
    assert [s for _, s, _ in runs] == ["ok"] * 5 + ["oom", "ok"]


def test_sweep_ceiling_capped_below_failed_size():
    run_at, calls = fake_runner(oom_from=32, ceiling=40)
    sweep_axis(run_at, "full", 1, 500)
    assert calls == [1, 2, 4, 8, 16, 32, 31]


def test_sweep_skips_ceiling_at_or_below_last_fit():
    run_at, calls = fake_runner(oom_from=32, ceiling=12)
    sweep_axis(run_at, "full", 1, 500)
    assert calls == [1, 2, 4, 8, 16, 32]


def test_sweep_stops_at_item_limit():
    run_at, calls = fake_runner(oom_from=10**6, ceiling=10**6)
    sweep_axis(run_at, "full", 1, 20)
    assert calls == [1, 2, 4, 8, 16, 20]


def test_sweep_stops_on_failure_and_first_oom():
    calls = []

    def run_at(b):
        calls.append(b)
        return ("failed" if b == 4 else "ok"), {}
    assert [s for _, s, _ in sweep_axis(run_at, "full", 1, 500)] == [
        "ok", "ok", "failed"]
    run_at, calls = fake_runner(oom_from=1, ceiling=8)
    sweep_axis(run_at, "full", 1, 500)
    assert calls == [1]


def test_axis_arguments_and_status():
    assert axis_args("naive", 8) == ["--full-batch-size", "8",
                                     "--variant-batch-size", "auto"]
    assert "--only" in axis_args("incremental", 64)
    assert axis_limit("naive", 499) == 500
    assert axis_limit("incremental", 499) == 499
    assert [status_of(c) for c in (0, OOM_EXIT, 1)] == ["ok", "oom", "failed"]
    args = parse_args([])
    assert args.layouts == ["random", "consecutive"]
    assert args.axes == ["naive", "incremental"]


def _stats(x):
    return {"median": x, "min": x, "max": x, "all": [x]}


def _stage(gpu_s, n):
    return {"n_predictions": n,
            "gpu_s": {k: _stats(gpu_s) for k in ("1d", "2d", "total")},
            "end_to_end_s": _stats(gpu_s * 1.1),
            "predictions_per_s": {"gpu": n / gpu_s, "e2e": n / gpu_s / 1.1}}


def _result(axis, b, gpu_s):
    telemetry = {"temp_c": 50, "sm_clock_mhz": 1980, "power_w": 600}
    workloads = {"incremental": {
        "reference": _stage(0.1, 1), "variants": _stage(gpu_s - 0.1, 499),
        "total": _stage(gpu_s, 500), "batch_size": 128,
        "peak_gpu_mem_bytes": 2**34,
    }}
    if axis == "naive":
        workloads["full"] = {**_stage(gpu_s, 500), "batch_size": b,
                             "batches": 1, "peak_gpu_mem_bytes": 2**35}
    return {
        "config": {"n_mutations": 499},
        "workloads": workloads,
        "speedup": {"gpu": _stats(30.0)} if axis == "naive" else None,
        "batch_sizing": {"ceiling": {"full": 25, "variant": 300}},
        "passes": [{"telemetry_start": telemetry,
                    "telemetry_end": telemetry}],
    }


def test_markdown_compares_axes_and_layouts():
    sweep = {
        layout: {
            "naive": [row("naive", 1, "ok", _result("naive", 1, 100.0)),
                      row("naive", 2, "ok", _result("naive", 2, 80.0)),
                      row("naive", 4, "oom", None)],
            "incremental": [row("incremental", 64, "ok",
                                _result("incremental", 64, 4.0)),
                            row("incremental", 128, "ok",
                                _result("incremental", 128, scale)),
                            row("incremental", 256, "oom", None)],
        }
        for layout, scale in (("random", 2.0), ("consecutive", 2.5))
    }
    s = axis_summary(sweep["random"]["incremental"])
    assert s["best"]["batch_size"] == 128
    assert s["max_fit"] == 128 and s["first_oom"] == 256
    md = markdown(sweep, parse_args([]), {"gpu": "H200"})
    assert "## Best per layout" in md
    # best naive 500 / 80 s, best incremental 500 / 2 s: 40×
    assert "| random | 6.2 pred/s (2) | 250.0 pred/s (128) | 40.0× |" in md
    assert "## Random vs consecutive (incremental)" in md
    assert "## consecutive, naive" in md and "| 4 | OOM |" in md


def test_profile_summary_counts_gpu_kernels_only():
    def ev(name, dev, us, count=1, cpu=0.0, annotation=False):
        return SimpleNamespace(key=name, device_type=dev, count=count,
                               self_device_time_total=us,
                               self_cpu_time_total=cpu,
                               is_user_annotation=annotation)
    events = [
        ev("aten::conv1d", DeviceType.CPU, 600.0, cpu=50.0),  # its kernels' time
        ev("cudnn_conv", DeviceType.CUDA, 600.0, count=4),
        ev("triton_fused", DeviceType.CUDA, 200.0, count=8),
        # GPU-side annotation ranges overlap the kernels: not counted
        ev("ProfilerStep*", DeviceType.CUDA, 800.0, annotation=True),
        ev("## Call CompiledFxGraph", DeviceType.CUDA, 700.0, annotation=True),
    ]
    s = summarize_events(events, iters=2)
    assert s["device_ms_per_iter"] == pytest.approx(0.4)
    assert s["cpu_ms_per_iter"] == pytest.approx(0.025)
    assert s["kernel_launches_per_iter"] == 6
    assert [k["name"] for k in s["top_kernels"]] == ["cudnn_conv",
                                                    "triton_fused"]
    assert s["top_kernels"][0]["share"] == pytest.approx(0.75)


def test_fold_stacks_and_flamegraph(tmp_path):
    def ev(name, stack, us, parent=None, dev=DeviceType.CPU):
        # torch lists the innermost frame first
        return SimpleNamespace(name=name, stack=stack, t=us,
                               cpu_parent=parent, device_type=dev)
    graph = ev("## Call CompiledFxGraph", ["bench.py(7): stage",
                                           "bench.py(5): run"], 0)
    events = [
        ev("aten::conv1d", ["model.py(10): block", "bench.py(5): run"], 300),
        ev("aten::conv1d", ["model.py(10): block", "bench.py(5): run"], 100),
        ev("aten::add", ["model.py(12): block;x", "bench.py(5): run"], 50),
        graph,
        # launches carry the GPU time but no stack: they take their parent's
        ev("cuLaunchKernel", None, 30, parent=graph),
        ev("cuLaunchKernel", None, 20, parent=graph),
        ev("aten::empty", [], 5),         # no stack, no parent: names only
        ev("aten::view", ["bench.py(5): run"], 0),  # no time: skipped
        ev("triton_k", None, 50, dev=DeviceType.CUDA),  # GPU side: skipped
    ]
    folded = fold_stacks(events, lambda e: e.t)
    assert folded == {
        "bench.py(5): run;model.py(10): block;aten::conv1d": 400,
        "bench.py(5): run;model.py(12): block:x;aten::add": 50,
        "bench.py(5): run;bench.py(7): stage;## Call CompiledFxGraph;"
        "cuLaunchKernel": 50,
        "aten::empty": 5,
    }
    del folded["bench.py(5): run;bench.py(7): stage;## Call CompiledFxGraph;"
               "cuLaunchKernel"], folded["aten::empty"]
    path = tmp_path / "stacks.txt"
    write_folded(folded, path)
    assert path.read_text().splitlines()[0].endswith(" 400")

    import xml.etree.ElementTree as ET
    svg = flamegraph_svg(folded, "test <gpu>")
    root = ET.fromstring(svg)  # well-formed, names escaped
    ns = "{http://www.w3.org/2000/svg}"
    titles = [t.text for t in root.iter(f"{ns}title")]
    assert "all — 450.0 µs (100.0%)" in titles
    assert "aten::conv1d — 400.0 µs (88.9%)" in titles
    ET.fromstring(flamegraph_svg({}, "empty"))  # no stacks: still valid


def test_normalize_frame_drops_run_specific_parts():
    from spliceai2_bench.profiling import normalize_frame
    assert normalize_frame(
        "<built-in method to of Tensor object at 0x76e7d0a69360>"
    ) == "<built-in method to of Tensor object>"
    assert normalize_frame(
        "## Call CompiledFxGraph fyyenzygph36wxjyfxbxzgkj4gyt4ccwjcrh3 ##"
    ) == "## Call CompiledFxGraph ##"
    assert normalize_frame(
        "/tmp/torchinductor_mattijn/2y/c2y6khulrtmn6nbqzz.py(3581): call"
    ) == "<inductor code>: call"
    # ordinary frames are kept, line numbers included
    assert normalize_frame("spliceai2_bench/mutation_benchmark.py(222): run") \
        == "spliceai2_bench/mutation_benchmark.py(222): run"
    assert normalize_frame("a;b") == "a:b"


def test_diff_flamegraph(tmp_path):
    import xml.etree.ElementTree as ET
    a = {"run;conv": 400.0, "run;add": 100.0}
    b = {"run;conv": 200.0, "run;topk": 50.0}
    root = ET.fromstring(diff_flamegraph_svg(a, b, "t", "A", "B"))
    ns = "{http://www.w3.org/2000/svg}"
    groups = {g.find(f"{ns}title").text.split(" — ")[0]: g
              for g in root.iter(f"{ns}g")}
    # widths follow B: add (only in A) is not drawn
    assert set(groups) == {"all", "run", "conv", "topk"}
    tip = groups["conv"].find(f"{ns}title").text
    assert "A 400.0 µs, B 200.0 µs, Δ -200.0 µs (-50.0%)" in tip
    assert "[only in B]" in groups["topk"].find(f"{ns}title").text
    assert "hsl(215" in groups["conv"].find(f"{ns}rect").get("fill")  # faster
    assert "hsl(0," in groups["topk"].find(f"{ns}rect").get("fill")   # slower
    # widths from A: the removed frame shows up
    root = ET.fromstring(diff_flamegraph_svg(a, b, "t", width_from="a"))
    names = {g.find(f"{ns}title").text.split(" — ")[0]
             for g in root.iter(f"{ns}g")}
    assert "add" in names and "topk" not in names


def test_flamediff_normalises_per_prediction(tmp_path):
    import json

    from spliceai2_bench.flamediff import default_label, scale, write_diff

    def run(name, preds, value):
        d = tmp_path / f"{name}.profile"
        d.mkdir()
        (d / "full.stacks_gpu.txt").write_text(f"run;conv {value}\n")
        (d / "full.meta.json").write_text(json.dumps(
            {"iterations": 3, "predictions_per_iter": preds}))
        return d / "full.stacks_gpu.txt"

    a = run("random_naive_b1", 3, 900)    # 3 iterations × 3 preds: 100 µs
    b = run("random_naive_b8", 24, 3600)  # 3 × 24: 50 µs per prediction
    assert scale(a, "prediction") == pytest.approx(1 / 9)
    assert scale(a, "iteration") == pytest.approx(1 / 3)
    assert default_label(b) == "random_naive_b8/full"
    out = write_diff(a, b, tmp_path / "d" / "diff.svg")
    svg = out.read_text()
    assert "100.0 µs" in svg and "50.0 µs" in svg and "per prediction" in svg
    (tmp_path / "random_naive_b1.profile" / "full.meta.json").unlink()
    with pytest.raises(ValueError):
        scale(a, "prediction")
    assert scale(a, "raw") == 1.0


def test_sweep_diff_jobs_and_kernel_table():
    from spliceai2_bench.sweep import diff_jobs, kernel_table

    def r(layout, axis, b, status="ok"):
        p = f"{layout}_{axis}_b{b}.profile"
        return {"batch_size": b, "status": status, "profile": {
            "full": {"stacks_gpu": f"{p}/full.stacks_gpu.txt",
                     "predictions_per_iter": 2 * b,
                     "top_kernels": [{"name": "conv",
                                      "device_ms_per_iter": 0.2 * b},
                                     {"name": f"k{b}",
                                      "device_ms_per_iter": 0.01}]}}}

    sweep = {layout: {"naive": [r(layout, "naive", 1), r(layout, "naive", 2),
                                r(layout, "naive", 4, "oom")]}
             for layout in ("random", "consecutive")}
    names = [n for _, _, n in diff_jobs(sweep)]
    assert names == [
        "random_naive_b2_vs_b1.full.diff_gpu.svg",
        "consecutive_naive_b2_vs_b1.full.diff_gpu.svg",
        "consecutive_vs_random_naive_b2.full.diff_gpu.svg",
    ]
    kernels, runs = kernel_table(sweep["random"]["naive"])["full"]
    assert kernels[0] == "conv"
    assert runs[1]["conv"] == pytest.approx(100.0)  # 0.2 ms / 2 predictions
    assert set(runs) == {1, 2}
