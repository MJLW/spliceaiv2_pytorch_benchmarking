import json

import pytest
import torch

from spliceai2_bench import ablation
from spliceai2_bench.ablation import (
    bench_args,
    configs,
    gains,
    length_label,
    markdown,
    parse_args,
    row,
)
from spliceai2_bench.mutation_benchmark import deltas_to_host, make_emit
from spliceai2_bench.mutation_benchmark import parse_args as mutbench_args
from spliceai2_ism.incremental import Optimizations


def _stats(v):
    return {"median": v, "min": v, "max": v, "all": [v]}


def _acc():
    return {"n": 499, "max_dp_1d": 1e-4, "max_dp_2d": 2e-3,
            "topk_identical": 400, "mean_topk_overlap": 0.999}


def fake_result(per_variant_ms, naive=False):
    n_var = 499
    var_s = per_variant_ms * n_var / 1e3

    def stage(gpu_s, n):
        return {"n_predictions": n,
                "gpu_s": {"1d": _stats(gpu_s * 0.8), "2d": _stats(gpu_s * 0.2),
                          "total": _stats(gpu_s)},
                "end_to_end_s": _stats(gpu_s * 1.2),
                "end_to_end_phases_s": {"load": _stats(0.01),
                                        "run": _stats(gpu_s),
                                        "to_host": _stats(gpu_s * 0.2)},
                "predictions_per_s": {"gpu": n / gpu_s,
                                      "e2e": n / gpu_s / 1.2}}

    r = {
        "config": {"n_mutations": n_var, "opts": "all"},
        "env": {"gpu": "test", "torch": "x"},
        "warmup_s": {"full batch of 8": {"1d": 10, "2d": 3},
                     "reference pass": {"1d": 5, "2d": 3},
                     "variant batch of 32": {"1d": 50, "2d": 90}},
        "accuracy": {"full_pass": _acc(),
                     "vs_unoptimized": {"reference": _acc(),
                                        "variants": _acc()}},
        "workloads": {"incremental": {
            "reference": stage(0.039, 1),
            "variants": stage(var_s, n_var),
            "total": stage(var_s + 0.039, n_var + 1),
            "peak_gpu_mem_bytes": 5 * 2**30,
        }},
    }
    if naive:
        r["workloads"]["full"] = {"predictions_per_s": {"gpu": 25.0}}
        r["speedup"] = {"gpu": _stats(20.0)}
    return r


def test_configs_cover_every_optimization():
    c = configs()
    for name in Optimizations.names():
        assert c[f"only_{name}"] == ["--opts", name]
        assert c[f"all_minus_{name}"] == ["--opts", f"all,-{name}"]
    assert c["all_no_output"] == ["--opts", "all", "--e2e-output", "none"]
    assert {"none", "all", "naive", "all_deltas", "all_freezing",
            "all_reduce_overhead", "all_max_autotune"} <= c.keys()


def test_bench_args_time_naive_only_in_its_config():
    args = parse_args([])
    assert "--only" in bench_args(args, "all")
    assert "--only" not in bench_args(args, "naive")
    assert bench_args(args, "all_freezing")[-1] == "--freezing"
    with pytest.raises(SystemExit):
        parse_args(["--configs", "all,bogus"])


def test_row_and_gains():
    r = row(fake_result(2.5))
    assert r["per_variant_ms"] == pytest.approx(2.5)
    assert r["compile_s"] == pytest.approx(148)  # reference + variant batch
    assert r["vs_unoptimized"]["max_dp_2d"] == 2e-3
    name = Optimizations.names()[0]
    rows = {"none": row(fake_result(2.5)), "all": row(fake_result(1.0)),
            f"only_{name}": row(fake_result(2.0)),
            f"all_minus_{name}": row(fake_result(1.5)),
            "naive": row(fake_result(1.0, naive=True))}
    g = gains(rows)
    assert g[name]["alone"] == pytest.approx(1.25)
    assert g[name]["on_top"] == pytest.approx(1.5)
    assert g["all"]["alone"] == pytest.approx(2.5)
    assert g[Optimizations.names()[1]]["alone"] is None
    args = parse_args(["--configs", f"none,all,only_{name},naive,all_deltas",
                       "--seq-lens", "10000,100000"])
    results = {
        "10kb": {"seq_len": 10_000, "rows": rows,
                 "status": {"all_deltas": "failed"}},
        "100kb": {"seq_len": 100_000, "rows": {"none": rows["none"]},
                  "status": {}},
    }
    md = markdown(results, args, {"gpu": "test"})
    assert "| all_deltas | failed |" in md
    assert "naive 25.0, 20.0×" in md
    # one column per length; runs missing at a length show as "–"
    assert "| Optimization | 10kb | 100kb |" in md
    assert f"| {name} | 1.25× / 1.50× | – / – |" in md
    assert "| ms/variant, `all` | 1.00 | – |" in md
    assert "## 10kb: 10,000 positions" in md
    assert "## 100kb: 100,000 positions" in md


def test_seq_lens():
    assert parse_args([]).seq_lens == [10_000, 20_000, 50_000, 100_000]
    args = parse_args(["--seq-lens", "16384"])  # one length
    assert args.seq_lens == [16_384]
    assert bench_args(args, "all")[:2] == ["--seq-len", "16384"]
    args = parse_args(["--seq-lens", "20000,10000"])
    assert args.seq_lens == [20_000, 10_000]  # run in the given order
    assert bench_args(args, "all", 10_000)[:2] == ["--seq-len", "10000"]
    assert length_label(10_000) == "10kb"
    assert length_label(16_384) == "16384nt"
    with pytest.raises(SystemExit):  # no --seq-len, not even as a prefix
        parse_args(["--seq-len", "16384"])
    for bad in ("", "0", "10000,10000", "10k"):
        with pytest.raises(SystemExit):
            parse_args(["--seq-lens", bad])
    with pytest.raises(SystemExit):  # more mutations than positions
        parse_args(["--seq-lens", "400,10000"])


def test_main_runs_every_config_per_length(tmp_path, monkeypatch):
    calls = []

    def fake_run(name, module, args, out_dir, env):
        calls.append((name, args[args.index("--seq-len") + 1]))
        label, cfg = name.split("/")
        ms = {"none": 2.5, "all": 1.0}[cfg] * (1 if label == "10kb" else 2)
        (out_dir / f"{name}.json").write_text(
            json.dumps(fake_result(ms)))
        return {"exit_code": 0, "wall_s": 1.0}

    monkeypatch.setattr(ablation, "run", fake_run)
    with pytest.raises(SystemExit) as e:
        ablation.main(["--configs", "none,all", "--seq-lens", "10000,20000",
                       "--out-dir", str(tmp_path)])
    assert e.value.code == 0
    assert calls == [("10kb/none", "10000"), ("10kb/all", "10000"),
                     ("20kb/none", "20000"), ("20kb/all", "20000")]
    out = json.loads((tmp_path / "ablation.json").read_text())
    assert list(out["lengths"]) == ["10kb", "20kb"]
    assert out["lengths"]["20kb"]["gains"]["all"]["alone"] == \
        pytest.approx(2.5)
    assert set(out["runs"]) == {c for c, _ in calls}
    assert (tmp_path / "10kb" / "none.json").exists()


class _Workload:
    def __init__(self, ref_out, n, rows):
        self.ref_out, self.n, self.rows = ref_out, n, rows

    def predictions(self, i):
        return self.n

    def reference_rows(self, i):
        return self.rows


def test_deltas_to_host_keeps_only_changes():
    torch.manual_seed(0)
    k, length = 4, 10
    # three predictions equal to the reference, then two small edits
    ss = torch.randn(1, 2, length).repeat(3, 1, 1)
    idxs = torch.tensor([[1, 3, 5, 7]] * 3)
    jxn = torch.randn(1, k, k).repeat(3, 1, 1)
    ss[1, 0, 6] += 3.0  # one prediction changes at one position
    w = _Workload((ss[:1].clone(), idxs[0].clone(), jxn[0].clone()), 3, [0])
    jxn[2, 0, 2] += 3.0  # another changes one junction
    host = deltas_to_host(w, 0, (ss, idxs, jxn), min_delta=1e-3)
    # the reference row as full probabilities, then the sparse deltas
    probs = host[0][0]
    assert probs.shape == (2, length)
    row_, pos, d_donor, _, snv2, donor, acceptor, d_jxn = host[-1]
    assert row_.tolist() == [1] and pos.tolist() == [6]
    assert d_donor[0] > 0
    assert snv2.tolist() == [2]
    assert (donor.item(), acceptor.item()) == (1, 5)
    assert d_jxn[0] > 0


def test_no_output_emits_nothing():
    args = mutbench_args(["--e2e-output", "none"])
    assert make_emit(args)(None, 0, ("ss", "idxs", "jxn")) is None


def _with_output(r, e2e, to_host):
    """A row with end-to-end throughput ``e2e`` and to_host seconds."""
    r = dict(r)
    r["e2e_predictions_per_s"] = e2e
    r["e2e_phases_s"] = {"load": 0.01, "run": 0.49, "to_host": to_host}
    return r


def test_output_cost_and_section():
    base = row(fake_result(1.0))
    rows = {"all_no_output": _with_output(base, 1000.0, 0.0),
            "all_deltas": _with_output(base, 980.0, 0.04),
            "all": _with_output(base, 900.0, 0.1),
            "all_pageable": _with_output(base, 600.0, 0.5)}
    c = ablation.output_cost(rows)
    m = c["modes"]
    assert m["all_no_output"]["relative_to_no_output"] == pytest.approx(1.0)
    assert m["all"]["relative_to_no_output"] == pytest.approx(0.9)
    assert m["all_pageable"]["to_host_share"] == pytest.approx(0.5)
    assert m["all_deltas"]["to_host_s"] == pytest.approx(0.04)
    cmp = c["all_vs_no_output"]
    assert cmp["e2e_predictions_per_s"]["ratio"] == pytest.approx(0.9)
    assert cmp["predictions_per_s"]["ratio"] == pytest.approx(1.0)
    assert cmp["to_host"]["ratio"] is None  # no output: 0 s, no ratio
    # a length without these runs
    assert ablation.output_cost({})["modes"]["all"] is None

    args = parse_args(["--configs", "all,all_no_output,all_deltas,"
                       "all_pageable", "--seq-lens", "10000,20000"])
    md = markdown({"10kb": {"seq_len": 10_000, "rows": rows, "status": {}},
                   "20kb": {"seq_len": 20_000, "rows": {}, "status": {}}},
                  args, {"gpu": "test"})
    assert "## Output cost" in md
    assert "| e2e pred/s, no output | 1000.0 | – |" in md
    assert "| e2e pred/s, sparse deltas | 980.0 (0.98×) | – |" in md
    assert "| e2e pred/s, full, pinned | 900.0 (0.90×) | – |" in md
    assert "| to_host, full, pageable | 0.500 s (50%) | – |" in md
    assert "### `all` vs `all_no_output`" in md
    assert "| e2e pred/s | 900.0 / 1000.0 (0.90×) | – |" in md
    assert "| e2e to_host, s | 0.100 / 0.000 | – |" in md
