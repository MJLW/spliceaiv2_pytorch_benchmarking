import pytest
import torch

from spliceai2_bench.ablation import (
    bench_args,
    configs,
    gains,
    markdown,
    parse_args,
    row,
)
from spliceai2_bench.mutation_benchmark import deltas_to_host
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
    args = parse_args(["--configs", f"none,all,only_{name},naive,all_deltas"])
    md = markdown(rows, {"all_deltas": "failed"}, args, {"gpu": "test"})
    assert "| all_deltas | failed |" in md
    assert "naive 25.0, 20.0×" in md


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
