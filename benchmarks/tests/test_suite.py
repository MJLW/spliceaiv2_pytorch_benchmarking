from spliceai2_bench.mutation_benchmark import (
    BASES,
    apply_substitution,
    random_substitutions,
)
from spliceai2_bench.suite import benchmarks, markdown, parse_args, summarize
from spliceai2_ism.encode import random_sequences

CROP = 64


def test_consecutive_substitutions():
    seq = random_sequences(1, 1000 + CROP, seed=0)[0]
    pos, alt = random_substitutions(seq, CROP, 1000, 50, seed=1,
                                    layout="consecutive")
    assert pos == list(range(pos[0], pos[0] + 50))
    assert 0 <= pos[0] and pos[-1] < 1000
    for p, a in zip(pos, alt):
        assert BASES[a] != seq[CROP // 2 + p]
        variant = apply_substitution(seq, CROP, p, a)
        assert variant[CROP // 2 + p] == BASES[a] and len(variant) == len(seq)


def test_random_substitutions_distinct():
    seq = random_sequences(1, 1000 + CROP, seed=0)[0]
    pos, _ = random_substitutions(seq, CROP, 1000, 200, seed=1)
    assert len(set(pos)) == 200 and max(pos) < 1000


def test_suite_runs_all_four_benchmarks():
    runs = benchmarks(parse_args([]))
    names = [name for name, _, _ in runs]
    assert names == ["bench_batch1", "bench_batch8",
                     "mutations_random", "mutations_consecutive"]
    for name, _, args in runs:
        if name.startswith("bench"):
            assert args[args.index("--n-seqs") + 1] == "500"
        else:
            assert args[args.index("--n-mutations") + 1] == "499"


def _stats(x):
    return {"median": x, "min": x, "max": x, "all": [x]}


def test_summary_of_mutation_run():
    telemetry = {"temp_c": 50, "sm_clock_mhz": 1980, "power_w": 500}
    stage = {"gpu_s": {k: _stats(1.0) for k in ("1d", "2d", "total")},
             "end_to_end_s": _stats(1.1)}
    r = {
        "config": {"n_mutations": 499, "layout": "random",
                   "variant_batch_size": 100},
        "accuracy": {"full_pass": {"max_dp_1d": 1e-4, "max_dp_2d": None,
                                   "mean_topk_overlap": 0.999}},
        "workloads": {
            "full": {**stage, "peak_gpu_mem_bytes": 2**30},
            "incremental": {"reference": stage, "variants": stage,
                            "total": stage, "peak_gpu_mem_bytes": 2**31},
        },
        "speedup": {"gpu": _stats(20.0), "e2e": _stats(18.0)},
        "passes": [{"telemetry_start": telemetry, "telemetry_end": telemetry}],
    }
    s = summarize("mutations_random", r)
    assert s["n"] == 500 and s["peak_gpu_gib"] == 2.0
    md = markdown({"mutations_random": {"exit_code": 0, "wall_s": 60,
                                        "summary": s}},
                  parse_args([]), {"gpu": "H100"})
    assert "| random |" in md and "20.0×" in md
