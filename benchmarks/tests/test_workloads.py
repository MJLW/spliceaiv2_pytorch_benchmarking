import numpy as np

from spliceai2_bench.workloads import bucket_lengths, interleave, normal_lengths
from spliceai2_ism.encode import random_sequences, random_sequences_of_lengths

CROP_LEN, MULTIPLE = 131_072, 16


def test_normal_lengths_valid_and_deterministic():
    lengths = normal_lengths(1000, 100_000, 10_000, CROP_LEN, MULTIPLE, seed=3)
    assert lengths == normal_lengths(1000, 100_000, 10_000, CROP_LEN, MULTIPLE,
                                     seed=3)
    assert all((n + CROP_LEN) % MULTIPLE == 0 for n in lengths)
    assert abs(np.mean(lengths) - 100_000) < 1_500
    assert 8_000 < np.std(lengths) < 12_000


def test_normal_lengths_clipped_to_minimum():
    lengths = normal_lengths(200, 0, 5_000, crop_len=100, multiple=16,
                             minimum=1024, seed=0)
    assert min(lengths) >= 1024
    assert all((n + 100) % 16 == 0 for n in lengths)


def test_bucket_lengths():
    assert bucket_lengths([1, 8192, 8193, 100_000], 8192) == [
        8192, 8192, 16384, 106_496
    ]


def test_interleave_spreads_workloads():
    assert interleave([2, 2]) == [(0, 0), (1, 0), (0, 1), (1, 1)]
    sched = interleave([1, 4])
    assert sorted(sched) == [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3)]
    assert sched.index((0, 0)) == 2  # the single batch lands mid-pass


def test_random_sequences_of_lengths():
    seqs = random_sequences_of_lengths([5, 10, 7], seed=2)
    assert [len(s) for s in seqs] == [5, 10, 7]
    # fixed-length helper is unchanged by the refactor
    assert random_sequences(2, 6, seed=4) == random_sequences_of_lengths(
        [6, 6], seed=4
    )
