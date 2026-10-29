import numpy as np

from spliceai2_ism.encode import (
    N_CODE,
    PAD_CODE,
    one_hot_codes,
    one_hot_encode,
    random_sequences,
)


def test_one_hot_layout():
    x = one_hot_encode("ACGTN", "GRCh38")
    assert x.shape == (14, 5)
    assert x.dtype == np.uint8
    expected_nuc = np.array([
        [1, 0, 0, 0, 1],  # A (N falls back to A, as in the reference)
        [0, 1, 0, 0, 0],  # C
        [0, 0, 1, 0, 0],  # G
        [0, 0, 0, 1, 0],  # T
    ])
    np.testing.assert_array_equal(x[:4], expected_nuc)
    np.testing.assert_array_equal(x[4], 1)
    np.testing.assert_array_equal(x[5:], 0)


def test_assembly_channel():
    x = one_hot_encode("AC", "GRCm38")
    np.testing.assert_array_equal(x[12], 1)
    assert x[4].sum() == 0
    # unknown assemblies fall back to GRCh38
    np.testing.assert_array_equal(one_hot_encode("AC", "hg19")[4], 1)


def test_padding():
    x = one_hot_encode("ACG", "GRCh38", length=6)
    assert x.shape == (14, 6)
    np.testing.assert_array_equal(x[:, :3], one_hot_encode("ACG", "GRCh38"))
    np.testing.assert_array_equal(x[:4, 3:], 0)  # padding: no nucleotide
    np.testing.assert_array_equal(x[4], 1)


def test_one_hot_codes_matches_string_encoder():
    codes = np.array([0, 1, 2, 3, N_CODE, PAD_CODE, PAD_CODE], dtype=np.uint8)
    np.testing.assert_array_equal(
        one_hot_codes(codes), one_hot_encode("ACGTN", length=7)
    )


def test_random_sequences_deterministic():
    a = random_sequences(3, 50, seed=1)
    b = random_sequences(3, 50, seed=1)
    assert a == b
    assert all(len(s) == 50 and set(s) <= set("ACGT") for s in a)
