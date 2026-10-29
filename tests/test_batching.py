from spliceai2_ism.batching import fit_batch_size

FIXED, PER_SNV = 1_000, 100


def peaks(*sizes):
    return {b: FIXED + PER_SNV * b for b in sizes}


def test_fits_linear_model_and_rounds_to_multiple():
    # 0.8 × 10,000 = 8,000 bytes: (8,000 − 1,000) / 100 = 70 → 64
    batch, fixed, per_snv = fit_batch_size(peaks(1, 5), capacity=10_000)
    assert (batch, fixed, per_snv) == (64, FIXED, PER_SNV)


def test_small_batches_are_not_rounded():
    # 0.8 × 1,900 = 1,520: (1,520 − 1,000) / 100 = 5
    assert fit_batch_size(peaks(1, 5), capacity=1_900)[0] == 5


def test_capped():
    assert fit_batch_size(peaks(1, 5), capacity=10**12)[0] == 256


def test_at_least_one_even_if_nothing_fits():
    assert fit_batch_size(peaks(1, 5), capacity=500)[0] == 1


def test_no_measurable_growth_means_the_cap():
    assert fit_batch_size({1: 5_000, 5: 5_000}, capacity=10**6)[0] == 256
