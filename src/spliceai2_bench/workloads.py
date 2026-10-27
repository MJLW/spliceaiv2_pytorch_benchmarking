"""Workload generation and interleaved scheduling."""
import numpy as np


def normal_lengths(n, mean, std, crop_len, multiple, minimum=1, seed=0):
    """Output lengths ~ N(mean, std), rounded so that length + crop_len is a
    multiple of ``multiple`` (the model's deepest pooling factor) and clipped
    to the smallest such length >= ``minimum``."""
    rng = np.random.default_rng(seed)
    in_lens = np.rint((rng.normal(mean, std, n) + crop_len) / multiple)
    lengths = in_lens.astype(np.int64) * multiple - crop_len
    lowest = -(-(minimum + crop_len) // multiple) * multiple - crop_len
    return np.maximum(lengths, lowest).tolist()


def bucket_lengths(lengths, bucket_size):
    """Rounds each length up to a multiple of ``bucket_size``."""
    return [-(-n // bucket_size) * bucket_size for n in lengths]


def interleave(num_batches):
    """Spreads the batches of several workloads evenly over one pass, so slow
    drifts (e.g. thermal throttling) affect every workload alike.

    Returns (workload_index, batch_index) pairs in execution order.
    """
    slots = [
        ((i + 0.5) / n, k, i)
        for k, n in enumerate(num_batches)
        for i in range(n)
    ]
    return [(k, i) for _, k, i in sorted(slots)]
