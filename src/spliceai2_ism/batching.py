"""Chooses the SNV batch size from the free GPU memory.

Peak memory of an SNV batch is about linear in its size: a fixed part (the
models, the haplotype's cache and buffers that do not grow with the batch)
plus a cost per SNV. Both are measured by running two small batches eagerly
on the longest gene of the run, which needs the most memory: the cache and
part of the per-SNV cost grow with gene length. The batch size is then the
largest that fits in a fraction of the GPU memory, rounded down and capped,
since throughput stops improving well before memory runs out.

Compiled stages usually need less memory than eager ones; for the cases
where they do not, ``ism.run_haplotype`` halves the batch size and retries
when a batch runs out of memory.
"""
import numpy as np
import torch

from spliceai2_ism.deltas import delta_1d, delta_2d
from spliceai2_ism.encode import one_hot_codes
from spliceai2_ism.runtime import inference
from spliceai2_ism.sequence import haplotype_window, saturation

MEMORY_FRACTION = 0.8
MAX_BATCH_SIZE = 256
MULTIPLE = 8
PROBE_SIZES = (1, 5)


def fit_batch_size(peaks, capacity, fraction=MEMORY_FRACTION,
                   cap=MAX_BATCH_SIZE, multiple=MULTIPLE):
    """Largest batch size whose extrapolated peak fits in ``fraction`` of
    ``capacity`` bytes. ``peaks``: {batch size: peak bytes} for two sizes.
    Returns (batch size, fixed bytes, bytes per SNV)."""
    (b1, p1), (b2, p2) = sorted(peaks.items())
    per_snv = max((p2 - p1) / (b2 - b1), 1.0)
    fixed = p1 - per_snv * b1
    n = int((fraction * capacity - fixed) // per_snv)
    n = min(n, cap)
    if n >= multiple:
        n -= n % multiple
    return max(n, 1), fixed, per_snv


def measure_peaks(inc, stages, region, device, sizes=PROBE_SIZES):
    """Peak allocated bytes of one SNV batch per size in ``sizes``, on the
    gene's reference sequence, with the haplotype cache resident.
    ``stages``: eager ``ism.Stages``."""
    window = haplotype_window(region, ())
    gene_len = window.gene_len
    x = torch.from_numpy(one_hot_codes(window.codes_t))[None].to(device)
    valid_len = torch.tensor([max(gene_len, inc.out_2d_len)], device=device)
    snv_pos, snv_alt = saturation(window)
    peaks = {}
    with inference(device):
        cache, hap_ss, hap_idx, hap_jxn = stages.reference(x, valid_len)
        hap_scores = hap_ss.amax(dim=1, keepdim=True)
        for b in sizes:
            take = np.arange(b) % len(snv_pos)
            pos = torch.from_numpy(snv_pos[take]).to(device)
            alt = torch.from_numpy(snv_alt[take]).to(device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            win, v_idx, v_jxn = stages.variants(cache, hap_scores, pos, alt,
                                                valid_len)
            r1 = delta_1d(inc, hap_ss, pos, win, gene_len, 0.0)
            r2 = delta_2d(hap_idx[0], hap_jxn[0], v_idx, v_jxn, 0.0,
                          gene_len=gene_len)
            torch.cuda.synchronize(device)
            peaks[b] = torch.cuda.max_memory_allocated(device)
            del win, v_idx, v_jxn, r1, r2
    del cache, hap_ss, hap_idx, hap_jxn, hap_scores, x
    torch.cuda.empty_cache()
    return peaks


def choose_batch_size(inc, stages, regions, device):
    """Batch size for the run, and what it was based on."""
    region = max(regions, key=lambda r: (max(r.out_lens), r.gene.length))
    peaks = measure_peaks(inc, stages, region, device)
    free, total = torch.cuda.mem_get_info(device)
    # memory this process may use: what is free, plus what it holds already
    capacity = free + torch.cuda.memory_reserved(device)
    batch, fixed, per_snv = fit_batch_size(peaks, capacity)
    return batch, {
        "measured_on": region.gene.name,
        "peak_bytes": {str(b): p for b, p in peaks.items()},
        "fixed_bytes": fixed,
        "bytes_per_snv": per_snv,
        "capacity_bytes": capacity,
        "total_bytes": total,
        "fraction": MEMORY_FRACTION,
        "cap": MAX_BATCH_SIZE,
    }
