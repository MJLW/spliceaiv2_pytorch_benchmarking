"""Haplotype predictions and sparse SNV deltas, as probabilities, computed
on the device. Positions are output indices (transcript orientation);
``output.py`` maps them to genomic coordinates.

Junctions are keyed by their (donor, acceptor) sites. An SNV can change the
top-k candidate sites, so a pair predicted for only one of haplotype and SNV
counts as probability 0 for the other. Genes shorter than the number of
candidate sites take the rest from the sequence after the gene; only pairs
with both sites in the gene are returned.
"""
import torch


def haplotype_rows(hap_ss, hap_idx, hap_jxn, gene_len):
    """1D probabilities (2, gene_len) and every donor < acceptor junction in
    the gene: (donor_idx, acceptor_idx, prob)."""
    probs = hap_ss[0, :, :gene_len].float().sigmoid()
    i, j = torch.triu_indices(len(hap_idx), len(hap_idx), 1,
                              device=hap_idx.device)
    keep = hap_idx[j] < gene_len  # sites are sorted: hap_idx[i] < hap_idx[j]
    i, j = i[keep], j[keep]
    return probs, hap_idx[i], hap_idx[j], hap_jxn.float().sigmoid()[i, j]


def delta_1d(inc, hap_ss, pos, win, gene_len, min_delta):
    """Splice-site deltas of each SNV's cone ``win`` (from ``variants_1d``)
    against the haplotype's logits ``hap_ss`` (1, 2, L), on the gene body.

    Returns (snv, out_idx, donor_delta, acceptor_delta) for the positions
    where either |delta| >= min_delta; outside the cones every delta is
    exactly zero.
    """
    o = inc.window_offsets(pos).unsqueeze(1) + torch.arange(
        win.shape[2], device=win.device
    )
    inside = (o >= 0) & (o < gene_len)
    hap = hap_ss[0][:, o.clamp(0, hap_ss.shape[2] - 1)].permute(1, 0, 2)
    d = win.float().sigmoid() - hap.float().sigmoid()
    keep = inside & (d.abs() >= min_delta).any(dim=1)
    snv, t = keep.nonzero(as_tuple=True)
    return snv, o[snv, t], d[snv, 0, t], d[snv, 1, t]


def delta_2d(hap_idx, hap_jxn, var_idx, var_jxn, min_delta, gene_len=None,
             chunk=None):
    """Junction deltas of B SNVs against the haplotype.

    hap_idx (K,) and var_idx (B, K) are sorted candidate sites, hap_jxn
    (K, K) and var_jxn (B, K, K) their logits. Returns (snv, donor_idx,
    acceptor_idx, delta) for |delta| >= min_delta: pairs of the SNV's sites
    (delta against the haplotype's probability, or 0 if the haplotype lacks
    the pair) and the haplotype's pairs the SNV lacks (delta = -prob).
    With ``gene_len``, only pairs with both sites below it.
    """
    k = len(hap_idx)
    chunk = chunk or max(1, 2 ** 26 // (k * k))
    upper = torch.ones(k, k, dtype=torch.bool, device=hap_idx.device).triu(1)
    p_h = hap_jxn.float().sigmoid()
    parts = []
    for s in range(0, len(var_idx), chunk):
        v_idx = var_idx[s:s + chunk]
        p_v = var_jxn[s:s + chunk].float().sigmoid()
        n = len(v_idx)

        # the SNV's pairs; sorted sites map to sorted sites, so i < j holds
        # in both
        j = torch.searchsorted(hap_idx, v_idx).clamp(max=k - 1)
        in_h = hap_idx[j] == v_idx
        both = in_h.unsqueeze(2) & in_h.unsqueeze(1)
        base = p_h[j.unsqueeze(2), j.unsqueeze(1)]
        d_v = p_v - torch.where(both, base, torch.zeros_like(base))
        b1, i1, j1 = (upper & (d_v.abs() >= min_delta)).nonzero(as_tuple=True)

        # the haplotype's pairs the SNV lacks
        jv = torch.searchsorted(v_idx, hap_idx.expand(n, k).contiguous())
        in_v = v_idx.gather(1, jv.clamp(max=k - 1)) == hap_idx
        lost = upper & ~(in_v.unsqueeze(2) & in_v.unsqueeze(1))
        b2, i2, j2 = (lost & (p_h >= min_delta)).nonzero(as_tuple=True)

        parts.append((
            torch.cat([b1, b2]) + s,
            torch.cat([v_idx[b1, i1], hap_idx[i2]]),
            torch.cat([v_idx[b1, j1], hap_idx[j2]]),
            torch.cat([d_v[b1, i1, j1], -p_h[i2, j2]]),
        ))
    snv, donor, acceptor, delta = (torch.cat(t) for t in zip(*parts))
    if gene_len is not None:
        keep = acceptor < gene_len  # donor < acceptor
        snv, donor, acceptor, delta = (t[keep] for t in
                                       (snv, donor, acceptor, delta))
    return snv, donor, acceptor, delta
