import torch

from spliceai2_ism.deltas import delta_2d, haplotype_rows


def logit(p):
    p = torch.as_tensor(p, dtype=torch.float32)
    return torch.log(p) - torch.log1p(-p)


def pairs(idx, probs):
    """{(donor, acceptor): prob} over the upper triangle."""
    k = len(idx)
    return {(int(idx[i]), int(idx[j])): float(probs[i, j])
            for i in range(k) for j in range(i + 1, k)}


def test_delta_2d_keys_pairs_by_site():
    torch.manual_seed(0)
    k = 4
    hap_idx = torch.tensor([10, 20, 30, 40])
    var_idx = torch.tensor([[10, 20, 30, 40],    # same sites
                            [10, 25, 30, 40]])   # site 20 replaced by 25
    p_h = torch.rand(k, k) * 0.9 + 0.05
    p_v = torch.rand(2, k, k) * 0.9 + 0.05
    snv, donor, acceptor, delta = delta_2d(
        hap_idx, logit(p_h), var_idx, logit(p_v), min_delta=0.0, chunk=1
    )
    hap = pairs(hap_idx, p_h)
    for b in range(2):
        var = pairs(var_idx[b], p_v[b])
        expected = {key: var.get(key, 0.0) - hap.get(key, 0.0)
                    for key in hap.keys() | var.keys()}
        got = {(int(d), int(a)): float(x)
               for s, d, a, x in zip(snv, donor, acceptor, delta) if s == b}
        assert got.keys() == expected.keys()
        for key in expected:
            assert abs(got[key] - expected[key]) < 1e-6
    # variant 0 has the haplotype's sites: only its own pairs, none "lost"
    assert int((snv == 0).sum()) == k * (k - 1) // 2


def test_delta_2d_threshold():
    hap_idx = torch.tensor([1, 2, 3])
    p = torch.full((3, 3), 0.5)
    p_v = p.clone().unsqueeze(0)
    p_v[0, 0, 2] = 0.52
    snv, donor, acceptor, delta = delta_2d(hap_idx, logit(p), hap_idx[None],
                                           logit(p_v), min_delta=1e-3)
    assert snv.tolist() == [0] and (donor.item(), acceptor.item()) == (1, 3)
    assert abs(delta.item() - 0.02) < 1e-5


def test_delta_2d_keeps_pairs_inside_gene():
    hap_idx = torch.tensor([1, 4, 8])
    var_idx = torch.tensor([[1, 4, 9]])
    p = torch.full((3, 3), 0.5)
    snv, donor, acceptor, _ = delta_2d(hap_idx, logit(p), var_idx,
                                       logit(p[None]), 0.0, gene_len=5)
    assert list(zip(donor.tolist(), acceptor.tolist())) == [(1, 4)]


def test_haplotype_rows_keeps_pairs_inside_gene():
    idx = torch.tensor([2, 5, 12])
    _, donor, acceptor, jp = haplotype_rows(torch.zeros(1, 2, 16), idx,
                                            torch.zeros(3, 3), gene_len=10)
    assert list(zip(donor.tolist(), acceptor.tolist())) == [(2, 5)]
    assert jp.shape == (1,)


def test_haplotype_rows():
    hap_ss = torch.zeros(1, 2, 12)
    idx = torch.tensor([2, 5, 7])
    jxn = torch.zeros(3, 3)
    probs, donor, acceptor, jp = haplotype_rows(hap_ss, idx, jxn, gene_len=10)
    assert probs.shape == (2, 10) and torch.allclose(probs, torch.tensor(0.5))
    assert list(zip(donor.tolist(), acceptor.tolist())) == [(2, 5), (2, 7), (5, 7)]
    assert jp.shape == (3,)
