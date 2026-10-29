"""The ISM pipeline with the released models on the GPU, compiled and eager,
against full forward passes of the mutated haplotypes.

Run with `uv run pytest -m gpu`; needs CUDA and access to the gated
Hugging Face weights (skipped otherwise). FP16 autocast makes both paths
round differently, so probabilities are compared with a tolerance: above
the ~1e-4 fp16 noise, below the default 1e-3 delta threshold.
"""
import numpy as np
import pytest
import torch

from spliceai2_ism.deltas import delta_1d, delta_2d
from spliceai2_ism.encode import one_hot_codes, random_sequences
from spliceai2_ism.incremental import IncrementalEnsemble
from spliceai2_ism.inputs import Gene, Haplotype, Variant
from spliceai2_ism.ism import Stages
from spliceai2_ism.predict import predict_full
from spliceai2_ism.runtime import inference
from spliceai2_ism.sequence import haplotype_window, read_region, saturation

pytestmark = pytest.mark.gpu

TOLERANCE = 5e-4
GENE_LEN, OUT_2D_LEN = 16_384, 1024


@pytest.fixture(scope="module")
def models():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    from spliceai2_ism.weights import load_released_models
    try:
        return load_released_models()
    except Exception as e:  # no login / no accepted terms / no network
        pytest.skip(f"released weights unavailable: {e}")


@pytest.mark.parametrize("compile_", [True, False], ids=["compiled", "debug"])
@pytest.mark.parametrize("strand", ["+", "-"])
def test_snv_deltas_match_full_passes(models, strand, compile_):
    device = torch.device("cuda")
    inc = IncrementalEnsemble(models, OUT_2D_LEN).eval().to(device)
    half = inc.crop_len // 2
    chrom = random_sequences(1, GENE_LEN + 2 * half + 50_000, seed=3)[0]
    gene = Gene("G", "chr1", half + 20_001, half + 20_000 + GENE_LEN, strand)
    def at(p, n=1):
        return chrom[p - 1:p - 1 + n]

    # an SNV, an 8 nt insertion and an 8 nt deletion in the gene; the gene
    # body keeps its length, so input lengths stay divisible by the pooling
    s, i, d = gene.start + 5_000, gene.start + 9_000, gene.start + 12_000
    hap = Haplotype("G", 0, (
        Variant("chr1", s, at(s), "C" if at(s) == "A" else "A", 0),
        Variant("chr1", i, at(i), at(i) + "ACGTTGCA", 0),
        Variant("chr1", d, at(d, 9), at(d), 0),
    ))
    region = read_region({"chr1": chrom}, gene, [hap], inc.crop_len,
                         OUT_2D_LEN, 16_384, inc.align)
    window = haplotype_window(region, hap.variants)
    assert window.gene_len == GENE_LEN
    codes_t = window.codes_t
    x = torch.from_numpy(one_hot_codes(codes_t))[None].to(device)
    valid_len = torch.tensor([GENE_LEN], device=device)
    stages = Stages(inc, compile_)

    # SNVs at both ends of the gene body and in between, one batch
    snv_pos, snv_alt = saturation(window)
    pick = np.array([0, 1, 3 * 8_000, 3 * 8_000 + 2, len(snv_pos) - 1])
    pos = torch.from_numpy(snv_pos[pick]).to(device)
    alt = torch.from_numpy(snv_alt[pick]).to(device)
    with inference(device):
        cache, hap_ss, hap_idx, hap_jxn = stages.reference(x, valid_len)
        win, v_idx, v_jxn = stages.variants(
            cache, hap_ss.amax(dim=1, keepdim=True), pos, alt, valid_len
        )
        snv1, o1, dd, da = delta_1d(inc, hap_ss, pos, win, gene.length, 0.0)
        snv2, don, acc, d2 = delta_2d(hap_idx[0], hap_jxn[0], v_idx, v_jxn, 0.0)
        hap_p1 = hap_ss[0, :, :gene.length].float().sigmoid()
        hap_pairs = _pairs(hap_idx[0], hap_jxn[0].float().sigmoid())
        del cache

        for b, (p, a) in enumerate(zip(snv_pos[pick], snv_alt[pick])):
            mutated = codes_t.copy()
            mutated[half + p] = a
            # the full pass sees only the gene body, no length bucket
            xm = torch.from_numpy(one_hot_codes(
                mutated[:gene.length + inc.crop_len]
            ))[None].to(device)
            ss, idx, jx = predict_full(models, xm, OUT_2D_LEN)
            full_p1 = ss[0].float().sigmoid()

            # 1D: haplotype + deltas == full pass, at every gene position
            recon = hap_p1.clone()
            m = snv1 == b
            recon[0, o1[m]] += dd[m].to(recon.dtype)
            recon[1, o1[m]] += da[m].to(recon.dtype)
            assert (recon - full_p1).abs().max().item() < TOLERANCE

            # 2D, for pairs the SNV's top-k and the full pass's both have
            m = snv2 == b
            recon2 = dict(hap_pairs)
            for d_, a_, v_ in zip(don[m].tolist(), acc[m].tolist(),
                                  d2[m].tolist()):
                recon2[(d_, a_)] = recon2.get((d_, a_), 0.0) + v_
            full2 = _pairs(idx[0], jx[0].float().sigmoid())
            shared = full2.keys() & set(_pairs(v_idx[b], v_jxn[b]).keys())
            assert len(shared) > 0.9 * len(full2)
            assert max(abs(recon2[k] - full2[k]) for k in shared) < TOLERANCE


def _pairs(idx, probs):
    """{(donor, acceptor): prob} over the upper triangle."""
    idx, probs = idx.cpu().numpy(), probs.float().cpu().numpy()
    i, j = np.triu_indices(len(idx), 1)
    return dict(zip(zip(idx[i].tolist(), idx[j].tolist()),
                    probs[i, j].tolist()))
