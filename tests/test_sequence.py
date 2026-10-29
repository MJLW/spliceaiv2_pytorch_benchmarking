import numpy as np
import pytest

from spliceai2_ism.encode import N_CODE, PAD_CODE, random_sequences
from spliceai2_ism.inputs import Gene, Haplotype, InputError, Variant
from spliceai2_ism.sequence import (
    haplotype_sequence,
    haplotype_window,
    output_length,
    read_region,
    reverse_complement,
    saturation,
    to_codes,
    to_string,
)

CHROM = random_sequences(1, 400, seed=7)[0]
FASTA = {"chr1": CHROM}
CROP, MIN_OUT, BUCKET, ALIGN = 40, 4, 16, 2


def region(gene, *variants):
    return read_region(FASTA, gene, [Haplotype(gene.name, "h", variants)],
                       CROP, MIN_OUT, BUCKET, ALIGN)


def var(pos, ref_len, alt):
    """Variant at pos whose REF is the reference's ref_len bases there."""
    return Variant("chr1", pos, CHROM[pos - 1:pos - 1 + ref_len], alt, line=9)


def test_to_codes_and_reverse_complement():
    assert to_codes("ACGTNacgx").tolist() == [0, 1, 2, 3, N_CODE, 0, 1, 2,
                                               N_CODE]
    rc = reverse_complement(np.array([0, 0, 1, 2, N_CODE, PAD_CODE], np.uint8))
    assert rc.tolist() == [PAD_CODE, N_CODE, 1, 2, 3, 3]


def test_output_length():
    assert output_length(100, 16, 2) == 112
    assert output_length(112, 16, 2) == 112
    with pytest.raises(ValueError):
        output_length(100, 15, 2)


def test_region_flank_covers_input_extension_and_deletions():
    gene = Gene("G", "chr1", 151, 160, "+")
    r = region(gene, var(130, 4, CHROM[129]))  # deletes 3 bases
    # crop / 2 + bucket + min output length + deleted + longest REF
    flank = 20 + 16 + 4 + 3 + 4
    assert r.start == 151 - flank
    assert to_string(r.codes) == CHROM[r.start - 1:160 + flank]


def test_region_padded_past_chromosome_ends():
    r = region(Gene("G", "chr1", 5, 10, "+"))
    assert r.start < 1
    assert (r.codes[:1 - r.start] == PAD_CODE).all()
    assert to_string(r.codes[1 - r.start:1 - r.start + 4]) == CHROM[:4]


def test_unknown_chromosome_and_gene_past_end():
    with pytest.raises(InputError, match="not in the FASTA"):
        region(Gene("G", "chr2", 5, 10, "+"))
    with pytest.raises(InputError, match="past the end"):
        region(Gene("G", "chr1", 395, 405, "+"))


def test_haplotype_sequence_applies_each_kind_of_variant():
    gene = Gene("G", "chr1", 151, 200, "+")
    snv = var(160, 1, "A" if CHROM[159] != "A" else "C")
    ins = var(170, 1, CHROM[169] + "GT")       # inserts GT after 170
    dele = var(180, 4, CHROM[179])              # deletes 181-183
    mnv = var(190, 2, "TT" if CHROM[189:191] != "TT" else "GG")
    r = region(gene, snv, ins, dele, mnv)
    codes, pos, ins_off, hap_pos, outside = haplotype_sequence(
        r, [mnv, snv, dele, ins])
    assert outside == []
    # haplotype positions run on, shifted by the indels before them
    assert (np.diff(hap_pos) == 1).all()
    hap_at = dict(zip(zip(pos.tolist(), ins_off.tolist()), hap_pos.tolist()))
    assert hap_at[(170, 0)] == 170 and hap_at[(170, 2)] == 172
    # +2 inserted after 170, -3 deleted after 180
    assert hap_at[(171, 0)] == 173 and hap_at[(184, 0)] == 183
    coords = list(zip(pos.tolist(), ins_off.tolist()))
    base = dict(zip(coords, to_string(codes)))
    assert base[(160, 0)] == snv.alt
    assert (base[(170, 0)], base[(170, 1)], base[(170, 2)]) == (
        CHROM[169], "G", "T")
    assert (171, 0) in base and (170, 3) not in base
    assert all((p, 0) not in base for p in (181, 182, 183))
    assert base[(180, 0)] == CHROM[179] and base[(184, 0)] == CHROM[183]
    assert base[(190, 0)] + base[(191, 0)] == mnv.alt
    # ordered: positions non-decreasing, offsets increasing after an anchor
    assert coords == sorted(coords)
    assert len(codes) == len(r.codes) + 2 - 3


def test_ref_mismatch_and_overlap():
    gene = Gene("G", "chr1", 151, 200, "+")
    wrong = Variant("chr1", 160, "ACGT" if CHROM[159:163] != "ACGT" else "TTTT",
                    "A", line=4)
    with pytest.raises(InputError, match="but the FASTA has"):
        haplotype_sequence(region(gene, wrong), [wrong])
    a, b = var(160, 4, CHROM[159]), var(162, 1, "A")
    with pytest.raises(InputError, match="overlaps"):
        haplotype_sequence(region(gene, a, b), [a, b])


@pytest.mark.parametrize("strand", ["+", "-"])
def test_window_coordinates(strand):
    gene = Gene("G", "chr1", 151, 170, strand)
    ins = var(160, 1, CHROM[159] + "AC")
    dele = var(165, 3, CHROM[164])
    r = region(gene, ins, dele)
    w = haplotype_window(r, [ins, dele])
    assert w.gene_len == 20 + 2 - 2
    assert w.out_len == 32 and len(w.codes_t) == 32 + CROP
    ref = list(range(151, 161)) + [0, 0] + list(range(161, 166)) + [168, 169,
                                                                     170]
    hap = list(range(151, 171))  # +2 inserted, -2 deleted: back in step
    if strand == "-":  # output index 0 is at TX_END
        ref, hap = ref[::-1], hap[::-1]
    assert w.reference_pos.tolist() == ref
    assert w.haplotype_pos.tolist() == hap
    # the model input around the gene body is the haplotype sequence
    codes, pos, ins_off, _, _ = haplotype_sequence(r, [ins, dele])
    hap = to_string(codes)
    b0 = int(np.nonzero((pos == 151) & (ins_off == 0))[0][0])
    b1 = int(np.nonzero((pos == 170) & (ins_off == 0))[0][0]) + 1
    half = CROP // 2
    if strand == "+":
        assert to_string(w.codes_t) == hap[b0 - half:b0 + 32 + half]
    else:
        fwd = to_codes(hap[b1 - 32 - half:b1 + half])
        assert to_string(w.codes_t) == to_string(reverse_complement(fwd))
        assert w.forward_bases(np.array([0, 2])).tolist() == [3, 1]


def test_insertions_at_gene_edges():
    gene = Gene("G", "chr1", 151, 170, "+")
    before = var(150, 1, CHROM[149] + "A")  # after TX_START - 1: flank
    last = var(169, 1, CHROM[168] + "C")    # after TX_END - 1: gene
    after = var(170, 1, CHROM[169] + "G")   # after TX_END: flank
    w = haplotype_window(region(gene, before, last, after),
                         [before, last, after])
    # only the insertion after 169 is in the gene: after 151..169, before 170
    ref = w.reference_pos.tolist()
    assert w.gene_len == 21 and ref.count(0) == 1
    assert ref[ref.index(0) - 1:ref.index(0) + 2] == [169, 0, 170]
    # the insertion before the gene shifts haplotype positions by one
    assert w.haplotype_pos.tolist() == list(range(152, 173))


def test_indels_before_the_region_shift_haplotype_positions():
    gene = Gene("G", "chr1", 351, 360, "+")
    far = var(10, 1, CHROM[9] + "AAA")  # outside the region, 3 nt inserted
    w = haplotype_window(region(gene), [far])
    assert w.outside == (far,)
    assert w.reference_pos.tolist() == list(range(351, 361))
    assert w.haplotype_pos.tolist() == list(range(354, 364))


def test_whole_gene_deleted():
    gene = Gene("G", "chr1", 151, 160, "+")
    dele = var(150, 12, CHROM[149])
    with pytest.raises(InputError, match="deletes the whole gene"):
        haplotype_window(region(gene, dele), [dele])


def test_saturation_skips_n_and_covers_three_alts():
    gene = Gene("G", "chr1", 151, 160, "+")
    w = haplotype_window(region(gene), [])
    codes_t = w.codes_t.copy()
    half = CROP // 2
    codes_t[half + 2] = N_CODE  # an N in the gene body is not saturated
    w = type(w)(**{**w.__dict__, "codes_t": codes_t})
    pos, alt = saturation(w)
    assert len(pos) == 3 * 9 and 2 not in pos.tolist()
    body = codes_t[half:half + w.gene_len]
    for p in set(pos.tolist()):
        assert sorted(alt[pos == p].tolist()) == sorted(
            {0, 1, 2, 3} - {int(body[p])})
