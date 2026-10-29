"""End-to-end ISM on CPU with small random models: every reported number is
checked against independent full forward passes of the haplotype and SNV
sequences, built with plain string operations."""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from helpers import CROP_LEN, small_models
from spliceai2_ism import ism
from spliceai2_ism.encode import one_hot_encode, random_sequences
from spliceai2_ism.incremental import IncrementalEnsemble
from spliceai2_ism.inputs import REFERENCE, Gene, Haplotype, Variant
from spliceai2_ism.ism import prepare_regions, run_ism
from spliceai2_ism.output import SCHEMAS, Writer
from spliceai2_ism.predict import predict_full

OUT_2D_LEN = 16
HALF = CROP_LEN // 2
CHROM = random_sequences(1, 1500, seed=11)[0]
GENES = [Gene("G1", "chr1", 401, 500, "+"), Gene("G2", "chr1", 801, 920, "-"),
         # shorter than OUT_2D_LEN: candidate sites also come from after it
         Gene("G3", "chr1", 1201, 1210, "-")]
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def other(base, k=1):
    return "ACGT"[("ACGT".index(base) + k) % 4]


def snv(pos, k=1):
    return Variant("chr1", pos, CHROM[pos - 1], other(CHROM[pos - 1], k), 0)


def ins(pos, bases):
    return Variant("chr1", pos, CHROM[pos - 1], CHROM[pos - 1] + bases, 0)


def dele(pos, n):
    """Deletes the n bases after pos."""
    return Variant("chr1", pos, CHROM[pos - 1:pos + n], CHROM[pos - 1], 0)


HAPLOTYPES = {  # as read_haplotypes returns them: the reference first
    "G1": [Haplotype("G1", REFERENCE, ()),
           Haplotype("G1", "indels", (snv(320),        # flank
                                      ins(430, "GT"),  # gene
                                      snv(450, 2),     # gene
                                      dele(460, 2)))],  # gene: 461-462 deleted
    "G2": [Haplotype("G2", REFERENCE, ()),
           Haplotype("G2", "snv", (snv(850),)),
           Haplotype("G2", "indels", (dele(700, 4),    # flank, before the gene
                                      ins(880, "A"),   # gene
                                      ins(920, "CC"),  # after TX_END: flank
                                      snv(1000, 3)))],  # flank
    "G3": [Haplotype("G3", REFERENCE, ()),
           Haplotype("G3", "snv", (snv(1205),)),
           Haplotype("G3", "indels", (dele(1203, 2),))],  # 1204-1205 deleted
}


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("ism")
    models = small_models()
    inc = IncrementalEnsemble(models, OUT_2D_LEN).eval()
    regions = prepare_regions(inc, GENES, HAPLOTYPES, {"chr1": CHROM},
                              length_bucket=16)
    with Writer(out_dir) as writer:
        stats = run_ism(inc, regions, HAPLOTYPES, writer, torch.device("cpu"),
                        compile_=False, batch_size=64, min_delta=0.0,
                        log=lambda *a, **k: None)
    tables = {name: pq.read_table(out_dir / f"{name}.parquet")
              for name in SCHEMAS}
    return models, tables, stats


def select(table, **eq):
    mask = None
    for col, value in eq.items():
        m = (pc.is_null(table[col]) if value is None
             else pc.equal(table[col], value))
        mask = m if mask is None else pc.and_(mask, m)
    return table.filter(mask).to_pydict()


def build(hap):
    """The haplotype's chromosome, forward strand: its string and, per
    base, (reference position or None, haplotype position) and the
    reference position it is at or follows."""
    by_pos = {v.pos: v for v in hap.variants}
    seq, ref, anchor, p = [], [], [], 1
    while p <= len(CHROM):
        v = by_pos.get(p)
        if v is None:
            seq.append(CHROM[p - 1])
            ref.append(p)
            anchor.append(p)
            p += 1
            continue
        for i, b in enumerate(v.alt):
            seq.append(b)
            ref.append(p + i if i < len(v.ref) else None)
            anchor.append(min(p + i, p + len(v.ref) - 1))
        p += len(v.ref)
    coords = [(r, i + 1) for i, r in enumerate(ref)]  # haplotype pos: 1, 2, ..
    return "".join(seq), coords, anchor


def gene_body(gene, coords, anchor):
    idx = [i for i, ((r, _), a) in enumerate(zip(coords, anchor))
           if (gene.start <= r <= gene.end if r is not None
               else gene.start <= a < gene.end)]
    return idx[0], idx[-1] + 1


def full_prediction(models, gene, hap, change=None):
    """Splice sites {(ref, hap): (donor, acceptor)} and junctions
    {((ref, hap), (ref, hap)): prob} from one full forward pass of the
    haplotype, with ``change`` = (haplotype position, forward base) applied.

    The pass predicts at least OUT_2D_LEN positions from the gene start (in
    transcript orientation), rounded up to an even input length; candidate
    sites come from the first max(gene body, OUT_2D_LEN) of them. Only the
    gene's bases and the junctions with both sites in the gene are
    returned."""
    seq, coords, anchor = build(hap)
    if change is not None:
        i = change[0] - 1
        seq = seq[:i] + change[1] + seq[i + 1:]
    b0, b1 = gene_body(gene, coords, anchor)
    valid = max(b1 - b0, OUT_2D_LEN)  # candidate sites, as in the ISM
    n = valid + valid % 2  # input length divisible by the pooling (2)
    if gene.strand == "+":
        seq = seq[b0 - HALF:b0 + n + HALF]
        coord = lambda o: coords[b0 + o]  # noqa: E731
    else:
        seq = seq[b1 - n - HALF:b1 + HALF].translate(COMPLEMENT)[::-1]
        coord = lambda o: coords[b1 - 1 - o]  # noqa: E731
    x = torch.from_numpy(one_hot_encode(seq))[None]
    with torch.inference_mode():
        out_ss, idxs, jxn = predict_full(models, x, OUT_2D_LEN,
                                         torch.tensor([valid]))
    probs = out_ss[0].sigmoid()
    sites = {coord(o): (float(probs[0, o]), float(probs[1, o]))
             for o in range(b1 - b0)}
    idxs, p = idxs[0].tolist(), jxn[0].sigmoid()
    junctions = {(coord(idxs[i]), coord(idxs[j])): float(p[i, j])
                 for i in range(len(idxs)) for j in range(i + 1, len(idxs))
                 if idxs[j] < b1 - b0}
    return sites, junctions


def site_rows(rows, donor, acceptor):
    return {(r, h): (d, a) for r, h, d, a in zip(
        rows["reference_pos"], rows["haplotype_pos"], rows[donor],
        rows[acceptor])}


def junction_rows(rows, value):
    return {((dr, dh), (ar, ah)): v for dr, dh, ar, ah, v in zip(
        rows["donor_reference_pos"], rows["donor_haplotype_pos"],
        rows["acceptor_reference_pos"], rows["acceptor_haplotype_pos"],
        rows[value])}


def assert_close(got, expected, atol=1e-5):
    for key in got.keys() | expected.keys():
        a, b = got.get(key, 0.0), expected.get(key, 0.0)
        assert np.allclose(a, b, atol=atol, rtol=0), key


def predictions(tables, gene, group):
    sites = site_rows(select(tables["predictions_sites"], gene=gene,
                             group=group), "donor_prob", "acceptor_prob")
    junctions = junction_rows(select(tables["predictions_junctions"],
                                     gene=gene, group=group), "prob")
    return sites, junctions


@pytest.mark.parametrize("gene", GENES, ids=lambda g: g.name)
def test_haplotype_predictions_match_full_pass(outputs, gene):
    models, tables, _ = outputs
    for hap in HAPLOTYPES[gene.name]:
        sites, junctions = predictions(tables, gene.name, hap.group)
        _, coords, anchor = build(hap)
        b0, b1 = gene_body(gene, coords, anchor)
        assert set(sites) == set(coords[b0:b1])
        exp_sites, exp_junctions = full_prediction(models, gene, hap)
        assert_close(sites, exp_sites)
        assert junctions.keys() == exp_junctions.keys()
        assert_close(junctions, exp_junctions)


def test_indel_coordinates(outputs):
    _, tables, _ = outputs
    sites, _ = predictions(tables, "G1", "indels")
    # +2 inserted after 430, -2 deleted after 460
    assert {(430, 430), (None, 431), (None, 432), (431, 433), (460, 462),
            (463, 463)} <= set(sites)
    assert not any(r in (461, 462) for r, _ in sites)
    sites, _ = predictions(tables, "G2", "indels")
    # -4 deleted after 700 shifts the whole gene; +1 inserted after 880
    assert {(801, 797), (880, 876), (None, 877), (881, 878),
            (920, 917)} <= set(sites)
    assert len(sites) == 121  # the insertion after TX_END is flank


@pytest.mark.parametrize("gene", GENES, ids=lambda g: g.name)
def test_snv_deltas_match_full_pass(outputs, gene):
    models, tables, _ = outputs
    for hap in HAPLOTYPES[gene.name]:
        seq, coords, anchor = build(hap)
        b0, b1 = gene_body(gene, coords, anchor)
        body = coords[b0:b1]
        picks = {body[0], body[len(body) // 3], body[-1]}
        picks |= {c for c in body if c[0] is None}  # inserted bases
        changes = [(c, other(seq[c[1] - 1], k % 3 + 1))
                   for k, c in enumerate(sorted(picks, key=str))]
        changes += [((v.pos, h), v.ref) for v in hap.variants  # revert
                    for r, h in body
                    if r == v.pos and len(v.ref) == len(v.alt) == 1]
        hap_sites, hap_jxn = predictions(tables, gene.name, hap.group)
        for (ref_pos, hap_pos), alt in changes:
            eq = dict(gene=gene.name, group=hap.group,
                      snv_haplotype_pos=hap_pos, snv_alt=alt)
            rows = select(tables["deltas_sites"], **eq)
            assert set(rows["snv_ref"]) == {seq[hap_pos - 1]}
            assert set(rows["snv_reference_pos"]) == {ref_pos}
            d_sites = site_rows(rows, "donor_delta", "acceptor_delta")
            d_jxn = junction_rows(select(tables["deltas_junctions"], **eq),
                                  "delta")
            sites = {k: tuple(np.add(v, d_sites.get(k, (0.0, 0.0))))
                     for k, v in hap_sites.items()}
            jxn = {k: hap_jxn.get(k, 0.0) + d_jxn.get(k, 0.0)
                   for k in hap_jxn.keys() | d_jxn.keys()}
            exp_sites, exp_jxn = full_prediction(models, gene, hap,
                                                 (hap_pos, alt))
            assert_close(sites, exp_sites)
            assert_close(jxn, exp_jxn)


def test_every_snv_scored(outputs):
    _, tables, stats = outputs
    for st in stats:  # no N in the test chromosome
        assert st["snvs"] == 3 * st["gene_len"]
    seq, coords, anchor = build(HAPLOTYPES["G2"][2])
    b0, b1 = gene_body(GENES[1], coords, anchor)
    rows = select(tables["deltas_sites"], gene="G2", group="indels")
    snvs = set(zip(rows["snv_reference_pos"], rows["snv_haplotype_pos"],
                   rows["snv_alt"]))
    assert len(snvs) == 3 * (b1 - b0)
    assert {(r, h) for r, h, _ in snvs} == set(coords[b0:b1])
    # SNV alleles are forward-strand bases, never the haplotype's own base
    assert all(alt != seq[h - 1] for _, h, alt in snvs)
    assert [(st["gene"], st["group"], st["variants"]) for st in stats] == [
        ("G1", REFERENCE, 0), ("G1", "indels", 4),
        ("G2", REFERENCE, 0), ("G2", "snv", 1), ("G2", "indels", 4),
        ("G3", REFERENCE, 0), ("G3", "snv", 1), ("G3", "indels", 1)]
    # the references are in the tables with the other haplotypes
    groups = set(zip(*(tables["predictions_sites"][c].to_pylist()
                       for c in ("gene", "group"))))
    assert groups == {(g, h.group) for g, haps in HAPLOTYPES.items()
                      for h in haps}
    assert np.isfinite(rows["donor_delta"]).all()


def sort_keys(table):
    keys = ["group", "snv_haplotype_pos", "snv_alt"]
    keys += [c for c in ("haplotype_pos", "donor_haplotype_pos",
                         "acceptor_haplotype_pos") if c in table.column_names]
    return table.sort_by([(c, "ascending") for c in keys])


def test_out_of_memory_halves_batch_size(outputs, tmp_path, monkeypatch):
    """A batch that runs out of memory is retried at half the size; the
    output is the same as without the failure."""
    models, tables, _ = outputs

    class OutOfMemoryAbove16(ism.Stages):
        def __init__(self, inc, compile_):
            super().__init__(inc, compile_)
            variants = self.variants

            def guarded(cache, hap_scores, pos, alt, valid_len):
                if len(pos) > 16:
                    raise torch.cuda.OutOfMemoryError("simulated")
                return variants(cache, hap_scores, pos, alt, valid_len)

            self.variants = guarded

    monkeypatch.setattr(ism, "Stages", OutOfMemoryAbove16)
    inc = IncrementalEnsemble(models, OUT_2D_LEN).eval()
    haps = {"G1": HAPLOTYPES["G1"]}
    regions = prepare_regions(inc, GENES[:1], haps, {"chr1": CHROM},
                              length_bucket=16)
    logged = []
    with Writer(tmp_path) as writer:
        stats = run_ism(inc, regions, haps, writer, torch.device("cpu"),
                        compile_=False, batch_size=64, min_delta=0.0,
                        log=lambda msg, **k: logged.append(msg))
    assert [st["batch_size"] for st in stats] == [16, 16]
    assert sum("batch size now 32" in m for m in logged) == 1
    assert sum("batch size now 16" in m for m in logged) == 1

    for name, values in (("deltas_sites", ("donor_delta", "acceptor_delta")),
                         ("deltas_junctions", ("delta",))):
        got = sort_keys(pq.read_table(tmp_path / f"{name}.parquet"))
        expected = sort_keys(tables[name].filter(
            pc.equal(tables[name]["gene"], "G1")))
        assert got.num_rows == expected.num_rows
        for col in got.column_names:
            if col in values:
                np.testing.assert_allclose(got[col].to_numpy(),
                                           expected[col].to_numpy(), atol=1e-5)
            else:
                assert got[col].to_pylist() == expected[col].to_pylist()


def test_short_gene_reports_only_junctions_inside(outputs):
    _, tables, _ = outputs
    gene = GENES[2]
    for name in ("predictions_junctions", "deltas_junctions"):
        rows = select(tables[name], gene="G3")
        assert rows["donor_reference_pos"]
        for col in ("donor_reference_pos", "acceptor_reference_pos"):
            assert all(gene.start <= p <= gene.end for p in rows[col])
    # minus strand: the donor lies at the higher coordinate
    rows = select(tables["predictions_junctions"], gene="G3")
    assert all(d > a for d, a in zip(rows["donor_haplotype_pos"],
                                     rows["acceptor_haplotype_pos"]))
