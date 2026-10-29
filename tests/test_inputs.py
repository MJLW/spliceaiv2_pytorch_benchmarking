import pytest

from spliceai2_ism.inputs import (
    REFERENCE,
    InputError,
    read_haplotypes,
    read_regions,
)

REGIONS = ("NAME\tCHROM\tSTRAND\tTX_START\tTX_END\n"
           "G1\tchr1\t+\t301\t400\nG2\tchr1\t-\t501\t620\n")


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def variants(tmp_path, *rows):
    header = "#CHROM\tPOS\tREF\tALT\tgroup\tgene\n"
    return write(tmp_path, "v.tsv", header + "".join(r + "\n" for r in rows))


@pytest.fixture
def genes(tmp_path):
    return read_regions(write(tmp_path, "r.tsv", REGIONS))


def test_regions(genes):
    assert list(genes) == ["G1", "G2"]
    assert genes["G2"].strand == "-" and genes["G2"].length == 120


def test_bad_strand(tmp_path):
    with pytest.raises(InputError, match="STRAND"):
        read_regions(write(tmp_path, "r.tsv",
                           "NAME\tCHROM\tSTRAND\tTX_START\tTX_END\n"
                           "G\tc\t.\t1\t5\n"))


def test_haplotypes_grouped_and_sorted(tmp_path, genes):
    haps = read_haplotypes(variants(
        tmp_path,
        "chr1\t350\tA\tG\tHG002.1\tG1",
        "chr1\t320\tc\tt\tHG002.1\tG1",
        "chr1\t330\tA\tC\t7\tG1",
        "chr1\t510\tA\tC\t7\tG2",  # same id in another gene: another haplotype
    ), genes)
    # the reference first, then the groups in order of first appearance
    assert [h.group for h in haps["G1"]] == [REFERENCE, "HG002.1", "7"]
    assert haps["G1"][0].variants == ()
    assert [v.pos for v in haps["G1"][1].variants] == [320, 350]
    assert haps["G1"][1].variants[0].ref == "C"  # upper-cased
    assert [h.group for h in haps["G2"]] == [REFERENCE, "7"]


def test_genes_without_variants_have_no_haplotypes(tmp_path, genes):
    haps = read_haplotypes(variants(tmp_path, "chr1\t330\tA\tC\ta\tG1"), genes)
    assert "G2" not in haps


def test_indels_and_mnvs_accepted(tmp_path, genes):
    haps = read_haplotypes(variants(
        tmp_path,
        "chr1\t330\tA\tATG\t0\tG1",   # insertion
        "chr1\t340\tACGT\tA\t0\tG1",  # deletion
        "chr1\t350\tAC\tGT\t0\tG1",   # MNV
    ), genes)
    assert [(v.ref, v.alt) for v in haps["G1"][1].variants] == [
        ("A", "ATG"), ("ACGT", "A"), ("AC", "GT")]


def test_overlapping_variants(tmp_path, genes):
    with pytest.raises(InputError, match="overlaps the variant on line 2"):
        read_haplotypes(variants(
            tmp_path, "chr1\t340\tACGT\tA\t0\tG1", "chr1\t342\tG\tC\t0\tG1",
        ), genes)
    # adjacent is fine
    read_haplotypes(variants(
        tmp_path, "chr1\t340\tACGT\tA\t0\tG1", "chr1\t344\tG\tC\t0\tG1",
    ), genes)


@pytest.mark.parametrize("row, message", [
    ("chr1\t330\tA\t-\t0\tG1", "non-empty sequences of A, C, G, T"),
    ("chr1\t330\tA\t<DEL>\t0\tG1", "non-empty sequences of A, C, G, T"),
    ("chr1\t330\tA\tN\t0\tG1", "non-empty sequences of A, C, G, T"),
    ("chr1\t330\tA\tA\t0\tG1", "ALT equals REF"),
    ("chr2\t330\tA\tC\t0\tG1", "differs from"),
    ("chr1\t330\tA\tC\t0\tG9", "not in the region file"),
    ("chr1\tx\tA\tC\t0\tG1", "POS must be an integer"),
    ("chr1\t330\tA\tC\t\tG1", "empty group"),
    ("chr1\t330\tA\tC\treference\tG1", "reserved for the reference"),
])
def test_rejected_variants(tmp_path, genes, row, message):
    with pytest.raises(InputError, match=message):
        read_haplotypes(variants(tmp_path, row), genes)


def test_two_variants_at_one_position(tmp_path, genes):
    with pytest.raises(InputError, match="already has a variant"):
        read_haplotypes(variants(
            tmp_path, "chr1\t330\tA\tC\t0\tG1", "chr1\t330\tA\tG\t0\tG1",
        ), genes)


def test_missing_column(tmp_path, genes):
    path = write(tmp_path, "v.tsv", "CHROM\tPOS\tREF\tALT\tgene\n")
    with pytest.raises(InputError, match="missing column"):
        read_haplotypes(path, genes)
