import gzip

import pytest

from spliceai2_ism.gencode import main
from spliceai2_ism.inputs import Gene, read_regions


def gff_line(chrom, kind, start, end, strand, **attrs):
    col9 = ";".join(f"{k}={v}" for k, v in attrs.items())
    return f"{chrom}\tHAVANA\t{kind}\t{start}\t{end}\t.\t{strand}\t.\t{col9}\n"


GFF = (
    "##gff-version 3\n"
    "#description: test\n"
    + gff_line("chr1", "gene", 65419, 71585, "+", ID="ENSG01.1",
               gene_id="ENSG01.1", gene_type="protein_coding",
               gene_name="OR4F5")
    + gff_line("chr1", "transcript", 65419, 71585, "+", ID="ENST01.1",
               gene_id="ENSG01.1", gene_type="protein_coding",
               gene_name="OR4F5")
    + gff_line("chr1", "gene", 14404, 29570, "-", ID="ENSG02.1",
               gene_id="ENSG02.1", gene_type="unprocessed_pseudogene",
               gene_name="WASH7P")
    + gff_line("chr1", "gene", 100, 200, "-", ID="ENSG03.1",
               gene_id="ENSG03.1", gene_type="protein_coding",
               gene_name="A%3BB")  # percent-encoded ';'
    # a pseudoautosomal gene, annotated on both X and Y
    + gff_line("chrX", "gene", 624344, 659411, "+", ID="ENSG04.1",
               gene_id="ENSG04.1", gene_type="protein_coding",
               gene_name="SHOX")
    + gff_line("chrY", "gene", 624344, 659411, "+", ID="ENSG05.1",
               gene_id="ENSG05.1", gene_type="protein_coding",
               gene_name="SHOX")
    + "##FASTA\n>chr1\nACGT\n"
)


@pytest.mark.parametrize("gzipped", [False, True])
def test_writes_protein_coding_regions(tmp_path, capsys, gzipped):
    gff = tmp_path / ("a.gff3.gz" if gzipped else "a.gff3")
    if gzipped:
        with gzip.open(gff, "wt") as f:
            f.write(GFF)
    else:
        gff.write_text(GFF)
    out = tmp_path / "regions.tsv"
    main([str(gff), "-o", str(out)])

    assert read_regions(out) == {
        "OR4F5": Gene("OR4F5", "chr1", 65419, 71585, "+"),
        "A;B": Gene("A;B", "chr1", 100, 200, "-"),
        "SHOX_ENSG04.1": Gene("SHOX_ENSG04.1", "chrX", 624344, 659411, "+"),
        "SHOX_ENSG05.1": Gene("SHOX_ENSG05.1", "chrY", 624344, 659411, "+"),
    }
    assert "1 gene names occur more than once (SHOX)" in capsys.readouterr().err


def test_no_protein_coding_genes(tmp_path):
    gff = tmp_path / "a.gff3"
    gff.write_text("##gff-version 3\n")
    with pytest.raises(SystemExit, match="no protein-coding genes"):
        main([str(gff)])
