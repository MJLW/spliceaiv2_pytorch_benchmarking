"""Writes the region TSV for the protein-coding genes of a GENCODE GFF3.

One row per `gene` feature with gene_type=protein_coding: NAME (gene_name),
CHROM, STRAND, TX_START, TX_END. GFF3 coordinates are 1-based and inclusive,
like the region file, so they are copied unchanged. The span is the gene's,
from the earliest transcript start to the latest transcript end.

gene_name is not unique: genes in the pseudoautosomal regions are annotated
on both chrX and chrY, and a few names have more than one locus. Every gene
with such a name is written as NAME_gene_id instead (e.g.
SHOX_ENSG00000185960.15), so region names stay unique.
"""
import argparse
import gzip
import sys
from collections import Counter
from urllib.parse import unquote


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _attributes(field):
    """GFF3 column 9 as a dict, values percent-decoded."""
    return {k: unquote(v) for k, _, v in
            (a.partition("=") for a in field.strip().split(";") if "=" in a)}


def read_genes(path, gene_type="protein_coding"):
    """[(gene_name, gene_id, chrom, strand, start, end)], in file order."""
    genes = []
    with _open(path) as f:
        for line in f:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attrs = _attributes(fields[8])
            if attrs.get("gene_type") != gene_type:
                continue
            gene_id = attrs["gene_id"]
            genes.append((attrs.get("gene_name", gene_id), gene_id, fields[0],
                          fields[6], int(fields[3]), int(fields[4])))
    return genes


def regions(genes):
    """Region rows (NAME, CHROM, STRAND, TX_START, TX_END) and the gene
    names that occur more than once."""
    counts = Counter(name for name, *_ in genes)
    rows = [(f"{name}_{gene_id}" if counts[name] > 1 else name, chrom, strand,
             start, end)
            for name, gene_id, chrom, strand, start, end in genes]
    return rows, sorted(n for n, c in counts.items() if c > 1)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Write the spliceai2-ism region TSV for the "
                    "protein-coding genes of a GENCODE GFF3."
    )
    p.add_argument("gff", help="GENCODE GFF3, optionally gzipped")
    p.add_argument("-o", "--out", help="output TSV (default: stdout)")
    args = p.parse_args(argv)

    rows, duplicated = regions(read_genes(args.gff))
    if not rows:
        sys.exit(f"{args.gff}: no protein-coding genes "
                 f"(expected GENCODE GFF3 with gene_type attributes)")
    out = open(args.out, "w") if args.out else sys.stdout
    try:
        out.write("NAME\tCHROM\tSTRAND\tTX_START\tTX_END\n")
        for row in rows:
            out.write("\t".join(map(str, row)) + "\n")
    finally:
        if args.out:
            out.close()
    if duplicated:
        shown = ", ".join(duplicated[:5]) + (", ..." if len(duplicated) > 5
                                              else "")
        print(f"{len(duplicated)} gene names occur more than once ({shown}); "
              f"those genes are named NAME_gene_id", file=sys.stderr)
    print(f"{len(rows)} protein-coding genes"
          + (f" written to {args.out}" if args.out else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
