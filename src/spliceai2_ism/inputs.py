"""Reads and validates the gene region and haplotype variant TSVs.

Region TSV columns: NAME, CHROM, STRAND, TX_START, TX_END (1-based,
inclusive).
Variant TSV columns: CHROM, POS, REF, ALT, group, gene (POS 1-based).
Header names are case-insensitive and may start with '#'.

A haplotype is (gene, group): all variants of that group in that gene,
applied together. Group ids per gene run 0, 1, 2, ...; if they start at 1,
group 0 is the haplotype without variants (the reference sequence).
Variants are SNVs, MNVs, insertions, deletions or complex changes, written
as in VCF (e.g. insertion A -> ATG, deletion ACGT -> A); see sequence.py for
how they are applied. Variants of one haplotype may not overlap.
"""
import csv
from dataclasses import dataclass

BASES = "ACGT"


class InputError(ValueError):
    pass


@dataclass(frozen=True)
class Gene:
    name: str
    chrom: str
    start: int  # 1-based, inclusive
    end: int    # 1-based, inclusive
    strand: str

    @property
    def length(self):
        return self.end - self.start + 1


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int  # 1-based
    ref: str
    alt: str
    line: int  # line in the variant TSV, for messages


@dataclass(frozen=True)
class Haplotype:
    gene: str
    group: int
    variants: tuple  # Variants, sorted by position


def _rows(path, required):
    """(line number, row) per data line, keys lower-cased."""
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise InputError(f"{path}: empty file")
        header = [h.strip().lstrip("#").lower() for h in header]
        missing = [c for c in required if c not in header]
        if missing:
            raise InputError(f"{path}: missing column(s) {', '.join(missing)}; "
                             f"found {', '.join(header)}")
        for line, values in enumerate(reader, start=2):
            if not values or not "".join(values).strip():
                continue
            if len(values) != len(header):
                raise InputError(f"{path}:{line}: expected {len(header)} "
                                 f"fields, found {len(values)}")
            yield line, {k: v.strip() for k, v in zip(header, values)}


def _int(value, path, line, column):
    try:
        return int(value)
    except ValueError:
        raise InputError(f"{path}:{line}: {column} must be an integer, "
                         f"got {value!r}") from None


def read_regions(path):
    """{gene name: Gene}, in file order."""
    genes = {}
    for line, row in _rows(path, ("name", "chrom", "strand", "tx_start",
                                  "tx_end")):
        start = _int(row["tx_start"], path, line, "TX_START")
        end = _int(row["tx_end"], path, line, "TX_END")
        if not 1 <= start <= end:
            raise InputError(f"{path}:{line}: need 1 <= TX_START <= TX_END")
        if row["strand"] not in ("+", "-"):
            raise InputError(f"{path}:{line}: STRAND must be '+' or '-', "
                             f"got {row['strand']!r}")
        if row["name"] in genes:
            raise InputError(f"{path}:{line}: gene {row['name']!r} listed twice")
        genes[row["name"]] = Gene(row["name"], row["chrom"], start, end,
                                  row["strand"])
    if not genes:
        raise InputError(f"{path}: no genes")
    return genes


def read_haplotypes(path, genes):
    """{gene name: [Haplotype, ...] ordered by group}, for genes with
    variants."""
    groups = {}  # gene -> group -> pos -> Variant
    for line, row in _rows(path, ("chrom", "pos", "ref", "alt", "group", "gene")):
        where = f"{path}:{line}"
        gene = genes.get(row["gene"])
        if gene is None:
            raise InputError(f"{where}: gene {row['gene']!r} is not in the "
                             f"region file")
        if row["chrom"] != gene.chrom:
            raise InputError(f"{where}: CHROM {row['chrom']!r} differs from "
                             f"gene {gene.name}'s chrom {gene.chrom!r}")
        pos = _int(row["pos"], path, line, "POS")
        if pos < 1:
            raise InputError(f"{where}: POS must be >= 1")
        ref, alt = row["ref"].upper(), row["alt"].upper()
        if not ref or not alt or set(ref + alt) - set(BASES):
            raise InputError(f"{where}: REF and ALT must be non-empty "
                             f"sequences of A, C, G, T (got REF {ref!r}, "
                             f"ALT {alt!r}); write indels as in VCF, with "
                             f"the preceding base")
        if ref == alt:
            raise InputError(f"{where}: ALT equals REF")
        group = _int(row["group"], path, line, "group")
        if group < 0:
            raise InputError(f"{where}: group must be >= 0")
        by_pos = groups.setdefault(gene.name, {}).setdefault(group, {})
        if pos in by_pos:
            raise InputError(f"{where}: group {group} of gene {gene.name} "
                             f"already has a variant at {gene.chrom}:{pos} "
                             f"(line {by_pos[pos].line})")
        by_pos[pos] = Variant(gene.chrom, pos, ref, alt, line)

    haplotypes = {}
    for name in genes:
        if name not in groups:
            continue
        by_group = groups[name]
        if min(by_group) == 1:
            by_group[0] = {}  # the reference sequence
        ids = sorted(by_group)
        if ids != list(range(len(ids))):
            missing = sorted(set(range(ids[-1] + 1)) - set(ids))
            raise InputError(
                f"{path}: gene {name}: group ids must run 0, 1, 2, ... "
                f"(or start at 1, making group 0 the reference); missing "
                f"{', '.join(map(str, missing))}"
            )
        haplotypes[name] = []
        for g in ids:
            variants = tuple(v for _, v in sorted(by_group[g].items()))
            for a, b in zip(variants, variants[1:]):
                if a.pos + len(a.ref) > b.pos:
                    raise InputError(
                        f"{path}:{b.line}: overlaps the variant on line "
                        f"{a.line} (group {g} of gene {name})"
                    )
            haplotypes[name].append(Haplotype(name, g, variants))
    return haplotypes
