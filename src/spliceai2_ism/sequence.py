"""Model input windows for gene haplotypes: FASTA extraction, variants
(SNVs, MNVs, insertions, deletions), strands and coordinates.

Sequences are numpy arrays of base codes: 0-3 = A, C, G, T, plus
``encode.N_CODE`` and ``encode.PAD_CODE`` (past a chromosome end).

Coordinates. A variant REF -> ALT pairs bases from the left: the first
min(len(REF), len(ALT)) bases are kept or substituted, extra ALT bases are
inserted after the last REF base, extra REF bases are deleted. Every
haplotype base has
  * a haplotype position: 1-based, on the haplotype's own chromosome (the
    reference chromosome with all of the haplotype's variants applied), so
    equal to the reference position up to the first indel and shifted by
    every indel before it;
  * a reference position: its own for reference bases (kept or
    substituted), none for inserted bases. Deleted bases are absent.
Internally, inserted bases carry the position of the reference base they
follow and an insertion offset 1, 2, ... (0 for reference bases).

The gene body of a haplotype is its bases at reference positions
TX_START..TX_END, plus bases inserted after TX_START..TX_END - 1. The model
predicts an output region starting at the gene body, in transcript
orientation (minus-strand genes are reverse-complemented), ``crop_len // 2``
nt of flank on both sides. The output region is the gene body rounded up to
the length bucket (and to at least the number of junction candidate
sites); the extra positions are haplotype sequence after the gene's 3' end,
predicted but never reported or saturated.
"""
from dataclasses import dataclass, field

import numpy as np

from spliceai2_ism.encode import N_CODE, PAD_CODE
from spliceai2_ism.inputs import BASES, InputError

_CODES = np.full(256, N_CODE, dtype=np.uint8)
for _i, _b in enumerate(BASES):
    _CODES[ord(_b)] = _CODES[ord(_b.lower())] = _i
_COMPLEMENT = np.array([3, 2, 1, 0, N_CODE, PAD_CODE], dtype=np.uint8)


def to_codes(seq):
    """Base codes of a string; lowercase counts as uppercase, anything other
    than ACGT is N."""
    return _CODES[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def to_string(codes):
    return "".join(BASES[c] if c < 4 else "N" for c in codes)


def reverse_complement(codes):
    return _COMPLEMENT[codes[::-1]]


def output_length(gene_len, bucket, align):
    """Gene length rounded up to a multiple of ``bucket``."""
    if bucket % align:
        raise ValueError(f"length bucket must be a multiple of {align}")
    return -(-gene_len // bucket) * bucket


@dataclass
class Region:
    """Forward-strand reference sequence around a gene, long enough for any
    of its haplotypes' model inputs."""

    gene: object         # inputs.Gene
    start: int           # 1-based genomic position of codes[0]
    codes: np.ndarray
    crop_len: int
    min_out_len: int     # output regions cover at least this many positions
    length_bucket: int
    align: int
    out_lens: set = field(default_factory=set)  # of its haplotypes

    def out_len(self, gene_len):
        return output_length(max(gene_len, self.min_out_len),
                             self.length_bucket, self.align)


def read_region(fasta, gene, haplotypes, crop_len, min_out_len,
                length_bucket, align):
    """``fasta``: mapping chrom -> sequence supporting len() and 0-based
    slicing to str (pyfaidx.Fasta with as_raw=True, or plain strings).

    The flank covers crop_len // 2 nt of model input, the output region's
    extension past the gene, and the bases any haplotype deletes from it.
    """
    deleted = max((sum(max(0, len(v.ref) - len(v.alt)) for v in h.variants)
                   for h in haplotypes), default=0)
    longest_ref = max((len(v.ref) for h in haplotypes for v in h.variants),
                      default=1)
    flank = (crop_len // 2 + length_bucket + min_out_len + deleted
             + longest_ref)
    try:
        chrom = fasta[gene.chrom]
    except KeyError:
        raise InputError(f"gene {gene.name}: chromosome {gene.chrom!r} is "
                         f"not in the FASTA") from None
    if gene.end > len(chrom):
        raise InputError(f"gene {gene.name}: end {gene.end} is past the end "
                         f"of {gene.chrom} ({len(chrom)} nt)")
    start, end = gene.start - flank, gene.end + flank
    lo, hi = max(start, 1), min(end, len(chrom))
    codes = np.full(end - start + 1, PAD_CODE, dtype=np.uint8)
    codes[lo - start:hi - start + 1] = to_codes(str(chrom[lo - 1:hi]))
    return Region(gene, start, codes, crop_len, min_out_len, length_bucket,
                  align)


def haplotype_sequence(region, variants):
    """The haplotype over the region, forward strand: base codes; reference
    position, insertion offset and haplotype position per base; and the
    variants outside the region (too far from the gene to affect it)."""
    codes, start, n = region.codes, region.start, len(region.codes)
    parts, outside, cur = [], [], 0
    # indels before the region still shift haplotype positions
    shift = sum(len(v.alt) - len(v.ref) for v in variants if v.pos < start)

    def keep(i, j):  # reference bases region[i:j]
        parts.append((codes[i:j], start + np.arange(i, j),
                      np.zeros(j - i, np.int32)))

    for v in sorted(variants, key=lambda v: v.pos):
        i = v.pos - start
        j = i + len(v.ref)
        if i < 0 or j > n:
            outside.append(v)
            continue
        if i < cur:
            raise InputError(f"variant TSV line {v.line}: overlaps another "
                             f"variant of the haplotype")
        if to_string(codes[i:j]) != v.ref:
            raise InputError(f"variant TSV line {v.line}: REF {v.ref} at "
                             f"{v.chrom}:{v.pos}, but the FASTA has "
                             f"{to_string(codes[i:j])}")
        keep(cur, i)
        alt = to_codes(v.alt)
        k = min(len(v.ref), len(v.alt))
        parts.append((alt[:k], v.pos + np.arange(k), np.zeros(k, np.int32)))
        if len(alt) > k:  # inserted after the last REF base
            parts.append((alt[k:],
                          np.full(len(alt) - k, v.pos + len(v.ref) - 1),
                          np.arange(1, len(alt) - k + 1, dtype=np.int32)))
        cur = j
    keep(cur, n)
    hap_codes, pos, ins = (np.concatenate(t) for t in zip(*parts))
    hap_pos = start + shift + np.arange(len(hap_codes), dtype=np.int64)
    return hap_codes, pos.astype(np.int64), ins, hap_pos, outside


@dataclass(frozen=True)
class HaplotypeWindow:
    """Model input of one haplotype and the coordinates of its outputs."""

    gene: object         # inputs.Gene
    codes_t: np.ndarray  # transcript orientation, out_len + crop_len codes
    gene_len: int        # haplotype bases in the gene body
    out_len: int
    crop_len: int
    reference_pos: np.ndarray  # (gene_len,) per output index; 0: inserted
    haplotype_pos: np.ndarray  # (gene_len,) per output index
    outside: tuple = ()  # variants ignored: outside the region

    def forward_bases(self, codes):
        """Forward-strand base codes of transcript-orientation codes."""
        codes = np.asarray(codes)
        return codes if self.gene.strand == "+" else 3 - codes


def haplotype_window(region, variants):
    gene, half = region.gene, region.crop_len // 2
    codes, pos, ins, hap_pos, outside = haplotype_sequence(region, variants)
    in_gene = np.where(ins == 0,
                       (pos >= gene.start) & (pos <= gene.end),
                       (pos >= gene.start) & (pos < gene.end))
    body = np.nonzero(in_gene)[0]
    if len(body) == 0:
        raise InputError(f"gene {gene.name}: a haplotype deletes the whole "
                         f"gene")
    b0, b1 = int(body[0]), int(body[-1]) + 1
    gene_len = b1 - b0
    out_len = region.out_len(gene_len)
    if gene.strand == "+":
        lo, hi = b0 - half, b0 + out_len + half
    else:
        lo, hi = b1 - out_len - half, b1 + half
    if lo < 0 or hi > len(codes):  # read_region's flank prevents this
        raise RuntimeError(f"gene {gene.name}: region too short")
    ref_pos = np.where(ins == 0, pos, 0)[b0:b1]
    hap_pos = hap_pos[b0:b1]
    if gene.strand == "+":
        codes_t = codes[lo:hi]
    else:  # output index 0 is the gene body's last forward-strand base
        codes_t = reverse_complement(codes[lo:hi])
        ref_pos, hap_pos = ref_pos[::-1], hap_pos[::-1]
    return HaplotypeWindow(gene, codes_t, gene_len, out_len, region.crop_len,
                           np.ascontiguousarray(ref_pos),
                           np.ascontiguousarray(hap_pos), tuple(outside))


def saturation(window):
    """Every SNV of the haplotype's gene body: output indices and new bases
    (transcript orientation), three per base that is A, C, G or T."""
    half = window.crop_len // 2
    base = window.codes_t[half:half + window.gene_len]
    idx = np.nonzero(base < 4)[0]
    pos = np.repeat(idx, 3)
    alt = (np.repeat(base[idx], 3).astype(np.int64)
           + np.tile([1, 2, 3], len(idx))) % 4
    return pos.astype(np.int64), alt
