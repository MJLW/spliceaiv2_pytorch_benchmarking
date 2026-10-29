"""Parquet outputs: four long-format tables.

predictions_sites.parquet, one row per gene-body base of each haplotype:
  gene, group, reference_pos, haplotype_pos, donor_prob, acceptor_prob
predictions_junctions.parquet, one row per donor-acceptor pair of each
haplotype's candidate sites inside the gene:
  gene, group, donor_reference_pos, donor_haplotype_pos,
  acceptor_reference_pos, acceptor_haplotype_pos, prob
deltas_sites.parquet, per haplotype and SNV, the bases where the donor or
acceptor probability changes by at least --min-delta:
  gene, group, snv_reference_pos, snv_haplotype_pos, snv_ref, snv_alt,
  reference_pos, haplotype_pos, donor_delta, acceptor_delta
deltas_junctions.parquet, per haplotype and SNV, the junctions whose
probability changes by at least --min-delta:
  gene, group, snv_reference_pos, snv_haplotype_pos, snv_ref, snv_alt,
  donor_reference_pos, donor_haplotype_pos, acceptor_reference_pos,
  acceptor_haplotype_pos, delta

group is the variant TSV's group id, or "reference" for the gene's
reference sequence. Positions are 1-based, forward strand. reference_pos is the GRCh38
position, null for bases the haplotype inserts; haplotype_pos is the
position on the haplotype's own chromosome (see sequence.py). snv_ref is
the haplotype's base, so at a haplotype variant it is the ALT allele.
delta = SNV probability - haplotype probability.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spliceai2_ism.inputs import BASES

_HAPLOTYPE = [("gene", pa.string()), ("group", pa.string())]
_SNV = [("snv_reference_pos", pa.int64()), ("snv_haplotype_pos", pa.int64()),
        ("snv_ref", pa.string()), ("snv_alt", pa.string())]
_SITE = [("reference_pos", pa.int64()), ("haplotype_pos", pa.int64())]
_JUNCTION = [
    ("donor_reference_pos", pa.int64()), ("donor_haplotype_pos", pa.int64()),
    ("acceptor_reference_pos", pa.int64()),
    ("acceptor_haplotype_pos", pa.int64()),
]
SCHEMAS = {
    "predictions_sites": pa.schema(
        [*_HAPLOTYPE, *_SITE, ("donor_prob", pa.float32()),
         ("acceptor_prob", pa.float32())]),
    "predictions_junctions": pa.schema(
        [*_HAPLOTYPE, *_JUNCTION, ("prob", pa.float32())]),
    "deltas_sites": pa.schema(
        [*_HAPLOTYPE, *_SNV, *_SITE, ("donor_delta", pa.float32()),
         ("acceptor_delta", pa.float32())]),
    "deltas_junctions": pa.schema(
        [*_HAPLOTYPE, *_SNV, *_JUNCTION, ("delta", pa.float32())]),
}


def _labels(codes, labels):
    """String column from integer codes, built as a dictionary array."""
    return pa.DictionaryArray.from_arrays(
        pa.array(np.asarray(codes, dtype=np.int32)), pa.array(list(labels))
    ).cast(pa.string())


def _haplotype(window, group, n):
    return [_labels(np.zeros(n, dtype=np.int32), [window.gene.name]),
            _labels(np.zeros(n, dtype=np.int32), [group])]


def _positions(window, idx):
    """reference_pos (null for inserted bases) and haplotype_pos columns of
    output indices."""
    idx = np.asarray(idx, dtype=np.int64)
    ref = window.reference_pos[idx]
    return [pa.array(ref, mask=ref == 0), pa.array(window.haplotype_pos[idx])]


def _f32(x):
    return pa.array(np.asarray(x, dtype=np.float32))


class Writer:
    """Writes the four tables; ``write_*`` take a
    ``sequence.HaplotypeWindow`` and device tensors or numpy arrays of
    output indices and transcript-orientation base codes."""

    def __init__(self, out_dir):
        self.files = {
            name: pq.ParquetWriter(out_dir / f"{name}.parquet", schema,
                                   compression="zstd")
            for name, schema in SCHEMAS.items()
        }
        self.rows = dict.fromkeys(SCHEMAS, 0)

    def close(self):
        for f in self.files.values():
            f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _write(self, name, columns):
        table = pa.Table.from_arrays(columns, schema=SCHEMAS[name])
        if table.num_rows:
            self.files[name].write_table(table)
            self.rows[name] += table.num_rows

    def write_predictions(self, window, group, probs, donor, acceptor, jxn):
        """probs (2, gene_len): donor and acceptor; donor, acceptor, jxn
        (P,): junction pairs."""
        probs, donor, acceptor, jxn = (
            _numpy(t) for t in (probs, donor, acceptor, jxn)
        )
        n = probs.shape[1]
        self._write("predictions_sites", [
            *_haplotype(window, group, n), *_positions(window, np.arange(n)),
            _f32(probs[0]), _f32(probs[1]),
        ])
        self._write("predictions_junctions", [
            *_haplotype(window, group, len(jxn)),
            *_positions(window, donor), *_positions(window, acceptor),
            _f32(jxn),
        ])

    def _snv(self, window, snv_idx, snv_ref, snv_alt):
        return [*_positions(window, snv_idx),
                _labels(window.forward_bases(snv_ref), BASES),
                _labels(window.forward_bases(snv_alt), BASES)]

    def write_site_deltas(self, window, group, snv_idx, snv_ref, snv_alt,
                          site, donor_delta, acceptor_delta):
        """Per row: the SNV's output index and bases (transcript codes),
        the site's output index and both deltas."""
        cols = [_numpy(t) for t in (snv_idx, snv_ref, snv_alt, site,
                                    donor_delta, acceptor_delta)]
        snv_idx, snv_ref, snv_alt, site, dd, da = cols
        self._write("deltas_sites", [
            *_haplotype(window, group, len(site)),
            *self._snv(window, snv_idx, snv_ref, snv_alt),
            *_positions(window, site), _f32(dd), _f32(da),
        ])

    def write_junction_deltas(self, window, group, snv_idx, snv_ref, snv_alt,
                              donor, acceptor, delta):
        """Per row: the SNV's output index and bases (transcript codes),
        the junction's donor and acceptor output indices and the delta."""
        cols = [_numpy(t) for t in (snv_idx, snv_ref, snv_alt, donor,
                                    acceptor, delta)]
        snv_idx, snv_ref, snv_alt, donor, acceptor, delta = cols
        self._write("deltas_junctions", [
            *_haplotype(window, group, len(delta)),
            *self._snv(window, snv_idx, snv_ref, snv_alt),
            *_positions(window, donor), *_positions(window, acceptor),
            _f32(delta),
        ])


def _numpy(t):
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
