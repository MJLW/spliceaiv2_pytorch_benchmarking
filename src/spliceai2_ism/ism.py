"""In-silico saturation mutagenesis (ISM) of gene haplotypes with SpliceAI2.

For every haplotype (gene + group of variants from the variant TSV; SNVs,
MNVs and indels) this runs one full prediction of the haplotype sequence,
then scores every possible SNV in the haplotype's gene body (three per base)
against it with
``IncrementalEnsemble``: each SNV recomputes only the part of the network it
can change. Both models of the released ensemble are used, with FP16
autocast and torch.compile, as the reference scorer.

Outputs, in --out-dir (see output.py for the columns):
  predictions_sites.parquet      every haplotype's donor and acceptor
                                 probabilities, per gene-body base;
  predictions_junctions.parquet  its junction probabilities;
  deltas_sites.parquet           per haplotype and SNV, the signed change
  deltas_junctions.parquet       in those, where |delta| >= --min-delta;
  run.json                       configuration, environment and timings.
"""
import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pyfaidx
import torch

from spliceai2_ism.batching import choose_batch_size
from spliceai2_ism.deltas import delta_1d, delta_2d, haplotype_rows
from spliceai2_ism.encode import one_hot_codes
from spliceai2_ism.incremental import IncrementalEnsemble
from spliceai2_ism.inputs import InputError, read_haplotypes, read_regions
from spliceai2_ism.output import SCHEMAS, Writer
from spliceai2_ism.runtime import (
    environment_info,
    forbid_eager_fallback,
    inference,
    raise_recompile_limit,
)
from spliceai2_ism.sequence import haplotype_window, read_region, saturation
from spliceai2_ism.weights import load_released_models


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SNV saturation mutagenesis of gene haplotypes with "
                    "SpliceAI2"
    )
    p.add_argument("--variants", type=Path, required=True,
                   help="TSV: CHROM, POS, REF, ALT, group, gene")
    p.add_argument("--regions", type=Path, required=True,
                   help="TSV: NAME, CHROM, STRAND, TX_START, TX_END "
                        "(1-based, inclusive)")
    p.add_argument("--fasta", type=Path, required=True,
                   help="reference genome FASTA (GRCh38), indexed or "
                        "indexable by pyfaidx")
    p.add_argument("--out-dir", type=Path,
                   help="default: results/ism-<timestamp>")
    p.add_argument("--genes",
                   help="optional: only these genes (comma-separated) of the "
                        "variant TSV, e.g. to split a run over a job array; "
                        "by default every gene in the variant TSV runs")
    p.add_argument("--variant-batch-size", type=_batch_size, default="auto",
                   help="SNVs per incremental batch, or 'auto' (default): "
                        "the largest that fits in GPU memory, measured on "
                        "the longest gene")
    p.add_argument("--out-2d-len", type=int, default=1024,
                   help="candidate sites per prediction for the junction head")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="smallest |change in probability| written")
    p.add_argument("--length-bucket", type=int, default=16_384,
                   help="gene lengths are rounded up to a multiple of this "
                        "(fewer compiled graphs); the extra positions are "
                        "predicted but not reported")
    p.add_argument("--debug", action="store_true",
                   help="run eagerly, without torch.compile")
    return p.parse_args(argv)


def _batch_size(value):
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError("a positive integer or 'auto'")
    return n


class Stages:
    """Haplotype pass and SNV batch, compiled unless ``compile_`` is false."""

    def __init__(self, inc, compile_):
        def reference(x, valid_len):
            cache, hap_ss = inc.reference_1d(x)
            idxs, jxn = inc.head_2d(cache[1], hap_ss, valid_len=valid_len)
            return cache, hap_ss, idxs, jxn

        def variants(cache, hap_scores, pos, alt, valid_len):
            z_var, win = inc.variants_1d(cache, pos, alt)
            scores = inc.patch_track(hap_scores, pos,
                                     win.amax(dim=1, keepdim=True))
            idxs, jxn = inc.head_2d(cache[1], scores, pos, z_var,
                                    valid_len=valid_len)
            return win, idxs, jxn

        self.reference, self.variants = reference, variants
        if compile_:
            self.reference = torch.compile(reference, dynamic=False)
            self.variants = torch.compile(variants, dynamic=False)


def run_haplotype(inc, stages, region, hap, writer, device, batch_size,
                  min_delta, log=print):
    """Predicts one haplotype and all its SNVs; returns stats."""
    gene = region.gene
    t0 = time.perf_counter()
    window = haplotype_window(region, hap.variants)
    for v in window.outside:
        warnings.warn(f"variant TSV line {v.line}: {v.chrom}:{v.pos} is too "
                      f"far from gene {gene.name} to affect it; ignored")
    codes_t = window.codes_t
    x = torch.from_numpy(one_hot_codes(codes_t))[None].to(device)
    # candidate sites: the gene body, or for genes shorter than out_2d_len
    # the first out_2d_len output positions (the rest is sequence after the
    # gene; junctions there are not reported)
    valid_len = torch.tensor([max(window.gene_len, inc.out_2d_len)],
                             device=device)

    with inference(device):
        cache, hap_ss, hap_idx, hap_jxn = stages.reference(x, valid_len)
        hap_idx, hap_jxn = hap_idx[0], hap_jxn[0]
        writer.write_predictions(
            window, hap.group, *haplotype_rows(hap_ss, hap_idx, hap_jxn,
                                               window.gene_len)
        )
        hap_scores = hap_ss.amax(dim=1, keepdim=True)

    snv_pos, snv_alt = saturation(window)
    half = inc.crop_len // 2
    snv_ref = codes_t[half + snv_pos]
    n_snvs, t_log = len(snv_pos), time.perf_counter()
    hap_state = (cache, hap_ss, hap_scores, hap_idx, hap_jxn, valid_len)
    s = 0
    while s < n_snvs:
        pos, alt = snv_pos[s:s + batch_size], snv_alt[s:s + batch_size]
        n = len(pos)
        if n < batch_size:  # same shape for every batch: repeat the last SNV
            pos = np.concatenate([pos, np.repeat(pos[-1:], batch_size - n)])
            alt = np.concatenate([alt, np.repeat(alt[-1:], batch_size - n)])
        try:
            out = _score_batch(inc, stages, hap_state, window.gene_len, pos,
                               alt, device, min_delta)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            out = None
        if out is None:  # outside the except block, so its tensors are freed
            torch.cuda.empty_cache()
            batch_size //= 2
            log(f"    out of GPU memory; batch size now {batch_size}",
                flush=True)
            continue
        snv1, site, d_donor, d_acceptor, snv2, donor, acceptor, d_jxn = out
        real = snv1 < n  # drop the padding SNVs
        i = snv1[real] + s
        writer.write_site_deltas(
            window, hap.group, snv_pos[i], snv_ref[i], snv_alt[i],
            site[real], d_donor[real], d_acceptor[real],
        )
        real = snv2 < n
        i = snv2[real] + s
        writer.write_junction_deltas(
            window, hap.group, snv_pos[i], snv_ref[i], snv_alt[i],
            donor[real], acceptor[real], d_jxn[real],
        )
        now = time.perf_counter()
        if now - t_log > 60:
            done = s + n
            rate = done / (now - t0)
            log(f"    {done:,}/{n_snvs:,} SNVs, {rate:,.0f} SNVs/s, "
                f"ETA {(n_snvs - done) / rate / 60:.1f} min", flush=True)
            t_log = now
        s += n
    del cache, hap_state
    return {"gene": gene.name, "group": hap.group,
            "variants": len(hap.variants) - len(window.outside),
            "gene_len": window.gene_len, "snvs": n_snvs,
            "seconds": time.perf_counter() - t0, "batch_size": batch_size}


def _score_batch(inc, stages, hap_state, gene_len, pos, alt, device,
                 min_delta):
    """One SNV batch: sparse 1D and 2D deltas as numpy arrays."""
    cache, hap_ss, hap_scores, hap_idx, hap_jxn, valid_len = hap_state
    pos_t = torch.from_numpy(pos).to(device)
    alt_t = torch.from_numpy(alt).to(device)
    with inference(device):
        win, v_idx, v_jxn = stages.variants(cache, hap_scores, pos_t, alt_t,
                                            valid_len)
        r1 = delta_1d(inc, hap_ss, pos_t, win, gene_len, min_delta)
        r2 = delta_2d(hap_idx, hap_jxn, v_idx, v_jxn, min_delta,
                      gene_len=gene_len)
    return tuple(t.cpu().numpy() for t in (*r1, *r2))


def prepare_regions(inc, genes, haplotypes, fasta, length_bucket):
    """Reference regions of ``genes`` (list of Gene). Builds every
    haplotype once, so REF mismatches and other input errors fail before
    any compute, and records the output lengths (compiled shapes)."""
    regions = []
    for gene in genes:
        haps = haplotypes[gene.name]
        # output regions of at least out_2d_len positions, so short genes
        # have enough candidate sites
        region = read_region(fasta, gene, haps, inc.crop_len, inc.out_2d_len,
                             length_bucket, inc.align)
        for hap in haps:
            region.out_lens.add(haplotype_window(region, hap.variants).out_len)
        regions.append(region)
    return regions


def run_ism(inc, regions, haplotypes, writer, device, *, compile_=True,
            batch_size=100, min_delta=1e-3, log=print):
    """Runs every haplotype of the genes in ``regions``; returns stats.

    If a batch runs out of GPU memory, the batch size is halved for it and
    every later batch."""
    stages = Stages(inc, compile_)
    if compile_:
        # one graph per output length, and per batch size after halvings
        n_lens = len(set().union(*(r.out_lens for r in regions)))
        raise_recompile_limit(n_lens * (batch_size.bit_length() + 1))
        forbid_eager_fallback()
    stats = []
    for region in regions:
        gene = region.gene
        haps = haplotypes[gene.name]
        log(f"{gene.name} ({gene.chrom}:{gene.start:,}-{gene.end:,} "
            f"{gene.strand}, {gene.length:,} nt): {len(haps)} haplotypes",
            flush=True)
        for hap in haps:
            st = run_haplotype(inc, stages, region, hap, writer, device,
                               batch_size, min_delta, log)
            batch_size = st["batch_size"]
            log(f"  group {hap.group}: {st['variants']} variants, "
                f"{st['gene_len']:,} nt gene body, "
                f"{st['snvs']:,} SNVs in {st['seconds']:.1f} s "
                f"({st['snvs'] / st['seconds']:,.0f} SNVs/s)", flush=True)
            stats.append(st)
    return stats


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        sys.exit("a CUDA GPU is required")
    device = torch.device("cuda")
    out_dir = args.out_dir or Path("results") / (
        "ism-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    for name in SCHEMAS:
        if (out_dir / f"{name}.parquet").exists():
            sys.exit(f"{out_dir / name}.parquet exists; choose another "
                     f"--out-dir")

    try:
        genes = read_regions(args.regions)
        haplotypes = read_haplotypes(args.variants, genes)
    except InputError as e:
        sys.exit(f"error: {e}")
    # the genes to run are those in the variant TSV; the region file only
    # supplies their coordinates and may list many more
    if args.genes:
        wanted = args.genes.split(",")
        unknown = [g for g in wanted if g not in genes]
        if unknown:
            sys.exit(f"--genes: not in the region file: {', '.join(unknown)}")
        for g in [g for g in wanted if g not in haplotypes]:
            warnings.warn(f"--genes: {g} has no variants in the TSV; skipped")
        selected = [genes[g] for g in wanted if g in haplotypes]
    else:
        selected = [g for g in genes.values() if g.name in haplotypes]
    if not selected:
        sys.exit("no genes to run")

    t0 = time.perf_counter()
    inc = IncrementalEnsemble(load_released_models(), args.out_2d_len)
    inc = inc.eval().to(device)
    load_s = time.perf_counter() - t0
    if args.length_bucket % inc.align:
        sys.exit(f"--length-bucket must be a multiple of {inc.align}")
    fasta = pyfaidx.Fasta(str(args.fasta), as_raw=True,
                          sequence_always_upper=True)
    try:
        regions = prepare_regions(inc, selected, haplotypes, fasta,
                                  args.length_bucket)
    except InputError as e:
        sys.exit(f"error: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(selected)} genes, "
          f"{sum(len(haplotypes[g.name]) for g in selected)} haplotypes; "
          f"writing to {out_dir}/", flush=True)
    t0 = time.perf_counter()
    batch_size, batch_info = args.variant_batch_size, None
    if batch_size == "auto":
        batch_size, batch_info = choose_batch_size(
            inc, Stages(inc, compile_=False), regions, device
        )
        print(f"batch size {batch_size} (auto: "
              f"{batch_info['bytes_per_snv'] / 2**20:,.0f} MiB per SNV + "
              f"{batch_info['fixed_bytes'] / 2**30:.2f} GiB, measured on "
              f"{batch_info['measured_on']}; "
              f"{batch_info['capacity_bytes'] / 2**30:.1f} GiB available)",
              flush=True)
    with Writer(out_dir) as writer:
        stats = run_ism(
            inc, regions, haplotypes, writer, device,
            compile_=not args.debug, batch_size=batch_size,
            min_delta=args.min_delta,
        )
        rows = dict(writer.rows)
    run_s = time.perf_counter() - t0
    config = {k: str(v) if isinstance(v, Path) else v
              for k, v in vars(args).items()}
    (out_dir / "run.json").write_text(json.dumps({
        "config": config, "env": environment_info(),
        "model": {"crop_len": inc.crop_len, "reach_nt": inc.reach,
                  "num_models": len(inc.models)},
        "batch_size": {"initial": batch_size,
                       "final": stats[-1]["batch_size"], "auto": batch_info},
        "load_s": load_s, "run_s": run_s,
        "rows": rows, "haplotypes": stats,
    }, indent=2))
    print(f"done in {_duration(load_s + run_s)} ({_duration(run_s)} after "
          f"loading the models); rows written: "
          + ", ".join(f"{n:,} {name}" for name, n in rows.items())
          + f" in {out_dir}/")


def _duration(seconds):
    """e.g. '2 h 05 min', '3 min 12 s', '41.3 s'."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, s = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {s:02d} s"
    return f"{minutes // 60} h {minutes % 60:02d} min"


if __name__ == "__main__":
    main()
