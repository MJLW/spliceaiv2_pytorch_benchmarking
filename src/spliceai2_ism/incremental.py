"""Incremental SpliceAI2 inference for single-base substitutions.

A substitution changes the network's activations only inside a cone that
widens block by block, to about ±56.5k nt after the last block of the
released models. ``IncrementalEnsemble`` runs the reference sequence once,
keeps every block's pooled output, and computes a variant by recomputing only
that cone; everything outside it is taken from the reference. Every output
position and every candidate site of the junction head is covered, so the
result equals a full forward pass on the variant sequence up to
floating-point rounding.

Three rewrites keep the per-variant work small; all are exact in exact
arithmetic:
  * the splice-site heads are applied before the linear upsampling (both are
    linear and the interpolation weights sum to one), so per-block 1D
    contributions are kept at pooled resolution, and the variant's 1D track
    is only stitched together inside the cone;
  * junction-head features at the top-k sites are interpolated directly from
    the pooled block outputs instead of from full-length upsampled features;
  * each block's input window carries one token-mixer radius of context per
    side beyond the outputs that can change; the mixer only produces those
    outputs, and the pointwise rest of the block (norm, FFN, residual) only
    runs on them.

``Optimizations`` switches further rewrites, also exact in exact arithmetic
and all on by default (each can be turned off for ablations):
  * ``stencil_resample``: a linear upsampling by p followed by the next
    block's average pooling is a fixed 3-tap (same pool) or stride-2 4-tap
    (pool doubles) filter at pooled resolution; it replaces materialising
    the full-resolution activations between blocks;
  * ``grouped_head_upsample``: splice-site head outputs of blocks that share
    a pooling factor (and of both models) are summed before one upsampling
    per factor, instead of one per block and model;
  * ``reference_depthwise``: the reference pass runs the token mixer as
    shifted products (``_depthwise_same``) instead of PyTorch's generic
    dilated depthwise kernel;
  * ``padded_ffn``: the FFN of windows whose width is not a multiple of 8
    runs on zero-padded columns (dropped afterwards), so the 1x1
    convolutions get aligned GEMM kernels.

Coordinates: full-resolution positions are model-input positions (0..N-1,
N = L + crop_len). A substitution at input position p is handled relative to
its anchor ``a`` = p rounded down to a multiple of ``align``, the deepest
pooling factor, so every window below covers whole pooling bins at every
resolution and its shape does not depend on p.
"""
from dataclasses import dataclass, fields, replace

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Optimizations:
    """Exact rewrites of the incremental path (see the module docstring)."""

    stencil_resample: bool = True
    grouped_head_upsample: bool = True
    reference_depthwise: bool = True
    padded_ffn: bool = True

    @classmethod
    def names(cls):
        return [f.name for f in fields(cls)]

    @classmethod
    def none(cls):
        return cls(**{name: False for name in cls.names()})

    @classmethod
    def parse(cls, spec):
        """From a comma-separated spec: ``all``, ``none``, names to enable,
        ``-name`` to disable; it starts from ``none`` unless the first item
        is ``all``. E.g. ``all,-padded_ffn`` or ``stencil_resample``."""
        items = [s.strip() for s in spec.split(",") if s.strip()]
        opts = cls() if items[:1] == ["all"] else cls.none()
        for item in items:
            if item in ("all", "none"):
                continue
            name = item.lstrip("-")
            if name not in cls.names():
                raise ValueError(f"unknown optimization {name!r}; choose "
                                 f"from {', '.join(cls.names())}")
            opts = replace(opts, **{name: not item.startswith("-")})
        return opts

    def spec(self):
        on = [n for n in self.names() if getattr(self, n)]
        if len(on) == len(self.names()):
            return "all"
        return ",".join(on) or "none"


def _acc(t):
    """t in its accumulation dtype: at least fp32 (fp64 stays fp64)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


def _first(v):
    return v[0] if isinstance(v, (tuple, list)) else v


def _ceil_to(x, m):
    return -(-x // m) * m


def _radius(conv):
    return (_first(conv.kernel_size) - 1) // 2 * _first(conv.dilation)


def _depthwise_window(conv, x, start, width):
    """Outputs ``start:start + width`` of a depthwise 'same' Conv1d on x, as
    a sum of shifted products accumulated in fp32; the slice needs the
    conv's radius of input on both sides, so no padding is involved. Under
    autocast the result is rounded to the autocast dtype, like the conv it
    replaces.

    Used on the short variant windows: cuDNN's depthwise kernel segfaulted
    there once the same layer had run on the full-length reference (torch
    2.14.1, cuDNN 9.24). Under torch.compile this fuses into one kernel.
    """
    k, d, r = _first(conv.kernel_size), _first(conv.dilation), _radius(conv)
    if start < r or start + width + r > x.shape[2]:
        raise ValueError("output slice needs a radius of input on each side")
    xs = _acc(x[..., start - r:start + width + r])
    w = _acc(conv.weight)
    out = sum(w[:, 0, j, None] * xs[..., j * d:j * d + width] for j in range(k))
    if conv.bias is not None:
        out = out + _acc(conv.bias)[:, None]
    device = x.device.type
    if torch.is_autocast_enabled(device):
        return out.to(torch.get_autocast_dtype(device))
    return out.to(x.dtype)


def _depthwise_same(conv, x):
    """``conv(x)`` for a depthwise 'same' Conv1d, via _depthwise_window."""
    r = _radius(conv)
    return _depthwise_window(conv, F.pad(x, (r, r)), r, x.shape[2])


def _ffn(block, x, pad_to=1):
    """The block's pointwise FFN branch on x. With ``pad_to`` > 1, x is
    zero-padded on the right to a multiple of it and the padded columns are
    dropped again: every column is computed independently, so the real ones
    are unchanged, and the 1x1 convolutions get aligned GEMM kernels."""
    width = x.shape[-1]
    pad = -width % pad_to
    if pad:
        x = F.pad(x, (0, pad))
    out = block._ffn_proj_down(
        block._ffn_act(block._ffn_proj_up(block._ffn_norm(x)))
    )
    return out[..., :width] if pad else out


def _block_window(block, x, start, width, pad_to=1):
    """Outputs ``start:start + width`` of MetaFormer.forward on x. The token
    mixer only produces those outputs, and the pointwise rest of the block
    only runs on them."""
    r = _radius(block._token_mixer)
    x = x[..., start - r:start + width + r]
    out = x[..., r:r + width] + _depthwise_window(
        block._token_mixer, block._token_norm(x), r, width
    )
    return out + _ffn(block, out, pad_to)


def _block_same(block, x, pad_to=1):
    """MetaFormer.forward on the whole of x, with the token mixer as shifted
    products (``_depthwise_same``)."""
    out = x + _depthwise_same(block._token_mixer, block._token_norm(x))
    return out + _ffn(block, out, pad_to)


def _pool_taps(src_pool, pool):
    """Filter weights and stride equivalent to linear upsampling by
    ``src_pool`` (nn.Upsample, align_corners=False) followed by average
    pooling by ``pool``, at the coarser resolution.

    For even p, upsampling by p and pooling by p gives each output bin i
    1/8 of bins i-1 and i+1 and 3/4 of bin i; pooling by 2p averages two
    neighbouring such bins. Upsampling by 1 is the identity.
    """
    if pool not in (src_pool, 2 * src_pool):
        raise ValueError(f"pooling {src_pool} -> {pool}: must stay or double")
    stride = pool // src_pool
    if src_pool == 1:
        return ((1.0,) if stride == 1 else (0.5, 0.5)), stride
    if src_pool % 2:
        raise ValueError(f"pooling factor {src_pool} must be 1 or even")
    if stride == 1:
        return (0.125, 0.75, 0.125), 1
    return (0.0625, 0.4375, 0.4375, 0.0625), 2


def _strided_taps(x, taps, stride, start, n):
    """out[..., i] = sum_j taps[j] * x[..., start + stride * i + j] for
    i < n, accumulated in fp32 and returned in x's dtype."""
    xf = _acc(x)
    end = start + stride * (n - 1) + 1
    out = sum(w * xf[..., start + j:end + j:stride]
              for j, w in enumerate(taps))
    return out.to(x.dtype)


def _halo(taps, stride):
    """Source bins a stencil output needs on each side of its own."""
    return (len(taps) - stride) // 2


def _resample_window(ctx, src_pool, pool, n):
    """The next block's pooled input from a gathered window ``ctx`` that has
    one extra source bin per side: equals
    ``AvgPool1d(pool)(Upsample(src_pool)(ctx)[..., src_pool:src_pool + n * pool])``
    without the full-resolution intermediate. Every tap lies inside ctx, so
    the Upsample's edge clamping never applies."""
    taps, stride = _pool_taps(src_pool, pool)
    return _strided_taps(ctx, taps, stride, 1 - _halo(taps, stride), n)


def _resample_full(z, src_pool, pool):
    """``AvgPool1d(pool)(Upsample(src_pool)(z))`` over a whole sequence. The
    Upsample clamps source indices at both ends, which equals replicating
    the edge bins."""
    taps, stride = _pool_taps(src_pool, pool)
    halo = _halo(taps, stride)
    n = z.shape[-1] // stride
    if halo:
        z = F.pad(z, (halo, halo), mode="replicate")
    return _strided_taps(z, taps, stride, 0, n)


@dataclass(frozen=True)
class BlockPlan:
    """Window geometry of one block, relative to the anchor ``a``."""

    pool: int         # pooling factor of the block
    radius: int       # token-mixer reach, in pooled positions
    changed: int      # block input may differ on [a - changed, a + align + changed)
    context: int      # block input is computed on [a - context, a + align + context)
    src_pool: int     # resolution the input window is gathered from (1: one-hot)
    src_start: int    # gather start, relative to a // src_pool
    src_width: int
    patch_start: int  # previous block's variant slice within the gathered window
    out_start: int    # variant output slice start, relative to a // pool
    out_crop: int     # its offset within the block's output window
    out_width: int


def plan_blocks(model):
    """Per-block windows for ``model`` and the half-width of the region where
    the last block's (upsampled) output can change."""
    align = model.max_pool
    plans, changed = [], 0
    for b in range(model.num_blocks):
        pool = _first(model._downsamplers[b].kernel_size)
        up = model._upsamplers[b]
        if float(_first(up.scale_factor)) != pool or up.mode != "linear":
            raise ValueError(f"block {b}: upsampling must undo the pooling")
        mixer = model._blocks[b]._token_mixer
        kernel = _first(mixer.kernel_size)
        if kernel % 2 == 0:
            raise ValueError("token mixer kernel must be odd")
        radius = (kernel - 1) // 2 * _first(mixer.dilation)

        # the variant output covers the changed input ± radius; computing it
        # needs another radius of input on each side
        context = _ceil_to(changed + 2 * radius * pool, align)
        out_start = -(changed // pool) - radius
        if b == 0:
            src_pool, src_start = 1, -context
            src_width, patch_start = 2 * context + align, 0
        else:
            prev = plans[-1]
            # one extra source position per side for the interpolation
            src_pool = prev.pool
            src_start = -(context // src_pool) - 1
            src_width = (2 * context + align) // src_pool + 2
            patch_start = prev.out_start - src_start
        plans.append(BlockPlan(
            pool, radius, changed, context, src_pool, src_start, src_width,
            patch_start, out_start,
            out_crop=(context - changed) // pool - radius,
            out_width=(2 * changed + align) // pool + 2 * radius,
        ))
        # linear upsampling spreads each pooled change by one bin per side
        changed = _ceil_to(changed + (radius + 1) * pool, align)
    return plans, changed


class IncrementalEnsemble(nn.Module):
    """Two-model ensemble that scores substitutions against a cached reference.

    Stages:
      * ``reference_1d``: full pass of the reference, returns the cache;
      * ``variants_1d``: a batch of substitutions; splice-site logits inside
        each one's cone (``patch_track`` turns them into full tracks);
      * ``head_2d``: top-k candidate sites + junction heads, for the
        reference (``pos=None``) or for variants.
    Splice-site logits are averaged over models as in ``Ensemble``.
    """

    def __init__(self, models, out_2d_len, opts=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.out_2d_len = out_2d_len
        self.opts = Optimizations() if opts is None else opts
        m = models[0]
        self.num_blocks = m.num_blocks
        self.crop_len = m.crop_len
        self.align = m.max_pool
        self.plans, self.reach = plan_blocks(m)
        if any(plan_blocks(o) != (self.plans, self.reach) for o in models[1:]):
            raise ValueError("models must share one architecture")
        half = self.crop_len // 2
        if half % self.align:
            raise ValueError(f"crop_len / 2 must be divisible by {self.align}")
        if self.reach > half:
            raise ValueError(
                f"a substitution reaches ±{self.reach} nt, beyond the "
                f"{half} nt flank; outputs near the edges would not be exact"
            )
        # Windows of substitutions near the ends of the output region extend
        # past the input; the cache is zero-padded by this many positions.
        # Values computed from the padding only reach positions outside the
        # output region, which the flank above guarantees.
        low = max(-p.src_start * p.src_pool for p in self.plans)
        high = max((p.src_start + p.src_width) * p.src_pool - self.align
                   for p in self.plans)
        stitch = self.reach + self.align  # 1D window plus one bin per side
        self.margin = _ceil_to(
            max(0, low - half, high - half, stitch - half), self.align
        )
        if self.opts.stencil_resample:  # fail early on unsupported pooling
            for prev, plan in zip(self.plans, self.plans[1:]):
                _pool_taps(prev.pool, plan.pool)
        # blocks per pooling factor, in order (grouped_head_upsample)
        self.pool_groups = {}
        for b, plan in enumerate(self.plans):
            self.pool_groups.setdefault(plan.pool, []).append(b)

    @property
    def _pad_to(self):
        return 8 if self.opts.padded_ffn else 1

    @property
    def window_width(self):
        """Width of each variant's 1D window: the cone around its anchor."""
        return 2 * self.reach + self.align

    def anchors(self, pos):
        """Anchor ``a`` and the substitution's column in the first input
        window, for substitutions at output positions ``pos`` (B,)."""
        p = pos + self.crop_len // 2
        a = p // self.align * self.align
        return a, p - a - self.plans[0].src_start

    def window_offsets(self, pos):
        """Output position of the first column of each variant's 1D window."""
        return self.anchors(pos)[0] - self.crop_len // 2 - self.reach

    def _window(self, a, pool, start, width):
        return (a // pool + start).unsqueeze(1) + torch.arange(
            width, device=a.device
        )

    def reference_1d(self, x):
        """Full pass of the reference. x: (1, 14, N) one-hot.

        Returns the cache ``(onehot, z, h)`` and splice-site logits (1, 2, L).
        ``z[k][b]`` is block b's pooled output of model k and ``h[k][b]`` its
        splice-site head, both zero-padded by ``margin`` (full-res units).
        """
        onehot = F.pad(x[0], (self.margin, self.margin))
        z, h = [], []
        for model in self.models:
            zs, hs = [], []
            xb = model._downsamplers[0](
                model._embed(x.to(model._embed.weight.dtype))
            )
            for b, plan in enumerate(self.plans):
                if self.opts.reference_depthwise:
                    zb = _block_same(model._blocks[b], xb, self._pad_to)
                else:
                    zb = model._blocks[b](xb)
                pad = self.margin // plan.pool
                zs.append(F.pad(zb[0], (pad, pad)))
                hs.append(F.pad(model._ss_heads[b](zb)[0], (pad, pad)))
                if b + 1 < self.num_blocks:
                    nxt = self.plans[b + 1].pool
                    if self.opts.stencil_resample:
                        xb = _resample_full(zb, plan.pool, nxt)
                    else:
                        xb = model._downsamplers[b + 1](
                            model._upsamplers[b](zb)
                        )
            z.append(zs)
            h.append(hs)
        return (onehot, z, h), self._stitch_1d(h, x.shape[2])

    def variants_1d(self, cache, pos, alt):
        """B independent substitutions at output positions ``pos`` (B,),
        new bases ``alt`` (B,) (0-3 = A, C, G, T).

        Returns each block's recomputed pooled outputs (for ``head_2d``) and
        the splice-site logits (B, 2, window_width) over each variant's cone,
        starting at output position ``window_offsets(pos)``. Outside the cone
        the variant's logits equal the reference's.
        """
        onehot, z, h = cache
        a, col = self.anchors(pos)
        first = self.plans[0]
        # gathers come out (C, B, W); convolutions get contiguous (B, C, W)
        x = onehot[:, self._window(a, 1, first.src_start + self.margin,
                                   first.src_width)].permute(1, 0, 2)
        x = x.to(self.models[0]._embed.weight.dtype).contiguous()
        x[torch.arange(len(a), device=x.device), :4, col] = (
            F.one_hot(alt, 4).to(x.dtype)
        )

        z_var, h_var = [], []
        for k, model in enumerate(self.models):
            zv, hv = [], []
            xb = model._downsamplers[0](model._embed(x))
            for b, plan in enumerate(self.plans):
                if b > 0:
                    # previous output: reference with the variant slice in it
                    prev = self.plans[b - 1]
                    ctx = z[k][b - 1][:, self._window(
                        a, plan.src_pool,
                        plan.src_start + self.margin // plan.src_pool,
                        plan.src_width,
                    )].permute(1, 0, 2)
                    s = plan.patch_start
                    ctx = torch.cat(
                        [ctx[..., :s], zv[-1], ctx[..., s + prev.out_width:]],
                        dim=2,
                    ).contiguous()
                    n_in = 2 * plan.context + self.align  # full resolution
                    if self.opts.stencil_resample:
                        xb = _resample_window(ctx, plan.src_pool, plan.pool,
                                              n_in // plan.pool)
                    else:
                        u = model._upsamplers[b - 1](ctx)[
                            ..., plan.src_pool:plan.src_pool + n_in
                        ].contiguous()
                        xb = model._downsamplers[b](u)
                zb = _block_window(model._blocks[b], xb, plan.out_crop,
                                   plan.out_width, self._pad_to)
                zv.append(zb)
                hv.append(model._ss_heads[b](zb))
            z_var.append(zv)
            h_var.append(hv)
        return z_var, self._stitch_window(h, a, h_var)

    def variants_track(self, cache, ref_ss, pos, alt):
        """``variants_1d`` with full-length (B, 2, L) splice-site tracks."""
        z_var, win = self.variants_1d(cache, pos, alt)
        return z_var, self.patch_track(ref_ss, pos, win)

    def patch_track(self, track, pos, win):
        """``track`` (1 or B, C, L) over the output region, with each
        variant's window ``win`` (B, C, window_width) written in."""
        a = self.anchors(pos)[0]
        n_out, width = track.shape[2], win.shape[2]
        pad = self.reach + self.align  # windows reach at most this far out
        full = F.pad(track, (pad, pad)).expand(len(a), -1, -1)
        idx = self._window(a, 1, pad - self.crop_len // 2 - self.reach, width)
        full = full.scatter(2, idx.unsqueeze(1).expand(-1, win.shape[1], -1),
                            win.to(full.dtype))
        return full[..., pad:pad + n_out]

    def _stitch_1d(self, h, n_in):
        """Upsamples and sums the per-block head outputs over the output
        region."""
        lo = self.margin + self.crop_len // 2
        hi = self.margin + n_in - self.crop_len // 2
        if self.opts.grouped_head_upsample:
            return self._upsample_grouped(
                lambda k, b: h[k][b].unsqueeze(0), lambda p: slice(lo, hi)
            )
        outs = []
        for model, hk in zip(self.models, h):
            out = 0.0
            for b in range(self.num_blocks):
                out = out + (1 / self.num_blocks) * (
                    model._upsamplers[b](hk[b].unsqueeze(0))[..., lo:hi]
                )
            outs.append(out)
        return torch.stack(outs).mean(dim=0)

    def _stitch_window(self, h, a, h_var):
        """Like ``_stitch_1d``, on each variant's cone only: the reference
        heads with the variant's slices in them, gathered with one extra bin
        per side for the interpolation, upsampled and cropped."""
        width = self.window_width

        def head(k, b):
            plan = self.plans[b]
            pool = plan.pool
            start = -(self.reach // pool) - 1
            ctx = h[k][b][:, self._window(
                a, pool, start + self.margin // pool, width // pool + 2,
            )].permute(1, 0, 2)
            s = plan.out_start - start
            return torch.cat(
                [ctx[..., :s], h_var[k][b], ctx[..., s + plan.out_width:]],
                dim=2,
            )

        if self.opts.grouped_head_upsample:
            return self._upsample_grouped(
                head, lambda pool: slice(pool, pool + width)
            )
        outs = []
        for k, model in enumerate(self.models):
            out = 0.0
            for b, plan in enumerate(self.plans):
                pool = plan.pool
                out = out + (1 / self.num_blocks) * (
                    model._upsamplers[b](head(k, b))[..., pool:pool + width]
                )
            outs.append(out)
        return torch.stack(outs).mean(dim=0)

    def _upsample_grouped(self, head, crop):
        """Mean over models of the per-block head outputs ``head(k, b)``
        after upsampling, cropped to ``crop(pool)``: the outputs of all
        blocks with the same pooling factor, in both models, are summed (in
        fp32) and upsampled once, which is exact because linear upsampling
        is linear and their windows coincide."""
        scale = 1 / (self.num_blocks * len(self.models))
        out, dtype = None, None
        for pool, blocks in self.pool_groups.items():
            acc = 0.0
            for k in range(len(self.models)):
                for b in blocks:
                    hb = head(k, b)
                    dtype = hb.dtype
                    acc = acc + _acc(hb)
            up = self.models[0]._upsamplers[blocks[0]](acc)[..., crop(pool)]
            out = up if out is None else out + up
        return (scale * out).to(dtype)

    def head_2d(self, z, out_ss, pos=None, z_var=None, valid_len=None):
        """Top-k candidate sites and junction logits, as ``Ensemble.head_2d``.

        Reference: ``head_2d(z, out_ss)``; variants: also pass the ``pos`` and
        ``z_var`` used for (and returned by) ``variants_1d``, and their full
        tracks (``patch_track``; only the max over channels is used, so a
        (B, 1, L) track of maxima works too). Output positions at or past
        ``valid_len`` (tensor, 1 or B values) are never candidate sites.
        """
        a = None if pos is None else self.anchors(pos)[0]
        scores = out_ss.amax(dim=1)
        if valid_len is not None:
            p = torch.arange(scores.shape[1], device=scores.device)
            scores = scores.masked_fill(p >= valid_len.reshape(-1, 1),
                                        float("-inf"))
        idxs = scores.topk(self.out_2d_len, dim=1)[1].sort(dim=1).values
        sites = idxs + self.crop_len // 2  # input coordinates
        outs = []
        for k, model in enumerate(self.models):
            out = 0.0
            for b, plan in enumerate(self.plans):
                feats = self._features_at(
                    z[k][b], None if z_var is None else z_var[k][b],
                    plan, a, sites,
                )
                out = out + (1 / self.num_blocks) * model._jxn_heads[b](
                    feats, idxs
                )
            outs.append(out)
        return idxs, torch.stack(outs).mean(dim=0)

    def _features_at(self, zb, zv, plan, a, sites):
        """Block output after linear upsampling, at input positions ``sites``
        (B, K); computed like nn.Upsample(mode="linear")."""
        pad = self.margin // plan.pool
        n = zb.shape[1] - 2 * pad
        src = ((sites.float() + 0.5) * (1.0 / plan.pool) - 0.5).clamp(min=0)
        i0 = src.long()
        w = (src - i0).unsqueeze(1)
        i1 = (i0 + 1).clamp(max=n - 1)

        def at(i):
            ref = zb[:, i + pad].permute(1, 0, 2)
            if zv is None:
                return ref
            rel = i - (a // plan.pool + plan.out_start).unsqueeze(1)
            inside = (rel >= 0) & (rel < plan.out_width)
            var = zv.gather(2, rel.clamp(0, plan.out_width - 1)
                            .unsqueeze(1).expand(-1, zv.shape[1], -1))
            return torch.where(inside.unsqueeze(1), var, ref)

        feats = (1 - w) * _acc(at(i0)) + w * _acc(at(i1))
        return feats.to(zb.dtype).contiguous()

    def recompute_fraction(self, n_in):
        """Block positions whose output is recomputed per variant, relative
        to a full pass."""
        full = sum(n_in // p.pool for p in self.plans)
        return sum(p.out_width for p in self.plans) / full
