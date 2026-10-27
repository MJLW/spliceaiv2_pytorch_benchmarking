"""Incremental SpliceAI2 inference for single-base substitutions.

A substitution changes the network's activations only inside a cone that
widens block by block, to about ±56.5k nt after the last block of the
released models. ``IncrementalEnsemble`` runs the reference sequence once,
keeps every block's pooled output, and computes a variant by recomputing only
that cone; everything outside it is taken from the reference. Every output
position and every candidate site of the junction head is covered, so the
result equals a full forward pass on the variant sequence up to
floating-point rounding.

Two rewrites keep the per-variant work small; both are exact in exact
arithmetic:
  * the splice-site heads are applied before the linear upsampling (both are
    linear and the interpolation weights sum to one), so per-block 1D
    contributions are kept at pooled resolution;
  * junction-head features at the top-k sites are interpolated directly from
    the pooled block outputs instead of from full-length upsampled features.

Coordinates: full-resolution positions are model-input positions (0..N-1,
N = L + crop_len). A substitution at input position p is handled relative to
its anchor ``a`` = p rounded down to a multiple of ``align``, the deepest
pooling factor, so every window below covers whole pooling bins at every
resolution and its shape does not depend on p.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _first(v):
    return v[0] if isinstance(v, (tuple, list)) else v


def _ceil_to(x, m):
    return -(-x // m) * m


def _depthwise_same(conv, x):
    """``conv(x)`` for a depthwise 'same' Conv1d, as a sum of shifted
    products accumulated in fp32; under autocast the result is rounded to
    the autocast dtype, like the conv it replaces.

    Used on the short variant windows: cuDNN's depthwise kernel segfaulted
    there once the same layer had run on the full-length reference (torch
    2.14.1, cuDNN 9.24). Under torch.compile this fuses into one kernel.
    """
    k, d = _first(conv.kernel_size), _first(conv.dilation)
    r = (k - 1) // 2 * d
    n = x.shape[2]
    xp = F.pad(x, (r, r)).float()
    w = conv.weight.float()
    out = sum(w[:, 0, j, None] * xp[..., j * d:j * d + n] for j in range(k))
    if conv.bias is not None:
        out = out + conv.bias.float()[:, None]
    device = x.device.type
    if torch.is_autocast_enabled(device):
        return out.to(torch.get_autocast_dtype(device))
    return out.to(x.dtype)


def _block_window(block, x):
    """MetaFormer.forward with the token mixer replaced by _depthwise_same."""
    x = x + _depthwise_same(block._token_mixer, block._token_norm(x))
    return x + block._ffn_proj_down(
        block._ffn_act(block._ffn_proj_up(block._ffn_norm(x)))
    )


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
      * ``variants_1d``: splice-site logits of a batch of substitutions;
      * ``head_2d``: top-k candidate sites + junction heads, for the
        reference (``pos=None``) or for variants.
    Splice-site logits are averaged over models as in ``Ensemble``.
    """

    def __init__(self, models, out_2d_len):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.out_2d_len = out_2d_len
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
        self.margin = _ceil_to(max(0, low - half, high - half), self.align)

    def anchors(self, pos):
        """Anchor ``a`` and the substitution's column in the first input
        window, for substitutions at output positions ``pos`` (B,)."""
        p = pos + self.crop_len // 2
        a = p // self.align * self.align
        return a, p - a - self.plans[0].src_start

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
            u = model._embed(x.float())
            for b, plan in enumerate(self.plans):
                zb = model._blocks[b](model._downsamplers[b](u))
                pad = self.margin // plan.pool
                zs.append(F.pad(zb[0], (pad, pad)))
                hs.append(F.pad(model._ss_heads[b](zb)[0], (pad, pad)))
                if b + 1 < self.num_blocks:
                    u = model._upsamplers[b](zb)
            z.append(zs)
            h.append(hs)
        return (onehot, z, h), self._stitch_1d(h, x.shape[2])

    def variants_1d(self, cache, pos, alt):
        """Splice-site logits (B, 2, L) of B independent substitutions.

        pos: (B,) output positions, alt: (B,) new base (0-3 = A, C, G, T).
        Also returns each block's recomputed pooled outputs for ``head_2d``.
        """
        onehot, z, h = cache
        a, col = self.anchors(pos)
        first = self.plans[0]
        # gathers come out (C, B, W); convolutions get contiguous (B, C, W)
        x = onehot[:, self._window(a, 1, first.src_start + self.margin,
                                   first.src_width)].permute(1, 0, 2)
        x = x.float().contiguous()
        x[torch.arange(len(a), device=x.device), :4, col] = (
            F.one_hot(alt, 4).to(x.dtype)
        )

        z_var, h_var = [], []
        for k, model in enumerate(self.models):
            zv, hv = [], []
            u = model._embed(x)
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
                    u = model._upsamplers[b - 1](ctx)[
                        ..., plan.src_pool:
                        plan.src_pool + 2 * plan.context + self.align
                    ].contiguous()
                zb = _block_window(model._blocks[b],
                                   model._downsamplers[b](u))
                zb = zb[..., plan.out_crop:plan.out_crop + plan.out_width]
                zv.append(zb)
                hv.append(model._ss_heads[b](zb))
            z_var.append(zv)
            h_var.append(hv)
        return z_var, self._stitch_1d(h, onehot.shape[1] - 2 * self.margin,
                                      a, h_var)

    def _stitch_1d(self, h, n_in, a=None, h_var=None):
        """Upsamples and sums the per-block head outputs over the output
        region; with ``h_var``, each variant's slices replace the reference."""
        lo = self.margin + self.crop_len // 2
        hi = self.margin + n_in - self.crop_len // 2
        outs = []
        for k, model in enumerate(self.models):
            out = 0.0
            for b, plan in enumerate(self.plans):
                hb = h[k][b].unsqueeze(0)
                if h_var is not None:
                    idx = self._window(
                        a, plan.pool, plan.out_start + self.margin // plan.pool,
                        plan.out_width,
                    )
                    hb = hb.expand(len(a), -1, -1).scatter(
                        2, idx.unsqueeze(1).expand(-1, hb.shape[1], -1),
                        h_var[k][b],
                    )
                out = out + (1 / self.num_blocks) * (
                    model._upsamplers[b](hb)[..., lo:hi]
                )
            outs.append(out)
        return torch.stack(outs).mean(dim=0)

    def head_2d(self, z, out_ss, pos=None, z_var=None):
        """Top-k candidate sites and junction logits, as ``Ensemble.head_2d``.

        Reference: ``head_2d(z, out_ss)``; variants: also pass the ``pos`` and
        ``z_var`` used for (and returned by) ``variants_1d``.
        """
        a = None if pos is None else self.anchors(pos)[0]
        scores = out_ss.amax(dim=1)
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

        feats = (1 - w) * at(i0).float() + w * at(i1).float()
        return feats.to(zb.dtype).contiguous()

    def recompute_fraction(self, n_in):
        """Block positions recomputed per variant, relative to a full pass."""
        full = sum(n_in // p.pool for p in self.plans)
        part = sum((2 * p.context + self.align) // p.pool for p in self.plans)
        return part / full
