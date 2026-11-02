import pytest
import torch

from helpers import (
    CROP_LEN,
    DEEP_BLOCKS,
    DEEP_CROP_LEN,
    NUM_BLOCKS,
    NUM_FEATURES,
    small_models,
)
from spliceai2_ism.encode import one_hot_encode, random_sequences
from spliceai2_ism.incremental import (
    IncrementalEnsemble,
    Optimizations,
    _block_same,
    _block_window,
    _depthwise_same,
    _depthwise_window,
    _resample_full,
    _resample_window,
    plan_blocks,
)
from spliceai2_ism.model import SpliceAI2
from spliceai2_ism.predict import Ensemble

OUT_LEN, OUT_2D_LEN = 96, 16

# no rewrites, all of them, and each one on its own
OPTS = [Optimizations.none(), Optimizations()] + [
    Optimizations.parse(name) for name in Optimizations.names()
]
OPT_IDS = [o.spec() for o in OPTS]


def reference_input(out_len=OUT_LEN, crop_len=CROP_LEN):
    seq = random_sequences(1, out_len + crop_len, seed=5)[0]
    return torch.from_numpy(one_hot_encode(seq))[None]


def substitute(x, pos, alt, crop_len=CROP_LEN):
    x = x.clone()
    x[0, :4, crop_len // 2 + pos] = 0
    x[0, alt, crop_len // 2 + pos] = 1
    return x


def check_against_full_passes(models, opts, x, positions, crop_len=CROP_LEN):
    """Reference and variants of ``x`` through IncrementalEnsemble with
    ``opts`` vs full forward passes (Ensemble)."""
    ens = Ensemble(models, OUT_2D_LEN).eval()
    inc = IncrementalEnsemble(models, OUT_2D_LEN, opts).eval()
    pos = torch.tensor(positions)
    ref_base = x[0, :4, crop_len // 2 + pos].argmax(dim=0)
    alt = (ref_base + 1 + torch.arange(len(pos)) % 3) % 4
    with torch.inference_mode():
        cache, ref_ss = inc.reference_1d(x)
        ref_idxs, ref_jxn = inc.head_2d(cache[1], ref_ss)
        ftrs, ens_ss = ens.trunk_1d(x)
        ens_idxs, ens_jxn = ens.head_2d(ftrs, ens_ss)
        torch.testing.assert_close(ref_ss, ens_ss)
        torch.testing.assert_close(ref_idxs, ens_idxs)
        torch.testing.assert_close(ref_jxn, ens_jxn)

        z_var, out_ss = inc.variants_track(cache, ref_ss, pos, alt)
        idxs, out_jxn = inc.head_2d(cache[1], out_ss, pos, z_var)
        for j in range(len(pos)):
            ftrs, ens_ss = ens.trunk_1d(substitute(x, pos[j], alt[j],
                                                   crop_len))
            ens_idxs, ens_jxn = ens.head_2d(ftrs, ens_ss)
            torch.testing.assert_close(out_ss[j], ens_ss[0])
            torch.testing.assert_close(idxs[j], ens_idxs[0])
            torch.testing.assert_close(out_jxn[j], ens_jxn[0])


def test_changes_stay_inside_planned_windows():
    # float64: positions outside the true cone are bit-identical, so any
    # difference outside the plan would show up exactly
    model = small_models()[0].double()
    plans, reach = plan_blocks(model)
    x = reference_input().double()
    for pos in (0, 37, OUT_LEN - 1):
        p = CROP_LEN // 2 + pos
        a = p // model.max_pool * model.max_pool
        xv = x.clone()
        xv[0, :4, p] = xv[0, :4, p].roll(1)
        u, uv = model._embed(x), model._embed(xv)
        for b in range(NUM_BLOCKS):
            u, uv = (
                model._upsamplers[b](
                    model._blocks[b](model._downsamplers[b](t))
                )
                for t in (u, uv)
            )
            changed = plans[b + 1].changed if b + 1 < NUM_BLOCKS else reach
            diff = (u != uv).any(dim=1)[0].nonzero().flatten()
            assert diff.numel() > 0
            assert diff.min() >= a - changed
            assert diff.max() < a + model.max_pool + changed


@pytest.mark.parametrize("opts", OPTS, ids=OPT_IDS)
@pytest.mark.parametrize("positions", [
    [0, 1, 37, 50, OUT_LEN - 1],  # includes both ends of the output region
    [40, 41, 42],                 # neighbouring positions, one batch
])
def test_reference_and_variants_match_full_forward(positions, opts):
    check_against_full_passes(small_models(), opts, reference_input(),
                              positions)


@pytest.mark.parametrize("opts", [Optimizations.none(), Optimizations()],
                         ids=["none", "all"])
def test_deeper_model_matches_full_forward(opts):
    # 12 blocks: pooling goes 1 -> 2 -> 4, so the stride-2 stencil from an
    # even pooling factor (2 -> 4) is exercised too
    out_len = 64
    models = small_models(num_blocks=DEEP_BLOCKS, crop_len=DEEP_CROP_LEN)
    x = reference_input(out_len, DEEP_CROP_LEN)
    check_against_full_passes(models, opts, x, [0, 30, out_len - 1],
                              DEEP_CROP_LEN)


def test_optimized_matches_unoptimized_closely():
    # float64: every rewrite is exact in exact arithmetic, so the two paths
    # may only differ by rounding
    models = [m.double() for m in small_models()]
    x = reference_input().double()
    pos = torch.tensor([0, 17, 40, 41, OUT_LEN - 1])
    alt = (x[0, :4, CROP_LEN // 2 + pos].argmax(dim=0) + 1) % 4
    outs = []
    for opts in (Optimizations.none(), Optimizations()):
        inc = IncrementalEnsemble(models, OUT_2D_LEN, opts).eval()
        with torch.inference_mode():
            cache, ref_ss = inc.reference_1d(x)
            ref = (ref_ss, *inc.head_2d(cache[1], ref_ss))
            z_var, out_ss = inc.variants_track(cache, ref_ss, pos, alt)
            outs.append(ref + (out_ss, *inc.head_2d(cache[1], out_ss, pos,
                                                    z_var)))
    for a, b in zip(*outs):
        if a.is_floating_point():
            # junction logits on and below the diagonal sit near -1e4
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
        else:
            assert torch.equal(a, b)


@pytest.mark.parametrize("src_pool,pool", [
    (1, 1), (1, 2), (2, 2), (2, 4), (4, 4), (8, 16), (16, 16),
])
def test_resample_matches_upsample_then_pool(src_pool, pool):
    torch.manual_seed(0)
    n_out = 12
    up = torch.nn.Upsample(scale_factor=src_pool, mode="linear")
    down = torch.nn.AvgPool1d(pool)
    # whole sequence: the Upsample clamps at both ends
    z = torch.randn(2, 3, n_out * pool // src_pool, dtype=torch.float64)
    torch.testing.assert_close(_resample_full(z, src_pool, pool),
                               down(up(z)))
    # window with one extra source bin per side, cropped after upsampling
    ctx = torch.randn(2, 3, n_out * pool // src_pool + 2, dtype=torch.float64)
    full = up(ctx)[..., src_pool:src_pool + n_out * pool]
    torch.testing.assert_close(_resample_window(ctx, src_pool, pool, n_out),
                               down(full))


@pytest.mark.parametrize("src_pool,pool", [(2, 8), (3, 3), (4, 2)])
def test_resample_rejects_other_pooling(src_pool, pool):
    with pytest.raises(ValueError):
        _resample_full(torch.zeros(1, 1, 24), src_pool, pool)


@pytest.mark.parametrize("width", [29, 32, 33])
def test_padded_ffn_matches_unpadded(width):
    block = small_models()[0]._blocks[5]
    x = torch.randn(2, NUM_FEATURES, width + 20)
    with torch.inference_mode():
        torch.testing.assert_close(_block_window(block, x, 10, width, 8),
                                   _block_window(block, x, 10, width))
        torch.testing.assert_close(_block_same(block, x, 8), block(x))


def test_optimizations_parse():
    assert Optimizations.parse("all") == Optimizations()
    assert Optimizations.parse("none") == Optimizations.none()
    one = Optimizations.parse("fused_junction")
    assert one.fused_junction and not one.stencil_resample
    loo = Optimizations.parse("all,-padded_ffn")
    assert not loo.padded_ffn and loo.stencil_resample
    assert loo.spec() == ",".join(n for n in Optimizations.names()
                                  if n != "padded_ffn")
    assert Optimizations().spec() == "all"
    assert Optimizations.none().spec() == "none"
    with pytest.raises(ValueError, match="unknown"):
        Optimizations.parse("all,-turbo")


def test_variant_window_covers_every_change():
    # the 1D window from variants_1d starts at window_offsets and is the
    # only part of the track that may differ from the reference
    models = small_models()
    inc = IncrementalEnsemble(models, OUT_2D_LEN).eval()
    out_len = 512  # longer than the window (2 × 124 + 2)
    x = reference_input(out_len)
    pos = torch.tensor([0, 250, out_len - 1])
    alt = (x[0, :4, CROP_LEN // 2 + pos].argmax(dim=0) + 1) % 4
    with torch.inference_mode():
        cache, ref_ss = inc.reference_1d(x)
        _, win = inc.variants_1d(cache, pos, alt)
        _, track = inc.variants_track(cache, ref_ss, pos, alt)
    assert win.shape == (3, 2, inc.window_width)
    offsets = inc.window_offsets(pos)
    for j in range(len(pos)):
        o = torch.arange(out_len)
        outside = (o < offsets[j]) | (o >= offsets[j] + inc.window_width)
        assert outside.any()
        assert torch.equal(track[j][:, outside], ref_ss[0][:, outside])
        inside = ~outside
        torch.testing.assert_close(
            track[j][:, inside], win[j][:, o[inside] - offsets[j]]
        )


def test_head_2d_ignores_positions_past_valid_len():
    inc = IncrementalEnsemble(small_models(), OUT_2D_LEN).eval()
    with torch.inference_mode():
        cache, ref_ss = inc.reference_1d(reference_input())
        idxs, _ = inc.head_2d(cache[1], ref_ss, valid_len=torch.tensor([40]))
    assert idxs.max() < 40


@pytest.mark.parametrize("dilation", [1, 4, 16])
def test_depthwise_same_matches_conv(dilation):
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(8, 8, 5, padding="same", dilation=dilation,
                           groups=8)
    x = torch.randn(3, 8, 70)
    torch.testing.assert_close(_depthwise_same(conv, x), conv(x))


def test_depthwise_window_matches_conv_slice():
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(8, 8, 5, padding="same", dilation=4, groups=8)
    x = torch.randn(3, 8, 70)
    torch.testing.assert_close(_depthwise_window(conv, x, 8, 50),
                               conv(x)[..., 8:58])
    with pytest.raises(ValueError):
        _depthwise_window(conv, x, 7, 50)


def test_block_window_matches_block_slice():
    block = small_models()[0]._blocks[3]
    x = torch.randn(2, NUM_FEATURES, 60)
    with torch.inference_mode():
        torch.testing.assert_close(_block_window(block, x, 10, 30),
                                   block(x)[..., 10:40])


def test_released_geometry_fits_flank():
    model = SpliceAI2(256, 20, crop_len=131_072)
    _, reach = plan_blocks(model)
    assert reach == 56_480
    inc = IncrementalEnsemble([model], out_2d_len=1024)
    assert inc.recompute_fraction(100_000 + 131_072) < 0.05


def test_rejects_crop_shorter_than_reach():
    model = SpliceAI2(NUM_FEATURES, NUM_BLOCKS, crop_len=128)
    with pytest.raises(ValueError, match="reaches"):
        IncrementalEnsemble([model], OUT_2D_LEN)
