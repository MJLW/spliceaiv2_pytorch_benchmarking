import pytest
import torch

from spliceai2_bench.encode import one_hot_encode, random_sequences
from spliceai2_bench.incremental import (
    IncrementalEnsemble,
    _depthwise_same,
    plan_blocks,
)
from spliceai2_bench.model import SpliceAI2
from spliceai2_bench.predict import Ensemble

# 8 blocks reach ±124 nt, so a 256 nt crop (128 per side) keeps outputs exact
NUM_FEATURES, NUM_BLOCKS, CROP_LEN, OUT_LEN, OUT_2D_LEN = 8, 8, 256, 96, 16


def small_models():
    models = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = SpliceAI2(NUM_FEATURES, NUM_BLOCKS, crop_len=CROP_LEN).eval()
        # non-trivial BatchNorm statistics, as in a trained model
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-1, 1)
                mod.running_var.uniform_(0.5, 2)
        models.append(model)
    return models


def reference_input():
    seq = random_sequences(1, OUT_LEN + CROP_LEN, seed=5)[0]
    return torch.from_numpy(one_hot_encode(seq))[None]


def substitute(x, pos, alt):
    x = x.clone()
    x[0, :4, CROP_LEN // 2 + pos] = 0
    x[0, alt, CROP_LEN // 2 + pos] = 1
    return x


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


def test_reference_matches_ensemble():
    models = small_models()
    ens = Ensemble(models, OUT_2D_LEN).eval()
    inc = IncrementalEnsemble(models, OUT_2D_LEN).eval()
    x = reference_input()
    with torch.inference_mode():
        cache, out_ss = inc.reference_1d(x)
        idxs, out_jxn = inc.head_2d(cache[1], out_ss)
        ftrs, ens_ss = ens.trunk_1d(x)
        ens_idxs, ens_jxn = ens.head_2d(ftrs, ens_ss)
    torch.testing.assert_close(out_ss, ens_ss)
    torch.testing.assert_close(idxs, ens_idxs)
    torch.testing.assert_close(out_jxn, ens_jxn)


@pytest.mark.parametrize("positions", [
    [0, 1, 37, 50, OUT_LEN - 1],  # includes both ends of the output region
    [40, 41, 42],                 # neighbouring positions, one batch
])
def test_variants_match_full_forward(positions):
    models = small_models()
    ens = Ensemble(models, OUT_2D_LEN).eval()
    inc = IncrementalEnsemble(models, OUT_2D_LEN).eval()
    x = reference_input()
    pos = torch.tensor(positions)
    ref_base = x[0, :4, CROP_LEN // 2 + pos].argmax(dim=0)
    alt = (ref_base + 1 + torch.arange(len(pos)) % 3) % 4
    with torch.inference_mode():
        cache, _ = inc.reference_1d(x)
        z_var, out_ss = inc.variants_1d(cache, pos, alt)
        idxs, out_jxn = inc.head_2d(cache[1], out_ss, pos, z_var)
        for j in range(len(pos)):
            ftrs, ens_ss = ens.trunk_1d(substitute(x, pos[j], alt[j]))
            ens_idxs, ens_jxn = ens.head_2d(ftrs, ens_ss)
            torch.testing.assert_close(out_ss[j], ens_ss[0])
            torch.testing.assert_close(idxs[j], ens_idxs[0])
            torch.testing.assert_close(out_jxn[j], ens_jxn[0])


@pytest.mark.parametrize("dilation", [1, 4, 16])
def test_depthwise_same_matches_conv(dilation):
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(8, 8, 5, padding="same", dilation=dilation,
                           groups=8)
    x = torch.randn(3, 8, 70)
    torch.testing.assert_close(_depthwise_same(conv, x), conv(x))


def test_released_geometry_fits_flank():
    model = SpliceAI2(256, 20, crop_len=131_072)
    _, reach = plan_blocks(model)
    assert 50_000 < reach <= 131_072 // 2
    inc = IncrementalEnsemble([model], out_2d_len=1024)
    assert inc.recompute_fraction(100_000 + 131_072) < 0.05


def test_rejects_crop_shorter_than_reach():
    model = SpliceAI2(NUM_FEATURES, NUM_BLOCKS, crop_len=128)
    with pytest.raises(ValueError, match="reaches"):
        IncrementalEnsemble([model], OUT_2D_LEN)
