import torch

from spliceai2_bench.encode import one_hot_encode, random_sequences
from spliceai2_bench.model import SpliceAI2
from spliceai2_bench.predict import Ensemble
from spliceai2_bench.weights import load_model

NUM_FEATURES, NUM_BLOCKS, CROP_LEN, OUT_LEN = 8, 8, 64, 96


def small_model(seed=0):
    torch.manual_seed(seed)
    return SpliceAI2(NUM_FEATURES, NUM_BLOCKS, crop_len=CROP_LEN).eval()


def batch(n=2):
    seqs = random_sequences(n, OUT_LEN + CROP_LEN)
    return torch.stack([torch.from_numpy(one_hot_encode(s)) for s in seqs])


def test_state_dict_keys_match_reference_names():
    keys = set(small_model().state_dict())
    for k in [
        "_embed.weight",
        "_blocks.0._token_norm.running_mean",
        "_blocks.0._token_mixer.weight",
        "_blocks.0._ffn_proj_up.weight",
        "_blocks.7._ffn_proj_down.bias",
        "_ss_heads.0.weight",
        "_jxn_heads.0._token_norm.weight",
        "_jxn_heads.0._query.weight",
        "_jxn_heads.7._key.bias",
    ]:
        assert k in keys


def test_output_shapes():
    model = small_model()
    x = batch().float()
    with torch.inference_mode():
        ftrs = model(x)
        assert len(ftrs) == NUM_BLOCKS
        assert all(f.shape == (2, NUM_FEATURES, OUT_LEN) for f in ftrs)
        assert model.forward_1d(ftrs).shape == (2, 2, OUT_LEN)
        idxs = torch.arange(0, OUT_LEN, 4).expand(2, -1)
        out_jxn = model.forward_2d(ftrs, idxs)
    k = idxs.shape[1]
    assert out_jxn.shape == (2, k, k)
    # donor must precede acceptor: lower triangle incl. diagonal is masked
    assert (out_jxn.tril() < -1e3).sum() == 2 * k * (k + 1) // 2


def test_ensemble_averages_models():
    models = [small_model(0), small_model(1)]
    ens = Ensemble(models, out_2d_len=16).eval()
    x = batch()
    with torch.inference_mode():
        ftrs, out_ss = ens.trunk_1d(x)
        expected = torch.stack(
            [m.forward_1d(m(x.float())) for m in models]
        ).mean(dim=0)
        torch.testing.assert_close(out_ss, expected)
        torch.testing.assert_close(ens.predict_1d(x), out_ss)
        idxs, out_jxn = ens.head_2d(ftrs, out_ss)
    assert idxs.shape == (2, 16)
    assert (idxs.diff(dim=1) > 0).all()
    assert out_jxn.shape == (2, 16, 16)


def test_head_2d_ignores_padding():
    ens = Ensemble([small_model(0), small_model(1)], out_2d_len=16).eval()
    valid_len = torch.tensor([40, 96])
    with torch.inference_mode():
        ftrs, out_ss = ens.trunk_1d(batch())
        idxs, _ = ens.head_2d(ftrs, out_ss, valid_len)
    assert (idxs < valid_len.unsqueeze(1)).all()
    assert idxs[0].max() < 40


def test_right_padding_leaves_real_outputs_unchanged():
    # receptive field of 8 blocks, kernel 5: ±108 nt, within crop_len / 2
    torch.manual_seed(0)
    model = SpliceAI2(NUM_FEATURES, NUM_BLOCKS, crop_len=256).eval()
    real_len, padded_len = 96, 128
    seq = random_sequences(1, real_len + 256)[0]
    x = torch.from_numpy(one_hot_encode(seq))[None].float()
    x_pad = torch.from_numpy(
        one_hot_encode(seq, length=padded_len + 256)
    )[None].float()
    with torch.inference_mode():
        out = model.forward_1d(model(x))
        out_pad = model.forward_1d(model(x_pad))
    assert out_pad.shape[2] == padded_len
    torch.testing.assert_close(out_pad[..., :real_len], out)


def test_load_lightning_style_checkpoint(tmp_path):
    ref = small_model(3)
    ckpt = {
        "state_dict": ref.state_dict(),
        "hyper_parameters": {
            "num_features": NUM_FEATURES,
            "num_blocks": NUM_BLOCKS,
            "kernel_size": 5,
            "dilation": tuple(2 ** (b // 2) for b in range(32)),
            "pool": tuple(2 ** (b // 4) for b in range(32)),
            "crop_len": CROP_LEN,
        },
    }
    path = tmp_path / "model.ckpt"
    torch.save(ckpt, path)
    loaded = load_model(path)
    x = batch().float()
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded.forward_1d(loaded(x)), ref.forward_1d(ref(x))
        )
