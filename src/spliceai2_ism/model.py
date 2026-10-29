"""Inference-only PyTorch implementation of the SpliceAI2 network.

Follows the architecture of Illumina's reference (spliceai2/model.py) and keeps
its module names, so the released Lightning checkpoints load with strict=True.
Lightning, activation checkpointing and the training code are left out.
"""
import torch
from torch import nn


class MetaFormer(nn.Module):
    """Pre-norm block: dilated depthwise conv token mixer + pointwise FFN."""

    def __init__(self, num_features, kernel_size, dilation):
        super().__init__()
        self._token_norm = nn.BatchNorm1d(num_features)
        self._token_mixer = nn.Conv1d(
            num_features,
            num_features,
            kernel_size,
            padding="same",
            dilation=dilation,
            groups=num_features,
        )
        self._ffn_norm = nn.BatchNorm1d(num_features)
        self._ffn_proj_up = nn.Conv1d(num_features, 4 * num_features, 1)
        self._ffn_act = nn.ReLU()
        self._ffn_proj_down = nn.Conv1d(4 * num_features, num_features, 1)

    def forward(self, x):
        x = x + self._token_mixer(self._token_norm(x))
        x = x + self._ffn_proj_down(
            self._ffn_act(self._ffn_proj_up(self._ffn_norm(x)))
        )
        return x


class Attention(nn.Module):
    """Junction head: RoPE query/key dot products over candidate sites.

    Returns (B, K, K) logits for donor i -> acceptor j; entries with j <= i
    are pushed down by 1e4.
    """

    def __init__(self, num_features):
        super().__init__()
        self._token_norm = nn.LayerNorm(num_features)
        self._query = nn.Conv1d(num_features, num_features, 1)
        self._key = nn.Conv1d(num_features, num_features, 1)
        self._num_features = num_features

    def forward(self, x, token_idxs):
        x = self._token_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        out = (self._num_features ** -0.5) * torch.einsum(
            "bcl,bcm->blm",
            self._rope(self._query(x), token_idxs),
            self._rope(self._key(x), token_idxs),
        )
        out -= out.clone().fill_(1e4).tril()
        return out

    def _rope(self, x, token_idxs):
        freqs = 1e4 ** -(
            torch.arange(0, self._num_features, 2, device=x.device)
            / self._num_features
        )
        angle_mtrx = token_idxs.unsqueeze(1) * freqs.unsqueeze(1).unsqueeze(0)
        sin, cos = angle_mtrx.sin(), angle_mtrx.cos()
        x1, x2 = x[:, 0::2], x[:, 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=1)


class SpliceAI2(nn.Module):
    """Multi-resolution conv trunk with per-block 1D and 2D heads.

    Input is (B, 14, L + crop_len) one-hot (4 nucleotides + 10 assemblies);
    outputs cover the central L positions. Every block's input length must be
    divisible by its pooling factor, i.e. (L + crop_len) % max_pool == 0.
    """

    def __init__(
        self,
        num_features,
        num_blocks,
        kernel_size=5,
        dilation=tuple(2 ** (b // 2) for b in range(32)),
        pool=tuple(2 ** (b // 4) for b in range(32)),
        crop_len=0,
    ):
        super().__init__()
        self._embed = nn.Conv1d(14, num_features, 1)
        self._downsamplers = nn.ModuleList(
            [nn.AvgPool1d(pool[b]) for b in range(num_blocks)]
        )
        self._blocks = nn.ModuleList(
            [MetaFormer(num_features, kernel_size, dilation[b])
             for b in range(num_blocks)]
        )
        self._upsamplers = nn.ModuleList(
            [nn.Upsample(scale_factor=pool[b], mode="linear")
             for b in range(num_blocks)]
        )
        self._ss_heads = nn.ModuleList(
            [nn.Conv1d(num_features, 2, 1) for _ in range(num_blocks)]
        )
        self._jxn_heads = nn.ModuleList(
            [Attention(num_features) for _ in range(num_blocks)]
        )

        self.num_features = num_features
        self.num_blocks = num_blocks
        self.crop_len = crop_len
        self.max_pool = max(pool[:num_blocks])

    def forward(self, x):
        """Returns the per-block features, cropped to the output window."""
        ftrs = []
        x = self._embed(x)
        for b in range(self.num_blocks):
            x = self._downsamplers[b](x)
            x = self._blocks[b](x)
            x = self._upsamplers[b](x)
            ftrs.append(x[..., self.crop_len // 2:-self.crop_len // 2 or None])
        return ftrs

    def forward_1d(self, ftrs):
        """Splice-site logits, (B, 2, L): donor and acceptor."""
        out_ss = 0.0
        for b in range(self.num_blocks):
            out_ss += (1 / self.num_blocks) * self._ss_heads[b](ftrs[b])
        return out_ss

    def forward_2d(self, ftrs, y_2d_idxs):
        """Junction logits, (B, K, K), between the K sites in y_2d_idxs."""
        out_jxn = 0.0
        for b in range(self.num_blocks):
            out_jxn += (1 / self.num_blocks) * self._jxn_heads[b](
                ftrs[b].gather(
                    2, y_2d_idxs.unsqueeze(1).repeat(1, ftrs[b].shape[1], 1)
                ),
                y_2d_idxs,
            )
        return out_jxn
