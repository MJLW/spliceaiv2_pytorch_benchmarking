"""Two-model ensemble prediction, split into 1D and 2D stages for timing."""
import torch
from torch import nn


class Ensemble(nn.Module):
    """Averages logits over models, as in the reference transcript example.

    Stages:
      * ``predict_1d`` / ``trunk_1d``: trunk + splice-site heads.
      * ``head_2d``: top-k candidate sites + junction heads.
    """

    def __init__(self, models, out_2d_len):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.out_2d_len = out_2d_len

    def predict_1d(self, x):
        """Splice-site logits only; per-block features are not kept."""
        return self.trunk_1d(x)[1]

    def trunk_1d(self, x):
        x = x.float()
        ftrs = [model(x) for model in self.models]
        out_ss = torch.stack(
            [model.forward_1d(f) for model, f in zip(self.models, ftrs)]
        ).mean(dim=0)
        return ftrs, out_ss

    def head_2d(self, ftrs, out_ss, valid_len=None):
        """``valid_len`` (B,) excludes padded positions past each sequence's
        real length from the candidate sites."""
        scores = out_ss.amax(dim=1)
        if valid_len is not None:
            pos = torch.arange(scores.shape[1], device=scores.device)
            scores = scores.masked_fill(
                pos >= valid_len.unsqueeze(1), float("-inf")
            )
        idxs = scores.topk(self.out_2d_len, dim=1)[1].sort(dim=1).values
        out_jxn = torch.stack(
            [model.forward_2d(f, idxs) for model, f in zip(self.models, ftrs)]
        ).mean(dim=0)
        return idxs, out_jxn


def predict_full(models, x, out_2d_len, valid_len=None):
    """``Ensemble.trunk_1d`` + ``head_2d`` with the original model code, one
    model's full-length features at a time: the 1D pass picks the top-k
    sites, a second pass runs the junction heads on them. Used to check the
    incremental path against plain full forward passes."""
    x = x.float()
    out_ss = torch.stack([m.forward_1d(m(x)) for m in models]).mean(dim=0)
    scores = out_ss.amax(dim=1)
    if valid_len is not None:
        pos = torch.arange(scores.shape[1], device=scores.device)
        scores = scores.masked_fill(pos >= valid_len.unsqueeze(1), float("-inf"))
    idxs = scores.topk(out_2d_len, dim=1)[1].sort(dim=1).values
    out_jxn = torch.stack([m.forward_2d(m(x), idxs) for m in models]).mean(dim=0)
    return out_ss, idxs, out_jxn
