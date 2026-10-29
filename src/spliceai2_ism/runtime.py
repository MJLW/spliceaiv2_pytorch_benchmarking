"""Inference context and torch.compile settings shared by the ISM run and
the benchmarks."""
import contextlib

import torch
import torch._dynamo


@contextlib.contextmanager
def inference(device):
    """inference_mode, plus FP16 autocast on CUDA (as the reference scorer)."""
    with torch.inference_mode(), torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        yield


def environment_info():
    cuda = torch.cuda.is_available()
    return {
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
    }


def raise_recompile_limit(n_graphs):
    cfg = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, name):
            setattr(cfg, name, max(getattr(cfg, name), n_graphs + 2))
            return


def forbid_eager_fallback():
    """Never fall back to eager silently if a stage keeps recompiling."""
    for flag in ("fail_on_recompile_limit_hit", "fail_on_cache_limit_hit"):
        if hasattr(torch._dynamo.config, flag):
            setattr(torch._dynamo.config, flag, True)
