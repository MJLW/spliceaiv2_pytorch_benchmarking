"""Download and load the released SpliceAI2 checkpoints from Hugging Face."""
import inspect
import pickle
import warnings
from pathlib import Path

import torch

from spliceai2_ism.model import SpliceAI2

REPO_ID = "illumina-ai/SpliceAI2"
MODEL_FOLDERS = (
    "spliceai2_multispecies_13M_01",
    "spliceai2_multispecies_13M_02",
)
CKPT_NAME = "model.ckpt"

_MODEL_ARGS = set(inspect.signature(SpliceAI2.__init__).parameters) - {"self"}


def download_models(cache_dir=None):
    """Fetches both model folders (gated repo: needs an accepted-terms login)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        REPO_ID,
        allow_patterns=[f"{folder}/*" for folder in MODEL_FOLDERS],
        cache_dir=cache_dir,
    )
    return Path(path)


def checkpoint_paths(model_dir):
    paths = [Path(model_dir) / folder / CKPT_NAME for folder in MODEL_FOLDERS]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint(s): {', '.join(missing)}")
    return paths


def load_model(ckpt_path):
    """Builds a SpliceAI2 from a Lightning checkpoint's hparams and weights."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Lightning checkpoints can carry non-tensor training state (e.g. LR
        # scheduler counters) that the safe unpickler rejects.
        warnings.warn(
            f"{ckpt_path}: falling back to torch.load(weights_only=False)"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hparams = {
        k: v for k, v in ckpt["hyper_parameters"].items() if k in _MODEL_ARGS
    }
    model = SpliceAI2(**hparams)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval()


def load_released_models():
    """Both released models, downloaded into (or read from) the HF cache."""
    return [load_model(p) for p in checkpoint_paths(download_models())]
