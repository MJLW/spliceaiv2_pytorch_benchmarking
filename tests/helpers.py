"""Small random SpliceAI2 models with the released architecture's shape."""
import torch

from spliceai2_ism.model import SpliceAI2

# 8 blocks reach ±124 nt, so a 256 nt crop (128 per side) keeps outputs exact
NUM_FEATURES, NUM_BLOCKS, CROP_LEN = 8, 8, 256
# 12 blocks (pools 1, 2 and 4, so pooling doubles twice) reach ±920 nt
DEEP_BLOCKS, DEEP_CROP_LEN = 12, 2048


def small_models(seeds=(0, 1), num_blocks=NUM_BLOCKS, crop_len=CROP_LEN):
    models = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = SpliceAI2(NUM_FEATURES, num_blocks, crop_len=crop_len).eval()
        # non-trivial BatchNorm statistics, as in a trained model
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.uniform_(-1, 1)
                mod.running_var.uniform_(0.5, 2)
        models.append(model)
    return models
