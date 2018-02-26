import torch
import random

def set_seed(seed):
    """
    Sets seed for both Python and PyTorch RNGs.
    Seeds CPU, GPU and multi-GPU.
    Returns nothing
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) #apparently this is now not needed https://discuss.pytorch.org/t/random-seed-initialization/7854/4

    assert torch.initial_seed() == seed, 'Seeding failed, somehow'