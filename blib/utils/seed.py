import torch

def set_seed(seed):
    """
    Sets the random seed for CPU, GPU and multi-GPU.
    Returns nothing
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    assert torch.initial_seed() == seed, 'Seeding failed, somehow'