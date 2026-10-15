import torch

def recursive_parameter_collection(weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns a list of scalar parameter views and its integer count.
    """
    pack = []
    for w,b in zip(weights,biases):
        for ws,bs in zip(w,b):
            pack.extend([*ws,bs])
    return pack,len(pack)
        
