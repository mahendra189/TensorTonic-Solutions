import torch

def mlp_forward(inputs: torch.Tensor, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> tuple:
    """
    Returns the final output tensor and a list of layer-output tensors.
    """
    l = []
    a = inputs
    for w,b in zip(weights,biases):
        # print(w,b)
        a = torch.tanh(w @ a + b)
        l.append(a)
    return a,l
    