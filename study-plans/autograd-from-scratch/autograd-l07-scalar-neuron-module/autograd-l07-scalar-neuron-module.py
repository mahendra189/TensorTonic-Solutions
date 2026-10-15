import torch

def scalar_neuron_module(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns a scalar tensor preserving the input dtype and device.
    """
    if bias.ndim == 1 and weights.ndim == 1:
        a = bias + weights @ inputs
        a = torch.tanh(a) if nonlinear else a
        return a
    else:
        a = bias + torch.sum(weights * inputs)
        a = (torch.tanh(a) if nonlinear else a)
        return a
