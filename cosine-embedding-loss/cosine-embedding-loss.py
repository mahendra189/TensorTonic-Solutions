import torch

def cosine_embedding_loss(
    x1: list,
    x2: list,
    label: int,
    margin: float
) -> float:

    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)

    dot_product = torch.sum(x1 * x2)

    norm_x1 = torch.sqrt(torch.sum(x1 ** 2))
    norm_x2 = torch.sqrt(torch.sum(x2 ** 2))

    cos_sim = dot_product / (norm_x1 * norm_x2)

    if label == 1:
        loss = 1 - cos_sim

    elif label == -1:
        loss = torch.clamp(cos_sim - margin, min=0)

    else:
        raise ValueError("label must be 1 or -1")

    return float(loss)