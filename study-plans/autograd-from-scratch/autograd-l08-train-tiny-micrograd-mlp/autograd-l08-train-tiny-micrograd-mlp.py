
import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int
) -> tuple:

    # Clone parameters so the supplied tensors remain unchanged.
    ws = [w.clone() for w in weights]
    bs = [b.clone() for b in biases]
    loss_history = []

    for _ in range(steps):
        # Forward pass: retain activations for backpropagation.
        activations = [inputs]

        for w, b in zip(ws, bs):
            h = torch.tanh(activations[-1] @ w.T + b)
            activations.append(h)

        # Scalar output for each example: shape (B,)
        predictions = activations[-1].squeeze(-1)

        # Summed squared error; remains a scalar for an empty batch.
        loss = ((predictions - targets) ** 2).sum()
        loss_history.append(loss.clone())

        # Empty batches have zero loss and no parameter updates.
        if inputs.shape[0] == 0:
            continue

        # dL/d(predictions), shape (B, 1)
        delta = 2 * (predictions - targets).unsqueeze(-1)

        # Calculate every gradient using the same parameter values.
        weight_grads = [None] * len(ws)
        bias_grads = [None] * len(bs)

        for i in reversed(range(len(ws))):
            # Derivative of tanh(z): 1 - tanh(z)^2
            delta = delta * (1 - activations[i + 1] ** 2)

            # W has shape (out_features, in_features).
            weight_grads[i] = delta.T @ activations[i]
            bias_grads[i] = delta.sum(dim=0)

            # Propagate through the current layer before updating.
            if i > 0:
                delta = delta @ ws[i]

        # Simultaneous gradient-descent update.
        ws = [
            w - learning_rate * dw
            for w, dw in zip(ws, weight_grads)
        ]
        bs = [
            b - learning_rate * db
            for b, db in zip(bs, bias_grads)
        ]

    # Final forward pass using trained parameters.
    h = inputs
    for w, b in zip(ws, bs):
        h = torch.tanh(h @ w.T + b)

    predictions = h.squeeze(-1)
    final_loss = ((predictions - targets) ** 2).sum()

    return predictions, final_loss, ws, bs, loss_history
