import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:

    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    N = Z1.shape[0]

    # Similarity matrix
    logits = (Z1 @ Z2.T) / temperature

    # Numerically stable log-softmax
    max_logits = np.max(logits, axis=1, keepdims=True)

    log_sum_exp = (
        max_logits
        + np.log(
            np.sum(
                np.exp(logits - max_logits),
                axis=1,
                keepdims=True
            )
        )
    )

    log_probs = logits - log_sum_exp

    # Positive pair = diagonal
    positive_log_probs = np.diag(log_probs)

    # InfoNCE
    loss = -np.mean(positive_log_probs)

    return float(loss)