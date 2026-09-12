import numpy as np

def cross_entropy_loss(
    y_true: list[int],
    y_pred: list[list[float]]
) -> float:

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Probability assigned to the correct class
    correct_probs = y_pred[np.arange(len(y_true)), y_true]

    # Avoid log(0)
    correct_probs = np.clip(correct_probs, 1e-15, 1.0)

    # Cross entropy
    loss = -np.log(correct_probs)

    # Mean loss
    return float(np.mean(loss))