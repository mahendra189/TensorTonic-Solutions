import numpy as np 

def cohens_kappa(rater1: list, rater2: list) -> float:
    """
    Returns Cohen's kappa as a float.
    """
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)

    classes = np.unique(np.concatenate([rater1, rater2]))
    N = len(rater1)

    cm = np.zeros((len(classes), len(classes)), dtype=int)

    class_to_index = {c: i for i, c in enumerate(classes)}

    for actual, pred in zip(rater1, rater2):
        cm[class_to_index[actual], class_to_index[pred]] += 1

    Po = np.trace(cm) / N

    actual_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)

    actual_probs = actual_counts / N
    pred_probs = pred_counts / N

    Pe = np.sum(actual_probs * pred_probs)

    if Pe == 1:
        return 1.0

    kappa = (Po - Pe) / (1 - Pe)

    return kappa