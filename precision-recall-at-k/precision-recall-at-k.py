def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    
    ratio_t_r = list(set(recommended[:k]) & set(relevant))
    return [len(ratio_t_r)/k , len(ratio_t_r)/len(relevant)]