import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty((0,0),dtype=int)
    # Your code here
    if max_len is not None:
        target_len = max_len
    else:
        target_len = max(len(s) for s in seqs)

    results = []
    for seq in seqs:
        seq = seq[:target_len]
        if len(seq) < target_len:
            seq = seq + [pad_value]* (target_len - len(seq))
        results.append(seq)
    return np.array(results)
    