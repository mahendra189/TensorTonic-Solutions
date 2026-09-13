import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    labels , counts = np.unique(tokens,return_counts=True)
    indexing = {
        l:c for l,c in zip(labels,counts)
    }
    

    n = ([indexing.get(v,0) for v in vocab])
    return np.array(n)
    