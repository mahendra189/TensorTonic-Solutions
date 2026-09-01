import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    if not 0<= p <1:
        raise ValueError("p must be in range [0,1)")
    x = np.array(x)
    keep_prob = 1-p
    if rng is not None:
        random_patterns = rng.random(x.shape)
    else:
        random_patterns = np.random.random(x.shape)
    dropout_pattern = (random_patterns >= p)/keep_prob
    output  = x* dropout_pattern
    return output,dropout_pattern
    
    

    