import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code here
    w = np.array(w,dtype=float)
    g = np.array(g,dtype=float)
    G = np.array(G,dtype=float)
    
    G_n = G + g**2

    w_n = w - lr*(g)/(np.sqrt(G_n + eps))
    return {
        "new_w":w_n,
        "new_G":G_n
    }