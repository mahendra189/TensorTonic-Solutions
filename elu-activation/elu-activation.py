import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # Write code here
    l = []
    for v in x:
        if v > 0:
            l.append(v)
        else:
            l.append(alpha * (math.exp(v) - 1))
    return l