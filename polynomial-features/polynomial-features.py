import numpy as np
def polynomial_features(values: list, degree: int) -> list:
    """
    Returns powers from zero through degree for every value.
    """
    d = degree + 1
    # Write code here
    result = np.array([
        np.full(d, v) ** np.arange(d)
        for v in values
    ])
    return list(result)
    