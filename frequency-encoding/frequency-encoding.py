import numpy as np
def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    # Write code here
    labels, counts = np.unique(values, return_counts=True)

    m = {l: counts[i] / len(values) for i, l in enumerate(labels)}

    return [m[v] for v in values]