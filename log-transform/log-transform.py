import math

def log_transform(values: list) -> list:
    """
    Returns the log1p-transformed values rounded to four decimals.
    """
    return np.log1p(values)