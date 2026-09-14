from statistics import median
def moving_median(values: list, window_size: int) -> list:
    """
    Returns the median of every complete sliding window.
    """
    # Write code here
    sma = []
    k = window_size
    for i in range(0,len(values)-k+1):
        sma.append(median(values[i:i+k]))
    return sma