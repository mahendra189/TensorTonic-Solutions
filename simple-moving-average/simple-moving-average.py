def simple_moving_average(values: list, window_size: int) -> list:
    """
    Returns the mean of every complete sliding window.
    """
    # Write code here
    sma = []
    k = window_size
    for i in range(0,len(values)-k+1):
        print(i)
        sma.append(sum(values[i:i+k])/k)
    return sma