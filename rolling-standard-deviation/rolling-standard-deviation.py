import math

def rolling_std(values: list, window_size: int) -> list:
    """
    Returns the population standard deviation of every complete window.
    """
    # Write code here
    k = window_size
    rstd = []
    for i in range(len(values)-k+1):
        wmean = sum(values[i:i+k])/k
        rstd.append(
            math.sqrt(
                (1/k)* 
                sum(
                    [(x - wmean)**2 for x in values[i:i+k]]
                )
            )
        )
    return rstd