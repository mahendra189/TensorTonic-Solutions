def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """
    used = series
    # Write code here
    for i in range(order):
        diff = []
        
        for i in range(1,len(used)):
            diff.append(used[i]-used[i-1])
        used = diff
    return diff
    
        