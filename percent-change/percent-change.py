def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    # Write code here
    c = []
    for i in range(1,len(series)):
        if series[i-1] == 0:
            c.append(0.0)
            continue
        c.append((series[i]-series[i-1])/series[i-1])
    return c
        