def cumulative_returns(returns: list) -> list:
    """
    Returns the compounded cumulative return after every period.
    """
    # Write code here
    wt = 1.0
    cr = []

    for i in range(len(returns)):
        w = wt * (1 + returns[i])
        cr.append(w-1)
        wt = w
    return cr