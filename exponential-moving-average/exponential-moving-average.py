def exponential_moving_average(values: list, alpha: float) -> list:
    """
    Returns the exponential moving average at every position.
    """
    # Write code here
    ema = []
    ema.append(values[0])

    for v in values[1:]:
        ema.append((
            alpha * v + (1-alpha) * ema[-1]
        ))
    return ema