def weighted_moving_average(values: list, weights: list) -> list:
    """
    Returns the weighted average of every complete window.
    """
    # Write code here
    wma = []
    k = len(weights)
    s = sum(weights)
    for i in range(len(values) - k + 1):
        window = values[i:i+k]

        weighted_sum = sum(
            w * v
            for w, v in zip(weights, window)
        )

        wma.append(weighted_sum / s)

    return wma