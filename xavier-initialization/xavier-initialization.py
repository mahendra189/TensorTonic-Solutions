def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    limit = math.sqrt(6/(fan_in+fan_out))
    # W = np.random.uniform(-limit,limit,(fan_in,fan_out))

    W = np.array(W)
    return  W * 2*limit - limit