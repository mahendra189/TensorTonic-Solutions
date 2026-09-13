def interaction_features(X: list) -> list:
    """
    Returns original features followed by unique pairwise products.
    """
    # Write code here
    lent = len(X[0])
    for i in range(len(X)):
        for j in range(lent):
            for k in range(lent):
                if j != k and j < k:
                    X[i].append(X[i][j]* X[i][k])
                
    return X