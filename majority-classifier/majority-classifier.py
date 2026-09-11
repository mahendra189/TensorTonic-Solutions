import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    # Write code here
    labels ,count = np.unique(y_train,return_counts=True)
    i = np.max(count)
    majority = labels[count == i ].max()
    return np.full(len(X_test),majority)
    