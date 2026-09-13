import numpy as np

def naive_bayes_bernoulli(X_train: list, y_train: list, X_test: list) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]

    log_posteriors = np.zeros((len(X_test), n_classes))

    for c_idx, c in enumerate(classes):
        X_c = X_train[y_train == c]
        n_c = len(X_c)

        # P(class)
        log_prior = np.log(n_c / len(X_train))

        # P(feature=1 | class), with Laplace smoothing
        prob = (np.sum(X_c, axis=0) + 1) / (n_c + 2)

        log_prob_1 = np.log(prob)
        log_prob_0 = np.log(1 - prob)

        for i, x in enumerate(X_test):
            log_likelihood = np.sum(
                x * log_prob_1 +
                (1 - x) * log_prob_0
            )

            log_posteriors[i, c_idx] = log_prior + log_likelihood

    return np.round(log_posteriors,4)