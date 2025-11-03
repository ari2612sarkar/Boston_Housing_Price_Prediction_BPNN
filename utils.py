import numpy as np

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def create_folds(X, y, k):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k
    folds = []
    for i in range(k):
        test_idx = indices[i*fold_size:(i+1)*fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        folds.append((train_idx, test_idx))
    return folds
