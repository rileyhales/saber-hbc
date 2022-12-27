import numpy as np


def nse(x1: np.array, x2: np.array):
    """
    Nash-Sutcliffe efficiency coefficient

    Args:
        x1: array of observed data
        x2: array of simulated data

    Returns:

    """
    return 1 - np.sum((x1 - x2) ** 2) / np.sum((x1 - np.mean(x1)) ** 2)


def kge2012(x1: np.array, x2: np.array) -> float:
    """
    Kling-Gupta efficiency coefficient (2012)

    Args:
        x1: array of observed data
        x2: array of simulated data

    Returns:
        float
    """
    r = np.corrcoef(x1, x2)[0, 1]
    alpha = np.std(x2) / np.std(x1)
    beta = np.mean(x2) / np.mean(x1)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
