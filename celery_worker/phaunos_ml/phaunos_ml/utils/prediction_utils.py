import numpy as np


##################################################
# Temporal integration of short-term predictions #
##################################################

def mean_pooling(predictions):
    """Compute the average of all predictions per class.

    Args:
        predictions: short-term predicted probabilities per class.
            Shape = (n_examples, n_classes)
    """

    return np.mean(predictions, axis=0)

def max_pooling(predictions):
    """Compute the max probability per class.

    Args:
        predictions: short-term predicted probabilities per class.
            Shape = (n_examples, n_classes)
    """

    return np.max(predictions, axis=0)
