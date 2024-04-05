"""

get_data.py

Load the MNIST dataset

"""

from typing import Tuple

import jax.numpy as jnp
import jax.random as jrandom
from jax.typing import ArrayLike
from sklearn.datasets import fetch_openml, load_digits


def train_test_split(X: ArrayLike, y: ArrayLike, prng_key: jrandom.PRNGKey,
                     train_frac: float = 0.9) -> Tuple[ArrayLike]:
    """ Split X and y into train and test set """
    
    # split into train/test set
    shuffled_indices = jrandom.permutation(prng_key, X.shape[0])
    X = X[shuffled_indices]
    split_idx = int(X.shape[0] * train_frac)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test


def get_digits_data(prng_key: jrandom.PRNGKey, train_frac: float = 0.9) -> Tuple[ArrayLike]:
    """ Load the digits dataset (8x8 pixel images of handwritten digits), split it into
        train and test part. Return the data as a tuple of jax.numpy arrays """
    
    X, y = load_digits(return_X_y=True)
    # normalize the images
    X = X / 16.0
    # cast targets to integers
    y = y.astype(int)
   
    return train_test_split(jnp.array(X), jnp.array(y), prng_key, train_frac)



def get_mnist_data(prng_key: jrandom.PRNGKey, train_frac: float = 0.9) -> Tuple[ArrayLike]:
    """ Load the MNIST dataset and split it into a train and
        test part. Return the data as jax.numpy arrays """

    mnist = fetch_openml('mnist_784')
    X, y = mnist['data'].to_numpy(), mnist['target'].to_numpy()

    # normalize the images
    X = X / 255.0
    # cast targets to integers
    y = y.astype(int)
    
    return train_test_split(jnp.array(X), jnp.array(y), prng_key, train_frac)





