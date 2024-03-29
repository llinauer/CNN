"""

get_data.py

Load the MNIST dataset

"""

from typing import Tuple

import jax.numpy as jnp
import jax.random as jrandom
from jax.typing import ArrayLike
from sklearn.datasets import fetch_openml


def get_mnist_data(prng_key: jrandom.PRNGKey, train_split: float = 0.9) -> Tuple[ArrayLike]:
    """ Load the MNIST dataset and split it into a train and
        test part. Return the data as jax.numpy arrays """

    mnist = fetch_openml('mnist_784')
    X, y = mnist['data'].to_numpy(), mnist['target'].to_numpy()

    # normalize the images
    X = X / 255.0
    # cast targets to integers
    y = y.astype(int)

    # split into train/test set
    shuffled_indices = jrandom.permutation(prng_key, X.shape[0])
    X = X[shuffled_indices]
    split_idx = int(X.shape[0] * train_split)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], X[split_idx:]

    # cast to jax.numpy arrays and return
    return jnp.array(X_train), jnp.array(y_train), jnp.array(X_test), jnp.array(y_test)




