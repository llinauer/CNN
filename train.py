"""

train.py

Train a MLP Classifier on MNIST using
Gradient Descent or Newtons method

"""

import argparse
from typing import List, Tuple

from jax import random as jrandom
from jax.typing import ArrayLike
from jax.scipy.special import logsumexp
import jax.numpy as jnp

from get_data import get_mnist_data


def parse_args() -> argparse.Namespace:
    """ Parse CL arguments """
    parser = argparse.ArgumentParser(description='Train a MLP classifier on MNIST using Gradient descent or Newtons method')
    parser.add_argument('--method', choices=['grad', 'newton'], help='Choose between Gradient descent and Newtons method')
    parser.add_argument('--seed', help='Random seed', type=int, default=99)

    return parser.parse_args()


def init_random_weights(layer_sizes: List[int], prng_key: jrandom.PRNGKey) -> List[Tuple]:
    """ Initialize the weights of a MLP with hidden dimensions specified in layer_sizes """
    keys = jrandom.split(prng_key, len(layer_sizes))
    return [init_dense_layer(in_dim, out_dim, key) for in_dim, out_dim, key in zip(layer_sizes[:-1], layer_sizes[1:], keys)]


def init_dense_layer(input_dim, output_dim, prng_key, scale=1e-2) -> Tuple[ArrayLike]:
    """ Initialize a fully-connected (dense) NN layer with input_dim input neurons
        and output_dim output neurons """

    w_key, b_key = jrandom.split(prng_key)
    return scale * jrandom.normal(w_key, (output_dim, input_dim)), scale * jrandom.normal(b_key, (output_dim,))


def relu(x):
    """ ReLU activation function """
    return jnp.maximum(0, x)


def predict(params: List[Tuple], image: ArrayLike):
    """ Predict which out of 10 digits the input image belongs to """
    
    activations = image
    
    # loop over all layers up to the last
    for w, b in params[:-1]:
        out = jnp.dot(w, activations) + b
        activations = relu(out)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


if __name__ == '__main__':

    # parse CL arguments
    args = parse_args()

    # create a random key
    prng_key = jrandom.key(args.seed)
    prng_key, data_key = jrandom.split(prng_key)

    # get data
    X_train, y_train, X_test, y_test = get_mnist_data(data_key, train_split=0.9)

