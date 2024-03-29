"""

train.py

Train a MLP Classifier on MNIST using
Gradient Descent or Newtons method

"""

from jax import random as jrandom
from get_data import get_mnist_data


def parse_args() -> argparse.Namespace:
    """ Parse CL arguments """
    parser = argparse.ArgumentParser(description='Train a MLP classifier on MNIST using Gradient descent or Newtons method')
    parser.add_argument('--method', choices=['grad', 'newton'], help='Choose between Gradient descent and Newtons method')
    parser.add_argument('--seed', help='Random seed', type=int, default=99)
    


if __name__ == '__main__':

    # parse CL arguments
    args = parse_args()

    # create a random key
    prng_key = jrandom.key(args.seed)

    # get data
    X_train, y_train, X_test, y_test = get_mnist_data(prng_key, train_split=0.9)

