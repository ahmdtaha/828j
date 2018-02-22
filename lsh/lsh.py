"""
Library functions to produce locality sensitive hashes using randomized projection (SimHash) [1].
This module does not store hashes or bases, it is up to the user to efficiently store the
data.

1. Charikar, Moses S. "Similarity estimation techniques from rounding algorithms." Proceedings of the thiry-fourth annual ACM symposium on Theory of computing. ACM, 2002.
"""

import numpy as np
from bitarray import bitarray


def generate_basis(n, dim):
    """
    Generate a new random basis for n-bit hashes (e.g. n random hyperplanes) for dim-dimensional data
    :param n: The number of bits that hashes
    :param dim: The dimensionality of the data to be hashed
    :return: A list of n hyperplanes that can be used to produce hashes
    """
    return [np.linalg.norm(np.random.randn(dim)) for i in range(n)]


def lsh(v, basis):
    """
    Hash a given vector v against the given basis. The hyperplanes in basis must have the same dimension
    as the vector v
    :param v: The vector to hash
    :param basis: A list of hyperplanes to use as a hashing basis
    :return: The hash of the vector v
    """

    signs = [np.sign(np.dot(v, b)) for b in basis]
    hash = bitarray(len(signs))

    for i in range(len(signs)):
        hash[i] = signs[i] > 0

    return hash.tobytes()


def hamming_distance(h1, h2):
    """
    Computes the hamming distance between hashes h1 and h2
    :param h1: A locality sensitive hash
    :param h2: A second locality sensitive hash produced from the same basis as h1
    :return: The hamming distance between h1 and h2
    """
    d = 0
    v = h1 ^ h2
    while not v == 0:
        d = d + 1
        v = v & (v - 1)

    return d
