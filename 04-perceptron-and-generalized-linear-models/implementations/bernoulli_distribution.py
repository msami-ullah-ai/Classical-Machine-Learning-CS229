import numpy as np


def bernoulli_probability(y, phi):

    probability = (phi ** y) * ((1 - phi) ** (1 - y))

    return probability


phi = 0.8

values = [0, 1]

for y in values:

    probability = bernoulli_probability(y, phi)

    print(f"P(y={y}) = {probability:.4f}")
