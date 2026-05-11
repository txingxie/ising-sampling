import math
import random
import numpy as np


def critical_temperature():
    return 2.0 / math.log(1.0 + math.sqrt(2.0))


def wolff_update(spins, T):
    N = spins.shape[0]
    p_add = 1.0 - math.exp(-2.0 / T)

    i0 = random.randrange(N)
    j0 = random.randrange(N)
    seed_spin = spins[i0, j0]

    cluster = {(i0, j0)}
    stack = [(i0, j0)]

    while stack:
        i, j = stack.pop()
        neighbors = [
            ((i + 1) % N, j),
            ((i - 1) % N, j),
            (i, (j + 1) % N),
            (i, (j - 1) % N),
        ]

        for ni, nj in neighbors:
            if (ni, nj) not in cluster and spins[ni, nj] == seed_spin:
                if random.random() < p_add:
                    cluster.add((ni, nj))
                    stack.append((ni, nj))

    for i, j in cluster:
        spins[i, j] *= -1

    return spins


def generate_ising_samples(num_samples, N=16, burn_in=1000, thin=10):
    T = critical_temperature()
    spins = np.random.choice([-1, 1], size=(N, N)).astype(np.int8)

    for _ in range(burn_in):
        wolff_update(spins, T)

    samples = []

    for _ in range(num_samples):
        for _ in range(thin):
            wolff_update(spins, T)
        samples.append(spins.copy())

    return np.stack(samples, axis=0)