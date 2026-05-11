import numpy as np


def energy_per_site(samples):
    right = np.roll(samples, shift=-1, axis=2)
    down = np.roll(samples, shift=-1, axis=1)

    energy = -(samples * right + samples * down).sum(axis=(1, 2))
    N = samples.shape[1]

    return energy / (N * N)


def magnetization(samples):
    return samples.mean(axis=(1, 2))


def correlation_function(samples, max_r=None):
    B, N, _ = samples.shape

    if max_r is None:
        max_r = N // 2

    corrs = []

    for r in range(1, max_r + 1):
        shifted_x = np.roll(samples, shift=-r, axis=2)
        shifted_y = np.roll(samples, shift=-r, axis=1)

        corr_x = (samples * shifted_x).mean()
        corr_y = (samples * shifted_y).mean()

        corrs.append(0.5 * (corr_x + corr_y))

    return np.array(corrs)


def wasserstein_1d(a, b, num_points=1000):
    qs = np.linspace(0.0, 1.0, num_points)
    aq = np.quantile(a, qs)
    bq = np.quantile(b, qs)
    return np.mean(np.abs(aq - bq))


def sampler_score(true_samples, generated_samples):
    random_samples = np.random.choice([-1, 1], size=generated_samples.shape)

    true_energy = energy_per_site(true_samples)
    gen_energy = energy_per_site(generated_samples)
    rand_energy = energy_per_site(random_samples)

    true_mag = np.abs(magnetization(true_samples))
    gen_mag = np.abs(magnetization(generated_samples))
    rand_mag = np.abs(magnetization(random_samples))

    true_corr = correlation_function(true_samples)
    gen_corr = correlation_function(generated_samples)
    rand_corr = correlation_function(random_samples)

    energy_error = wasserstein_1d(true_energy, gen_energy)
    energy_baseline = wasserstein_1d(true_energy, rand_energy)

    mag_error = wasserstein_1d(true_mag, gen_mag)
    mag_baseline = wasserstein_1d(true_mag, rand_mag)

    corr_error = np.mean(np.abs(true_corr - gen_corr))
    corr_baseline = np.mean(np.abs(true_corr - rand_corr))

    eps = 1e-8

    normalized_energy_error = energy_error / (energy_baseline + eps)
    normalized_mag_error = mag_error / (mag_baseline + eps)
    normalized_corr_error = corr_error / (corr_baseline + eps)

    total_error = (
        0.25 * normalized_energy_error
        + 0.25 * normalized_mag_error
        + 0.50 * normalized_corr_error
    )

    score = 100.0 * np.exp(-total_error)

    return score, {
        "energy_error_normalized": normalized_energy_error,
        "magnetization_error_normalized": normalized_mag_error,
        "correlation_error_normalized": normalized_corr_error,
    }