import numpy as np
import jax.numpy as jnp

import jax

DEFAULT_RELXATION_IDX = 100


def corr(X, relaxation_time=DEFAULT_RELXATION_IDX):
    assert X.ndim == 4

    phi = np.tanh(X[:, :, relaxation_time:, :])
    phi_mean = np.mean(phi, axis=(1))  # shape: (s, t, n)
    phi_last = phi_mean[:, -1, :]  # shape: (s, n,)
    phi_reversed = phi_mean[:, ::-1, :]  # shape: (s, t, n)

    N = phi_mean.shape[-1]

    corr = np.einsum("sn,stn->st", phi_last, phi_reversed) / N
    return corr


def tau(corr, dt):
    # corr: (s, t)

    norm_corr = corr / corr[:, [0]]
    t_val = dt * np.sum(norm_corr**2, axis=1)
    return t_val


def corr_keepaxis(x, relaxation_time):
    assert x.ndim == 4

    x = x[:, :, relaxation_time:, :]  # shape: (S, B, T, N)
    phi = np.tanh(x)  # shape: (S, B, T, N)

    phi_last = phi[:, :, -1, :]  # shape: (S, B, N)
    phi_reversed = phi[:, :, ::-1, :]  # shape: (S, B, T, N)

    n_neurons = phi.shape[3]
    corr = np.einsum("sbn,sbtn->sbt", phi_last, phi_reversed) / n_neurons
    return corr


def tau_keepaxis(corr, dt):
    norm_corr = corr / corr[:, :, [0]]
    t_val = dt * np.sum(norm_corr**2, axis=2)
    return t_val


def threshold_pass_time(X, dt, perturbation_batchs: int):

    assert X.ndim == 4
    assert X.shape[1] == perturbation_batchs

    ntime = X.shape[2]
    n = X.shape[3]
    nsample = X.shape[0]

    perturbation_idx = jnp.split(jnp.arange(n), perturbation_batchs)

    threshold_pass = jnp.abs(X) > 0.01
    # (nsample, perturbation_batchs, ntime, n)
    first_cross = jnp.argmax(
        threshold_pass, axis=2
    )  # (nsample, perturbation_batchs, n)
    never_crossed = ~jnp.any(threshold_pass, axis=2)
    first_cross = jnp.where(never_crossed, ntime, first_cross)
    first_cross = jnp.array(first_cross)

    avg_first_cross_batch = jnp.zeros((nsample,))

    for i in range(perturbation_batchs):
        include_idx = jnp.ones(n, dtype=bool)
        batch_idx = perturbation_idx[i]
        include_idx = include_idx.at[batch_idx].set(False)

        first_cross_batch = first_cross[..., include_idx]

        avg_first_cross_batch += jnp.mean(first_cross_batch, axis=(1, 2))

    avg_first_cross = avg_first_cross_batch / perturbation_batchs

    return dt * avg_first_cross


def threshold_pass_time_keepaxis(x, dt, perturbation_batchs: int, eps: float = 0.01):

    assert x.ndim == 4
    assert x.shape[1] == perturbation_batchs

    ntime = x.shape[2]
    n = x.shape[3]
    nsample = x.shape[0]

    perturbation_idx = jnp.array_split(jnp.arange(n), perturbation_batchs)

    threshold_pass = jnp.abs(x) > eps
    # (nsample, perturbation_batchs, ntime, n)
    first_cross = jnp.argmax(
        threshold_pass, axis=2
    )  # (nsample, perturbation_batchs, n)
    never_crossed = ~jnp.any(threshold_pass, axis=2)
    first_cross = jnp.where(never_crossed, ntime, first_cross)
    first_cross = jnp.array(first_cross)  # (nsample, perturbation_batchs, n)

    avg_first_cross_batch = jnp.zeros((nsample, perturbation_batchs))

    for i in range(perturbation_batchs):
        include_idx = jnp.ones(n, dtype=bool)
        batch_idx = perturbation_idx[i]
        include_idx = include_idx.at[batch_idx].set(False)

        first_cross_batch = first_cross[:, i, include_idx]  # (nsample, num_include_idx)
        per_sample_batch_mean = jnp.mean(first_cross_batch, axis=1)  # (nsample,)

        avg_first_cross_batch = avg_first_cross_batch.at[:, i].set(
            per_sample_batch_mean
        )

    # avg_first_cross = jnp.mean(avg_first_cross_batch, axis=1)

    return dt * avg_first_cross_batch
