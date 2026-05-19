import argparse

import diffrax as dfx
import jax
import jax.numpy as jnp


def lorenz96_vf(t, x, args):
    F = args
    return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + F


def generate_lorenz96(ts, n_dim=40, F=8.0, key=None):
    if key is None:
        key = jax.random.key(0)

    x0 = F + jax.random.normal(key, (n_dim,)) * 0.1

    term = dfx.ODETerm(lorenz96_vf)
    solver = dfx.Dopri5()
    saveat = dfx.SaveAt(ts=ts)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=x0,
        args=F,
        saveat=saveat,
    )
    return jnp.asarray(sol.ys).squeeze()


def simulate_reservoir_multi(adj, key, u, S_ts, ts, x0=None):
    n_res = adj.shape[0]
    if x0 is None:
        x0 = jax.random.uniform(key, (n_res,), minval=-0.1, maxval=0.1)

    interp = dfx.LinearInterpolation(ts=ts, ys=S_ts)

    def vf(t, x, args):
        s = interp.evaluate(t)
        return -x + adj @ jnp.tanh(x) + u @ s

    term = dfx.ODETerm(vf)
    sol = dfx.diffeqsolve(
        term,
        dfx.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=x0,
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )
    return jnp.asarray(sol.ys).squeeze()


def fit_ridge(
    x: jnp.ndarray,
    z: jnp.ndarray,
    alpha: float = 1e-4,
    bias: bool = True,
    standardize: bool = True,
):
    x = x.astype(jnp.float32)
    z = z.astype(jnp.float32)

    if standardize:
        mu = jnp.mean(x, axis=0, keepdims=True)
        sig = jnp.std(x, axis=0, keepdims=True) + 1e-6
        xs = (x - mu) / sig
    else:
        mu = jnp.zeros((1, x.shape[1]), dtype=x.dtype)
        sig = jnp.ones((1, x.shape[1]), dtype=x.dtype)
        xs = x

    if bias:
        xs = jnp.concatenate([xs, jnp.ones((xs.shape[0], 1), dtype=xs.dtype)], axis=1)

    d = xs.shape[1]
    xtx = (xs.T @ xs) / xs.shape[0]
    xtz = (xs.T @ z) / xs.shape[0]
    reg = alpha * jnp.eye(d, dtype=xs.dtype)
    if bias:
        reg = reg.at[-1, -1].set(0.0)
    w = jnp.linalg.solve(xtx + reg, xtz)
    return w, mu, sig


def predict_ridge(
    x: jnp.ndarray,
    w: jnp.ndarray,
    mu: jnp.ndarray,
    sig: jnp.ndarray,
    bias: bool = True,
    standardize: bool = True,
):
    x = x.astype(jnp.float32)
    if standardize:
        xs = (x - mu) / sig
    else:
        xs = x
    if bias:
        xs = jnp.concatenate([xs, jnp.ones((xs.shape[0], 1), dtype=xs.dtype)], axis=1)
    return xs @ w


def ridge_readout(
    x: jnp.ndarray,
    z: jnp.ndarray,
    alpha: float = 1e-4,
    bias: bool = True,
    standardize: bool = True,
):
    return fit_ridge(x, z, alpha=alpha, bias=bias, standardize=standardize)


def rmse(y_true, y_pred):
    return jnp.sqrt(jnp.mean((y_true - y_pred) ** 2))


def chaotic_prediction_rmse(
    x: jnp.ndarray,  # (T,N) reservoir states
    s: jnp.ndarray,  # (T,) true chaotic signal
    dt: float,
    horizons: list[float],
    washout_time: float = 50.0,
    alpha: float = 1e-4,
    bias: bool = True,
    standardize: bool = True,
):
    washout_steps = int(washout_time / dt)
    xw = x[washout_steps:]  # (Tw,N)
    sw = s[washout_steps:]  # (Tw,)
    Tw = xw.shape[0]

    rmses = []
    for H in horizons:
        k = int(jnp.round(H / dt))
        if k <= 0 or k >= Tw:
            rmses.append(jnp.nan)
            continue

        Xtr = xw[:-k]
        ztr = sw[k:]  # target is future value

        w, mu, sig = ridge_readout(
            Xtr,
            ztr,
            alpha=alpha,
            bias=bias,
            standardize=standardize,
        )
        yhat = predict_ridge(
            Xtr,
            w,
            mu,
            sig,
            bias=bias,
            standardize=standardize,
        )
        rmses.append(rmse(ztr, yhat))

    return jnp.asarray(rmses)
