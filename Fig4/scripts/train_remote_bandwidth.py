import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx


def block_indices(n: int, start: int, k: int) -> np.ndarray:
    start = int(start) % n
    return (np.arange(k, dtype=np.int32) + start) % n


def centered_block_indices(n: int, center: int, k: int) -> np.ndarray:
    """Return a contiguous block centered as closely as possible on ``center``."""
    start = int(center) - int(k) // 2
    return block_indices(n, start, k)


def fit_ridge(
    X: jnp.ndarray,
    y: jnp.ndarray,
    alpha: float = 1e-2,
    bias: bool = True,
    standardize: bool = True,
):
    X = X.astype(jnp.float32)
    y = y.astype(jnp.float32)

    if standardize:
        mu = jnp.mean(X, axis=0, keepdims=True)
        sig = jnp.std(X, axis=0, keepdims=True) + 1e-6
        Xs = (X - mu) / sig
    else:
        mu = jnp.zeros((1, X.shape[1]), dtype=X.dtype)
        sig = jnp.ones((1, X.shape[1]), dtype=X.dtype)
        Xs = X

    if bias:
        Xs = jnp.concatenate([Xs, jnp.ones((Xs.shape[0], 1), dtype=Xs.dtype)], axis=1)

    D = Xs.shape[1]
    XtX = (Xs.T @ Xs) / Xs.shape[0]
    Xty = (Xs.T @ y) / Xs.shape[0]
    reg = alpha * jnp.eye(D, dtype=Xs.dtype)
    if bias:
        reg = reg.at[-1, -1].set(0.0)
    w = jnp.linalg.solve(XtX + reg, Xty)
    return w, mu, sig


def predict_ridge(
    X: jnp.ndarray,
    w: jnp.ndarray,
    mu: jnp.ndarray,
    sig: jnp.ndarray,
    bias: bool = True,
    standardize: bool = True,
):
    X = X.astype(jnp.float32)
    if standardize:
        Xs = (X - mu) / sig
    else:
        Xs = X
    if bias:
        Xs = jnp.concatenate([Xs, jnp.ones((Xs.shape[0], 1), dtype=Xs.dtype)], axis=1)
    return Xs @ w


def rmse(y, yhat):
    return float(jnp.sqrt(jnp.mean((y - yhat) ** 2)))


def make_symbol_stream(
    T_total: float,
    dt: float,
    T_sym: float,
    rng: np.random.Generator,
    sigma_symbol_noise: float = 0.0,
):
    """
    Returns:
      ts_base: (T_steps,) times for interpolation
      s_ts:    (T_steps,) piecewise-constant signal with optional per-step noise
      s_sym:   (K,)       true symbol sequence (targets)
    """
    ts_base = np.arange(0.0, T_total + 1e-12, dt, dtype=np.float32)
    T_steps = ts_base.shape[0]
    K = int(np.floor(T_total / T_sym))  # number of full symbols

    s_sym = rng.uniform(-1.0, 1.0, size=(K,)).astype(np.float32)

    sym_idx = np.minimum((ts_base / T_sym).astype(np.int32), K - 1)
    s_ts = s_sym[sym_idx]

    if sigma_symbol_noise > 0:
        s_ts = s_ts + rng.normal(0.0, sigma_symbol_noise, size=s_ts.shape).astype(
            np.float32
        )

    return ts_base, s_ts.astype(np.float32), s_sym


def simulate_reservoir_saveat_zoh(
    key: jax.Array,
    adj: jnp.ndarray,
    B_in: jnp.ndarray,
    s_sym: np.ndarray,
    T_sym: float,
    save_ts: np.ndarray,
    dt0: float,
    rtol: float,
    atol: float,
):
    s_sym_j = jnp.asarray(s_sym, dtype=jnp.float32)
    save_ts_j = jnp.asarray(save_ts, dtype=jnp.float32)
    k_in = B_in.shape[1]

    def vf(t, x, args):
        adj_, B_in_, s_sym_ = args
        idx = jnp.floor(t / T_sym).astype(jnp.int32)
        idx = jnp.clip(idx, 0, s_sym_.shape[0] - 1)

        u_scalar = s_sym_[idx]
        u = jnp.full((k_in,), u_scalar, dtype=x.dtype)

        return -x + adj_ @ jnp.tanh(x) + B_in_ @ u

    t1 = float(np.asarray(save_ts)[-1])

    sol = dfx.diffeqsolve(
        dfx.ODETerm(vf),
        dfx.Dopri5(),
        t0=0.0,
        t1=t1,
        dt0=dt0,
        y0=jax.random.uniform(
            key,
            shape=(adj.shape[0],),
            minval=-0.5,
            maxval=0.5,
        ),
        args=(adj, B_in, s_sym_j),
        saveat=dfx.SaveAt(ts=save_ts_j),
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        max_steps=65536,
    )

    return jnp.asarray(sol.ys)
