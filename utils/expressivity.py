import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx

from utils.swneural import sw_adj, sw_nogen


def settle_term():
    def vf_on(t, x, args):
        adj, u, img = args  # adj: (n,n), u:(n,), img:(n,)
        return -x + adj @ jnp.tanh(x) + u * img

    def vf_off(t, x, args):
        adj = args
        return -x + adj @ jnp.tanh(x)

    return dfx.ODETerm(vf_on), dfx.ODETerm(vf_off)


def make_circle_inputs(key: jax.Array, n: int, n_theta: int = 256, alpha: float = 1.0):
    key0, key1 = jax.random.split(key)
    u0 = jax.random.normal(key0, (n,))
    u1 = jax.random.normal(key1, (n,))

    u0 /= jnp.linalg.norm(u0) + 1e-12
    u1 = u1 - (u1 @ u0) * u0
    u1 /= jnp.linalg.norm(u1) + 1e-12

    thetas = jnp.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    I = alpha * (
        jnp.cos(thetas)[:, None] * u0[None, :] + jnp.sin(thetas)[:, None] * u1[None, :]
    )
    return thetas, I


def solve_settle_single(
    adj: jnp.ndarray,
    u: jnp.ndarray,
    inp: jnp.ndarray,
    t_on: float,
    t_off: float,
    dt: float,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """
    dx/dt = -x + adj@tanh(x) + u*inp  (inp is static in time)
    return x(t_settle)
    """
    n = adj.shape[0]
    x0 = jnp.zeros((n,), dtype=jnp.float32)

    term_on, term_off = settle_term()
    solver = dfx.Dopri5()
    steps = dfx.PIDController(rtol=rtol, atol=atol)

    sol_on = dfx.diffeqsolve(
        term_on,
        solver,
        t0=0.0,
        t1=t_on,
        dt0=dt,
        y0=x0,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=steps,
        args=(adj, u, inp),
    )
    x_on = jnp.asarray(sol_on.ys).squeeze()
    sol_off = dfx.diffeqsolve(
        term_off,
        solver,
        t0=0.0,
        t1=t_off,
        dt0=dt,
        y0=x_on,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=steps,
        args=adj,
    )
    return jnp.asarray(sol_off.ys).squeeze()  # (n,)


solve_settle_batch = jax.jit(
    jax.vmap(
        solve_settle_single, in_axes=(None, None, 0, None, None, None, None, None)
    ),
    static_argnums=(3, 4, 5, 6, 7),
)


def circle_expressivity_metrics(X: np.ndarray, thetas: np.ndarray):
    """
    X: (K, n) states on the manifold (x(theta_k))
    thetas: (K,)
    Returns: LE, LG, mean_kappa
    """
    X = np.asarray(X, dtype=np.float64)
    thetas = np.asarray(thetas, dtype=np.float64)
    K, n = X.shape
    dtheta = float(thetas[1] - thetas[0])

    X_prev = np.roll(X, 1, axis=0)
    X_next = np.roll(X, -1, axis=0)

    v = (X_next - X_prev) / (2.0 * dtheta)
    a = (X_next - 2.0 * X + X_prev) / (dtheta**2)

    vnorm = np.linalg.norm(v, axis=1) + 1e-12
    LE = np.sum(vnorm) * dtheta

    vhat = v / vnorm[:, None]
    vhat_prev = np.roll(vhat, 1, axis=0)
    vhat_next = np.roll(vhat, -1, axis=0)
    dvhat = (vhat_next - vhat_prev) / (2.0 * dtheta)
    LG = np.sum(np.linalg.norm(dvhat, axis=1)) * dtheta

    v2 = np.sum(v * v, axis=1)
    a2 = np.sum(a * a, axis=1)
    va = np.sum(v * a, axis=1)
    kappa = np.sqrt(np.maximum(v2 * a2 - va * va, 0.0)) / (vnorm**3)
    mean_kappa = float(np.mean(kappa))

    return float(LE), float(LG), mean_kappa


def run_circle_expressivity(
    seed: int = 0,
    n: int = 300,
    p: float = 0.07,
    beta: float = 0.12,
    eirat: float = 0.8,
    n_theta: int = 256,
    alpha: float = 1.0,
    t_on: float = 5.0,
    t_off: float = 45.0,
    dt: float = 0.1,
):
    """
    Builds a reservoir adjacency, evaluates the circle manifold,
    returns (LE, LG, mean_kappa).
    """
    key = jax.random.key(seed)
    rng = np.random.default_rng(seed)

    adj_bool = sw_adj(rng, n, p, beta)
    adj = sw_nogen(rng, adj_bool, eirat)
    adj = jnp.asarray(adj, dtype=jnp.float32)

    u = jnp.ones((n,), dtype=jnp.float32) / jnp.sqrt(n)
    thetas, I = make_circle_inputs(
        jax.random.fold_in(key, 1), n, n_theta=n_theta, alpha=alpha
    )
    X = np.asarray(
        solve_settle_batch(adj, u, I, t_on, t_off, dt, 1e-5, 1e-5)
    ).squeeze()  # (K,n)

    LE, LG, kappa = circle_expressivity_metrics(X, np.asarray(thetas))
    return LE, LG, kappa
