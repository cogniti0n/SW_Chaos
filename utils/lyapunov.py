import math
import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx

from utils.swneural import sw_adj, sw_nogen


def _sech2(x):
    return 1.0 / (jnp.cosh(x) ** 2)


def lle_vector_field(t, y, args):
    """
    y = concat([x, v]) where x,v shape (n,)
    args = adj (n,n)
    """
    adj = args
    n = adj.shape[0]
    x = y[:n]
    v = y[n:]

    phi = jnp.tanh(x)
    dx = -x + adj @ phi
    dv = -v + adj @ (_sech2(x) * v)

    return jnp.concatenate([dx, dv])


def _compute_lle_inner(
    key: jax.Array,
    adj: jnp.ndarray,
    t_transient: float = 10.0,
    t_total: float = 300.0,
    dt0: float = 0.1,
    seg_len: float = 2.0,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    n = adj.shape[0]
    term = dfx.ODETerm(lle_vector_field)
    solver = dfx.Dopri5()
    steps = dfx.PIDController(rtol=rtol, atol=atol)

    x0 = jax.random.uniform(key, (n,), minval=0.0, maxval=1.0)
    v0 = jax.random.normal(jax.random.fold_in(key, 1), (n,))
    v0 /= jnp.linalg.norm(v0) + 1e-12

    y0 = jnp.concatenate([x0, v0])

    sol_tr = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_transient,
        dt0=dt0,
        y0=y0,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=steps,
        args=adj,
    )
    y = jnp.asarray(sol_tr.ys).squeeze(axis=0)

    x = y[:n]
    v = y[n:]
    v /= jnp.linalg.norm(v) + 1e-12

    nseg = int(math.floor(t_total / seg_len))
    t = 0.0
    sum_log = 0.0

    for _ in range(nseg):
        y0 = jnp.concatenate([x, v])

        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=seg_len,
            dt0=dt0,
            y0=y0,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=steps,
            args=adj,
        )
        y1 = jnp.asarray(sol.ys).squeeze(axis=0)
        x = y1[:n]
        v = y1[n:]
        nv = jnp.linalg.norm(v) + 1e-12
        sum_log += jnp.log(nv)
        v /= nv
        t += seg_len

    lle = sum_log / (t + 1e-12)
    return lle


def compute_lle(
    key: jax.Array,
    adj: jnp.ndarray,
    t_transient: float = 10.0,
    t_total: float = 300.0,
    dt0: float = 0.1,
    seg_len: float = 2.0,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    lle = _compute_lle_inner(
        key,
        adj,
        t_transient=t_transient,
        t_total=t_total,
        dt0=dt0,
        seg_len=seg_len,
        rtol=rtol,
        atol=atol,
    )
    return float(lle)


def create_adj_batch(
    key: jax.Array,
    n: int,
    beta: float,
    p: float,
    eirat: float,
    numsamples: int,
):
    model_keys = jax.random.split(key, numsamples)
    seed_gen_key = jax.random.fold_in(key, data=424242)
    seeds_gpu = jax.random.randint(
        seed_gen_key, shape=(numsamples,), minval=0, maxval=np.iinfo(np.int32).max
    )
    seeds_cpu = np.array(seeds_gpu)

    adj_storage = []
    for i in range(numsamples):
        rng = np.random.default_rng(seeds_cpu[i])
        adj_np = sw_adj(rng, n, p, beta)
        adj_masked = sw_nogen(rng, adj_np, eirat)
        adj_storage.append(adj_masked)

    adj_batch_np = np.stack(adj_storage)
    return adj_batch_np, model_keys


def shard_data(x: jnp.ndarray) -> jnp.ndarray:
    """
    Reshapes array for pmap: (B, ...) -> (Num_Devices, B_per_Device, ...)
    """
    num_devices = jax.local_device_count()
    batch_size = x.shape[0]

    if batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by device count ({num_devices})"
        )

    return x.reshape((num_devices, batch_size // num_devices) + x.shape[1:])


def unshard_data(x: jnp.ndarray) -> jnp.ndarray:
    """
    Collapses pmap output: (Num_Devices, B_per_Device, ...) -> (B, ...)
    """
    return x.reshape((-1,) + x.shape[2:])


local_lle_solver = jax.vmap(
    _compute_lle_inner, in_axes=(0, 0, None, None, None, None, None, None)
)
parallel_lle_solver = jax.pmap(
    local_lle_solver,
    in_axes=(0, 0, None, None, None, None, None, None),
    static_broadcasted_argnums=(2, 3, 4, 5, 6, 7),
)


def run_lle_multi_gpu(
    key: jax.Array,
    n: int,
    beta: float,
    eirat: float,
    numsamples: int,
    *,
    p: float = 0.07,
    t_transient: float = 10.0,
    t_total: float = 300.0,
    dt0: float = 0.1,
    seg_len: float = 2.0,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    num_devices = jax.local_device_count()
    if numsamples % num_devices != 0:
        raise ValueError(f"numsamples ({numsamples}) / GPU count ({num_devices})")

    adj_batch_np, model_keys = create_adj_batch(key, n, beta, p, eirat, numsamples)
    adj_batch_jax = jnp.asarray(adj_batch_np)

    adj_sharded = shard_data(adj_batch_jax)
    keys_sharded = shard_data(model_keys)

    sharded_output = parallel_lle_solver(
        keys_sharded,
        adj_sharded,
        t_transient,
        t_total,
        dt0,
        seg_len,
        rtol,
        atol,
    )
    lle = unshard_data(sharded_output)
    return np.asarray(lle)
