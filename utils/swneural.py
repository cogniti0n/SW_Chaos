import numpy as np
import networkx as nx

import os

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

import diffrax as dfx
import jax
import jax.numpy as jnp
from typing import Tuple
from jaxtyping import Array, Float, Int


def sw_adj(rng: np.random.Generator, n: int, p: float, beta: float) -> np.ndarray:
    """
    Generate small-world adjacency matrix

    Args:
        rng: numpy random number generator
        n: the number of nodes
        p: the overall edge density
        beta: the randomness factor

    Returns:
        The adjacency matrix of a small-world graph
    """
    # sanity check
    assert 0 <= p <= 1, "p must be in [0,1]"
    assert 0 <= beta <= 1, "beta must be in [0,1]"

    arange = np.arange(n)
    i, j = np.meshgrid(arange, arange, indexing="ij")

    mask = i != j
    d = np.abs(i - j)

    d_ij = np.minimum(d, n - d) / np.ceil(n / 2)
    p_ij = beta * p + (1 - beta) * np.heaviside(p - d_ij, 0.5)

    rand_vals = rng.random((n, n))

    return (rand_vals < p_ij) & mask


import scipy.sparse as sp


def sw_gen(rng: np.random.Generator, adj: np.ndarray, eirat: float):
    assert 0 <= eirat <= 1
    n = adj.shape[0]
    assert adj.shape == (n, n)

    is_exc = rng.random(n) < eirat
    node_sign = np.where(is_exc, 1.0, -1.0).astype(np.float32)
    adj_masked = adj.astype(np.float32) * node_sign[None, :]
    adj_sparse_topology = sp.csr_matrix(adj.T.astype(np.float32))
    G = nx.from_scipy_sparse_array(adj_sparse_topology, create_using=nx.DiGraph)

    nx.set_node_attributes(G, {i: bool(is_exc[i]) for i in range(n)}, name="type")
    return adj_masked, G


def sw_nogen(rng: np.random.Generator, adj: np.ndarray, eirat: float) -> np.ndarray:
    """
    Generate the weighted small-world adjacency matrix

    Args:
        seed: random seed
        adj: the unweighted adjacency matrix
        eirat: the E/I ratio

    Returns:
        A: The adjacency matrix of the network
    """
    # sanity check
    assert 0 <= eirat <= 1
    n = adj.shape[0]
    assert adj.shape == (n, n)

    is_exc = rng.random(n) < eirat
    node_sign = np.where(is_exc, 1.0, -1.0).astype(np.float32)
    adj_masked = adj.astype(np.float32) * node_sign[None, :]

    return adj_masked


def create_adj_batch(
    key,
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
    graph_storage = []

    # print(f"Generating {numsamples} adjacency matrices...")
    for i in range(numsamples):
        rng = np.random.default_rng(seeds_cpu[i])

        adj_np = sw_adj(rng, n, p, beta)
        adj_masked, graph = sw_gen(rng, adj_np, eirat)

        adj_storage.append(adj_masked)
        graph_storage.append(graph)

        # if (i + 1) % 5 == 0 or i == numsamples - 1:
        #    print(f"Generated {i + 1}/{numsamples} ({100 * (i + 1) / numsamples:.1f}%)")

    adj_batch_np = np.stack(adj_storage)

    return adj_batch_np, graph_storage, model_keys


def _dynnp(t, y, args):
    """
    The vector field equation: dy/dt = -y + A @ tanh(y)
    """
    adj = args
    return -y + adj @ jnp.tanh(y)


def solve_single_dynamics(
    adj: Float[Array, "n n"],
    perturbation: float,
    perturbation_idx: Int[Array, "k"],
    relaxation_time: float,
    tmax: float,
    dt: float,
) -> Float[Array, "ntime n"]:

    n = adj.shape[0]
    term = dfx.ODETerm(_dynnp)
    solver = dfx.Dopri5()
    stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    y0 = jnp.zeros((n,))
    equilibrium_sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=relaxation_time,
        dt0=0.1,
        y0=y0,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,
        args=adj,
    )
    equilibrium_state = jnp.asarray(equilibrium_sol.ys)[-1]

    perturbed_state = equilibrium_state.at[perturbation_idx].add(perturbation)

    saveat = dfx.SaveAt(ts=jnp.linspace(0, tmax, int(tmax / dt)))

    perturbed_sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tmax,
        dt0=dt,
        y0=perturbed_state,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=adj,
    )

    return jnp.asarray(perturbed_sol.ys)


batched_solver = jax.jit(
    jax.vmap(solve_single_dynamics, in_axes=(0, None, None, None, None, None)),
    static_argnums=(4, 5),
)


def run_perturbed(
    key,
    n: int,
    beta: float,
    eirat: float,
    perturbation: float,
    perturbation_idx_list: list[jnp.ndarray],
    tmax: float,
    numsamples: int,
    dt: float = 0.1,
    p: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the perturbed neuronal model

    Args:
        key: jax random key
        n: the number of nodes
        beta: randomness factor
        eirat: E/I ratio
        perturbation: the perturbation strength
        perturbation_idx_list: list of jnp.ndarray, each containing indices to perturb
        tmax: maximum time
        numsamples: number of samples
        dt: the timestep (default 0.1)
        p: edge density (default 0.07)

    Returns:
        X: dimensions (samples, perturbation batches, time, nodes), the dynamics results
        C: dimensions (samples,), average clustering coefficient of the graph
        L: dimensions (samples,), average path length of the graph
    """

    adj_batch_np, graph_storage, model_keys = create_adj_batch(
        key, n, beta, p, eirat, numsamples
    )
    adj_batch_jax = jnp.asarray(adj_batch_np)

    clst = np.zeros((numsamples,))
    avpl = np.zeros((numsamples,))
    for i in range(numsamples):
        clst[i] = nx.average_clustering(graph_storage[i])
        try:
            avpl[i] = nx.average_shortest_path_length(graph_storage[i])
        except nx.NetworkXException:
            avpl[i] = np.nan

    result_list = []

    for indices in perturbation_idx_list:

        batch_trajectories = batched_solver(
            adj_batch_jax, perturbation, indices, 10.0, tmax, dt
        )
        result_list.append(batch_trajectories)

    final_x = jnp.stack(result_list, axis=1)
    final_x_np = np.asarray(final_x)
    return final_x_np, clst, avpl


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


# batched solver and parallel solver
local_batch_solver = jax.vmap(
    solve_single_dynamics, in_axes=(0, None, None, None, None, None)
)
parallel_solver = jax.pmap(
    local_batch_solver,
    in_axes=(0, None, None, None, None, None),
    static_broadcasted_argnums=(4, 5),
)


def run_perturbed_multi_gpu(
    key: jax.Array,
    n: int,
    beta: float,
    eirat: float,
    perturbation: float,
    perturbation_idx_list: list[jnp.ndarray],
    tmax: float,
    numsamples: int,
    dt: float = 0.1,
    p: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the perturbed neuronal model on multiple GPU systems

    Args:
        key: jax random key
        n: the number of nodes
        beta: randomness factor
        eirat: E/I ratio
        perturbation: the perturbation strength
        perturbation_idx_list: list of jnp.ndarray, each containing indices to perturb
        tmax: maximum time
        numsamples: number of samples
        dt: the timestep (default 0.1)
        p: edge density (default 0.07)

    Returns:
        X: dimensions (samples, perturbation batches, time, nodes), the dynamics results
        C: dimensions (samples,), average clustering coefficient of the graph
        L: dimensions (samples,), average path length of the graph
    """

    num_devices = jax.local_device_count()
    # print(f"Running on {num_devices} devices.")

    if numsamples % num_devices != 0:
        raise ValueError(f"numsamples ({numsamples}) / GPU count ({num_devices})")
    adj_batch_np, graph_storage, _ = create_adj_batch(
        key, n, beta, p, eirat, numsamples
    )

    clst = np.zeros((numsamples,))
    avpl = np.zeros((numsamples,))
    for i in range(numsamples):
        clst[i] = nx.average_clustering(graph_storage[i])
        try:
            avpl[i] = nx.average_shortest_path_length(graph_storage[i])
        except nx.NetworkXException:
            avpl[i] = np.nan

    adj_batch_jax = jnp.asarray(adj_batch_np)
    adj_sharded = shard_data(adj_batch_jax)

    results_list = []

    for indices in perturbation_idx_list:
        indices = jnp.asarray(indices)

        sharded_output = parallel_solver(
            adj_sharded, perturbation, indices, 10.0, tmax, dt
        )

        full_batch_output = unshard_data(sharded_output)
        results_list.append(full_batch_output)

    final_x = jnp.stack(results_list, axis=1)
    final_x_np = np.asarray(final_x)
    return final_x_np, clst, avpl
