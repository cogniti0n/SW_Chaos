import jax
import jax.numpy as jnp
import diffrax as dfx
import numpy as np


def sw_rewire_random_edgeswap(
    rng: np.random.Generator,
    adj: np.ndarray,
    rewire_ratio: float,
    *,
    max_tries: int = 128,
    return_copies: bool = True,
    verbose: bool = True,
):
    if rewire_ratio < 0 or rewire_ratio > 1:
        raise ValueError("rewire_ratio entries must be in [0,1].")

    n = adj.shape[0]
    if adj.ndim != 2 or adj.shape[1] != n:
        raise ValueError(f"adj must be square (n,n). Got {adj.shape}.")

    adj_work = adj.T.copy()

    src0, _ = np.nonzero(adj_work != 0)
    m0 = int(src0.size)

    if m0 == 0:
        if verbose:
            print(f"# of edge swaps 0/0 (attempted 0) at ratio={rewire_ratio}")
        result = adj_work.T
        return result.copy() if return_copies else result

    succ_lists: list[list[int]] = []
    succ_pos: list[dict[int, int]] = []
    for u in range(n):
        nbrs = np.flatnonzero(adj_work[u] != 0).astype(int).tolist()
        succ_lists.append(nbrs)
        succ_pos.append({v: i for i, v in enumerate(nbrs)})

    active_sources = np.flatnonzero(
        np.fromiter((len(nbrs) > 0 for nbrs in succ_lists), dtype=bool)
    )
    if active_sources.size == 0:
        if verbose:
            print(f"# of edge swaps 0/{m0} (attempted 0) at ratio={rewire_ratio}")
        result = adj_work.T
        return result.copy() if return_copies else result

    def sample_source_with_out_edges() -> int:
        return int(active_sources[rng.integers(0, active_sources.size)])

    def sample_successor(u: int) -> int:
        nbrs = succ_lists[u]
        return int(nbrs[rng.integers(0, len(nbrs))])

    def replace_successor(u: int, old_v: int, new_v: int) -> bool:
        pos_u = succ_pos[u]
        idx = pos_u.pop(old_v, None)
        if idx is None:
            return False
        succ_lists[u][idx] = new_v
        pos_u[new_v] = idx
        return True

    def attempt_one_swap() -> bool:
        u = sample_source_with_out_edges()
        v = sample_successor(u)

        for _ in range(max_tries):
            x = sample_source_with_out_edges()
            y = sample_successor(x)

            if u == x and v == y:
                continue
            # Proposed swapped edges are u->y and x->v.
            if u == y or x == v:
                continue  # avoid self-loops
            if y in succ_pos[u]:
                continue  # avoid duplicates
            if v in succ_pos[x]:
                continue
            break
        else:
            return False

        w_uv = adj_work[u, v]
        w_xy = adj_work[x, y]
        if w_uv == 0 or w_xy == 0:
            return False

        adj_work[u, v] = 0
        adj_work[x, y] = 0
        adj_work[u, y] = w_uv
        adj_work[x, v] = w_xy

        if not replace_successor(u, v, y):
            return False
        if not replace_successor(x, y, v):
            return False

        return True

    target = int(np.floor(rewire_ratio * m0))

    if target <= 0:
        if verbose:
            print(f"# of edge swaps 0/{m0} (attempted 0) at ratio={rewire_ratio}")
        result = adj_work.T
        return result.copy() if return_copies else result

    succ = 0
    for _ in range(target):
        if attempt_one_swap():
            succ += 1

    if verbose:
        print(
            f"[t={target}] swaps {succ}/{m0} (attempted {target}) at ratio={rewire_ratio}"
        )

    result = adj_work.T
    return result.copy() if return_copies else result


def _dyn_multiple(t, y, args):
    adjs, t_cuts = args
    idx = jnp.searchsorted(t_cuts, t, side="right")
    _adj = adjs[idx]
    return -y + _adj @ jnp.tanh(y)


def rundynnp_rewire(
    key: jax.Array,
    adjs: jnp.ndarray,
    t_cuts: jnp.ndarray,
    perturbation: float,
    perturbation_idx: jnp.ndarray,
    tmax: float,
    dt: float,
) -> jnp.ndarray:

    assert adjs.shape[0] == t_cuts.shape[0] + 1
    n = adjs[0].shape[0]

    term = dfx.ODETerm(_dyn_multiple)
    solver = dfx.Dopri5()
    stepsize_controller = dfx.PIDController(rtol=1e-5, atol=1e-5)

    y0 = jax.random.normal(key, (n,)) * 0.5

    saveat = dfx.SaveAt(ts=jnp.linspace(0, tmax, int(tmax / dt)))
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0,
        t1=tmax,
        dt0=dt,
        y0=y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        args=(adjs, t_cuts),
    )

    return jnp.asarray(sol.ys)
