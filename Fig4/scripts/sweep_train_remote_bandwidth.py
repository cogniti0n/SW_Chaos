import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from utils.swneural import sw_adj, sw_nogen

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = FIG_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import Fig4.scripts.train_remote_bandwidth as train_mod


def sweep_beta_remote_bandwidth(
    numpoints: int,
    *,
    n: int = 300,
    p: float = 0.07,
    seed: int = 0,
    eirat: float = 0.2,
    k_in: int = 1,
    k_out: int = 30,
    in_start: int = 0,
    out_start: int = -1,
    amp: float = 0.5,
    T_total: float = 2000.0,
    T_sym: float = 5.0,
    deadline_frac: float = 1.0,
    washout_syms: int = 50,
    train_frac: float = 0.7,
    dt: float = 0.1,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    alpha: float = 1e-4,
    no_bias: bool = False,
    no_standardize: bool = False,
    use_raw_features: bool = False,
    logmin: float = -4.0,
    logmax: float = 0.0,
):
    logbeta = np.linspace(logmin, logmax, numpoints)
    betas = 10**logbeta

    rng_data = np.random.default_rng(seed)

    in_center = int(in_start) % n
    in_idx = train_mod.centered_block_indices(n, in_center, k_in)
    if out_start < 0:
        out_center_eff = (in_center + n // 2) % n
    else:
        out_center_eff = int(out_start) % n
    out_idx = train_mod.centered_block_indices(n, out_center_eff, k_out)

    B_in = np.zeros((n, k_in), dtype=np.float32)
    for j, node in enumerate(in_idx):
        B_in[node, j] = amp
    B_in = jnp.asarray(B_in)

    ts_base, s_ts, s_sym = train_mod.make_symbol_stream(
        T_total=T_total,
        dt=dt,
        T_sym=T_sym,
        rng=rng_data,
        sigma_symbol_noise=0.0,
    )
    K = s_sym.shape[0]

    u_ts = np.repeat(s_ts[:, None], k_in, axis=1)
    u_ts = jnp.asarray(u_ts, dtype=jnp.float32)

    f = float(deadline_frac)
    f = min(max(f, 0.05), 1.0)

    k_ids = np.arange(K, dtype=np.int32)
    save_ts = (k_ids + f) * T_sym - 1e-9
    save_ts = save_ts[save_ts <= T_total - 1e-9].astype(np.float32)
    K_save = save_ts.shape[0]

    y = s_sym[:K_save].astype(np.float32)

    W = int(washout_syms)
    if W >= K_save - 10:
        raise ValueError("washout_syms too large for stream length.")
    split = int(W + (K_save - W) * train_frac)

    rmse_tr_store = np.full((numpoints,), np.nan, dtype=np.float32)
    rmse_te_store = np.full((numpoints,), np.nan, dtype=np.float32)
    baseline_store = np.full((numpoints,), np.nan, dtype=np.float32)
    norm_rmse_store = np.full((numpoints,), np.nan, dtype=np.float32)

    start_time = time.time()
    print(f"Starting sweep over {numpoints} beta values...")
    progress = tqdm(
        betas,
        total=numpoints,
        desc="sweeping beta",
        position=0,
        leave=True,
        dynamic_ncols=True,
    )
    for i, beta in enumerate(progress):
        try:
            rng_graph = np.random.default_rng(seed)
            adj_bool = sw_adj(rng_graph, n, p, beta)
            adj_signed = sw_nogen(rng_graph, adj_bool, eirat)
            adj = jnp.asarray(adj_signed, dtype=jnp.float32)

            x_save = train_mod.simulate_reservoir_saveat_zoh(
                key=jax.random.key(seed),
                adj=adj,
                B_in=B_in,
                s_sym=s_sym,
                T_sym=T_sym,
                save_ts=save_ts,
                dt0=dt,
                rtol=rtol,
                atol=atol,
            )

            X = x_save[:, jnp.asarray(out_idx)]
            if not use_raw_features:
                X = jnp.tanh(X)

            Xtr, ytr = X[W:split], jnp.asarray(y[W:split])
            Xte, yte = X[split:], jnp.asarray(y[split:])

            w, mu, sig = train_mod.fit_ridge(
                Xtr,
                ytr,
                alpha=alpha,
                bias=not no_bias,
                standardize=not no_standardize,
            )
            yhat_tr = train_mod.predict_ridge(
                Xtr,
                w,
                mu,
                sig,
                bias=not no_bias,
                standardize=not no_standardize,
            )
            yhat_te = train_mod.predict_ridge(
                Xte,
                w,
                mu,
                sig,
                bias=not no_bias,
                standardize=not no_standardize,
            )

            rmse_tr = train_mod.rmse(ytr, yhat_tr)
            rmse_te = train_mod.rmse(yte, yhat_te)
            baseline = float(
                np.sqrt(np.mean((np.asarray(yte) - np.mean(np.asarray(ytr))) ** 2))
            )
            norm_rmse = float(rmse_te / baseline) if baseline > 0 else np.nan

            rmse_tr_store[i] = rmse_tr
            rmse_te_store[i] = rmse_te
            baseline_store[i] = baseline
            norm_rmse_store[i] = norm_rmse

            progress.set_postfix(
                beta=f"{beta:.5g}",
                rmse_te=f"{rmse_te:.4f}",
                nrmse=f"{norm_rmse:.4f}",
            )
            tqdm.write(
                f"beta={beta:.5g} "
                f"rmse_train={rmse_tr:.4f} "
                f"rmse_test={rmse_te:.4f} "
                f"baseline={baseline:.4f} "
                f"norm_rmse_test={norm_rmse:.4f}"
            )
        except Exception as exep:
            tqdm.write(f"Error processing beta={beta}: {exep}")

    end_time = time.time()
    print(f"Sweep completed in {end_time - start_time:.2f} seconds.")
    print(f"Final metric vectors (shape {rmse_te_store.shape}):")
    print("rmse_train:", rmse_tr_store)
    print("rmse_test:", rmse_te_store)
    print("baseline:", baseline_store)
    print("norm_rmse_test:", norm_rmse_store)

    out_path = OUTPUT_DIR / (f"remote_bandwidth_sweep_seed{seed}.npz")
    np.savez_compressed(
        out_path,
        beta=betas,
        rmse_train=rmse_tr_store,
        rmse_test=rmse_te_store,
        baseline=baseline_store,
        norm_rmse_test=norm_rmse_store,
        seed=seed,
        logmin=logmin,
        logmax=logmax,
        n=n,
        p=p,
        eirat=eirat,
        k_in=k_in,
        k_out=k_out,
        in_start=in_start,
        out_start=out_center_eff,
        in_center=in_center,
        out_center=out_center_eff,
        amp=amp,
        T_total=T_total,
        T_sym=T_sym,
        deadline_frac=deadline_frac,
        washout_syms=washout_syms,
        train_frac=train_frac,
        dt=dt,
        rtol=rtol,
        atol=atol,
        alpha=alpha,
        no_bias=no_bias,
        no_standardize=no_standardize,
        use_raw_features=use_raw_features,
    )

    return betas, rmse_tr_store, rmse_te_store, baseline_store, norm_rmse_store


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--n", type=int, default=300)
    argparser.add_argument("--p", type=float, default=0.07)
    argparser.add_argument("--eirat", type=float, default=0.2)
    argparser.add_argument("--seed", type=int, default=0)

    argparser.add_argument("--k-in", type=int, default=1)
    argparser.add_argument("--k-out", type=int, default=30)
    argparser.add_argument(
        "--in-start",
        type=int,
        default=0,
        help="center index for the input block (legacy flag name)",
    )
    argparser.add_argument(
        "--out-start",
        type=int,
        default=-1,
        help="center index for the readout block; negative uses N/2 from input center",
    )

    argparser.add_argument("--amp", type=float, default=0.5)
    argparser.add_argument("--sigma-sensor", type=float, default=0.0)

    argparser.add_argument("--T-total", type=float, default=2000.0)
    argparser.add_argument("--T-sym", type=float, default=5.0)
    argparser.add_argument("--deadline-frac", type=float, default=1.0)

    argparser.add_argument("--washout-syms", type=int, default=50)
    argparser.add_argument("--train-frac", type=float, default=0.7)

    argparser.add_argument("--dt", type=float, default=0.1)
    argparser.add_argument("--rtol", type=float, default=1e-5)
    argparser.add_argument("--atol", type=float, default=1e-5)

    argparser.add_argument("--alpha", type=float, default=1e-4)
    argparser.add_argument("--no-bias", action="store_true")
    argparser.add_argument("--no-standardize", action="store_true")
    argparser.add_argument("--use-raw-features", action="store_true")

    argparser.add_argument("--numpoints", type=int, default=30)
    argparser.add_argument("--logmin", type=float, default=-4.0)
    argparser.add_argument("--logmax", type=float, default=0.0)

    args = argparser.parse_args()

    sweep_beta_remote_bandwidth(
        numpoints=args.numpoints,
        n=args.n,
        p=args.p,
        seed=args.seed,
        eirat=args.eirat,
        k_in=args.k_in,
        k_out=args.k_out,
        in_start=args.in_start,
        out_start=args.out_start,
        amp=args.amp,
        T_total=args.T_total,
        T_sym=args.T_sym,
        deadline_frac=args.deadline_frac,
        washout_syms=args.washout_syms,
        train_frac=args.train_frac,
        dt=args.dt,
        rtol=args.rtol,
        atol=args.atol,
        alpha=args.alpha,
        no_bias=args.no_bias,
        no_standardize=args.no_standardize,
        use_raw_features=args.use_raw_features,
        logmin=args.logmin,
        logmax=args.logmax,
    )
