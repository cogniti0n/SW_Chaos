import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import diffrax as dfx

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def settle_term():
    def vf_on(t, x, args):
        adj, u, img = args  # adj: (n,n), u:(n,), img:(n,)
        return -x + adj @ jnp.tanh(x) + u * img

    def vf_off(t, x, args):
        adj = args
        return -x + adj @ jnp.tanh(x)

    return dfx.ODETerm(vf_on), dfx.ODETerm(vf_off)


def solve_settle_single(
    adj: jnp.ndarray,
    u: jnp.ndarray,
    img: jnp.ndarray,
    t_on: float,
    t_off: float,
    dt: float,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    snapshots: int = 0,
):
    """
    Returns x(t_settle) for one image.
    """
    n = adj.shape[0]
    y0 = jnp.zeros((n,), dtype=jnp.float32)

    term_on, term_off = settle_term()
    solver = dfx.Dopri5()
    steps = dfx.PIDController(rtol=rtol, atol=atol)

    sol_on = dfx.diffeqsolve(
        term_on,
        solver,
        t0=0.0,
        t1=t_on,
        dt0=dt,
        y0=y0,
        saveat=dfx.SaveAt(t1=True),  # only final state
        stepsize_controller=steps,
        args=(adj, u, img),
    )
    x_on = jnp.asarray(sol_on.ys)[0]
    if snapshots and snapshots > 0:
        ts_relax = jnp.linspace(0.0, t_off, snapshots)
        saveat = dfx.SaveAt(ts=ts_relax)
    else:
        saveat = dfx.SaveAt(t1=True)
    sol_off = dfx.diffeqsolve(
        term_off,
        solver,
        t0=0.0,
        t1=t_off,
        dt0=dt,
        y0=x_on,
        saveat=saveat,
        stepsize_controller=steps,
        args=adj,
    )
    x_off = jnp.asarray(sol_off.ys)
    if snapshots and snapshots > 0:
        return x_off
    # SaveAt(t1=True) returns shape (1, n); squeeze to (n,)
    return x_off[0]


solve_settle_batch_final = jax.jit(
    jax.vmap(
        lambda adj, u, imgs, t_on, t_off, dt: solve_settle_single(
            adj, u, imgs, t_on, t_off, dt, snapshots=0
        ),
        in_axes=(None, None, 0, None, None, None),
    ),
    static_argnums=(3, 4, 5),
)

solve_settle_batch_snapshots = jax.jit(
    jax.vmap(
        lambda adj, u, imgs, t_on, t_off, dt, snaps: solve_settle_single(
            adj, u, imgs, t_on, t_off, dt, snapshots=snaps
        ),
        in_axes=(None, None, 0, None, None, None, None),
    ),
    static_argnums=(3, 4, 5, 6),
)


class PerImageNormalize:
    def __call__(self, x):
        return (x - x.mean()) / (x.std().clamp_min(1e-6))


def make_mnist_loaders(n_side: int, batch_size: int = 128, num_workers: int = 2):
    """
    Resizes MNIST to (n_side, n_side) and flattens to length N=n_side^2.
    """
    tfm = transforms.Compose(
        [
            transforms.Resize((n_side, n_side)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def extract_features(loader, adj, u, t_on, t_off, dt, snaps=0, max_batches=None):
    """
    Runs the reservoir settle dynamics and returns:
      X: (num_samples, n) features
      y: (num_samples,) labels
    """
    feats = []
    labels = []

    for bi, (imgs, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        # imgs: (B,1,H,W) torch -> numpy -> jax
        img_np = imgs.numpy().astype(np.float32)
        B = img_np.shape[0]
        img_flat = img_np.reshape(B, -1)  # (B,N)

        img_jax = jnp.asarray(img_flat)

        if snaps == 0:
            xT = np.asarray(solve_settle_batch_final(adj, u, img_jax, t_on, t_off, dt))
        else:
            xSnaps = np.asarray(
                solve_settle_batch_snapshots(adj, u, img_jax, t_on, t_off, dt, snaps)
            )
            xT = xSnaps.reshape(xSnaps.shape[0], -1)
        feats.append(xT)
        labels.append(y.numpy())

    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(labels, axis=0)
    return X, Y


def train_ridge(
    Xtr, ytr, Xte, yte, l2=1e-4, bias=True, standardize=True, device="cuda"
):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    if standardize:
        mu = Xtr_t.mean(dim=0, keepdim=True)
        sig = Xtr_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xtr_t = (Xtr_t - mu) / sig
        Xte_t = (Xte_t - mu) / sig

    n_features = Xtr_t.shape[1]
    n_classes = int(ytr_t.max().item()) + 1

    if bias:
        ones_tr = torch.ones((Xtr_t.shape[0], 1), device=device, dtype=Xtr_t.dtype)
        ones_te = torch.ones((Xte_t.shape[0], 1), device=device, dtype=Xte_t.dtype)
        Xtr_t = torch.cat([Xtr_t, ones_tr], dim=1)
        Xte_t = torch.cat([Xte_t, ones_te], dim=1)
        n_features = n_features + 1

    Y = F.one_hot(ytr_t, num_classes=n_classes).to(Xtr_t.dtype)

    N = Xtr_t.shape[0]
    XtX = Xtr_t.T @ Xtr_t / N
    XtY = Xtr_t.T @ Y / N

    I = torch.eye(n_features, device=device, dtype=Xtr_t.dtype)

    if bias:
        I[-1, -1] = 0.0

    A = XtX + l2 * I
    W = torch.linalg.solve(A, XtY)

    logits = Xte_t @ W
    pred = logits.argmax(dim=1)
    acc = (pred == yte_t).float().mean().item()

    return acc
