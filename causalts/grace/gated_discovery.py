# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Gated Causal Discovery with Stability Selection.

Replaces softmax attention with independent Hard Concrete gates (Louizos et al. 2018)
and uses Stability Selection (Meinshausen & Bühlmann 2010) across L0 penalty strengths
to determine the causal graph without requiring a manual threshold.
"""

import inspect
import logging
import math
import numbers
import os
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from .result import GraceResult

EPS = 1e-6

logger = logging.getLogger(__name__)

_lightning_silenced = False
_parcorr_gpu_warned = False


def _get_parcorr_class():
    """Return ParCorrGPU class if CUDA available, else CPU PartialCorr with one-time warning."""
    global _parcorr_gpu_warned
    try:
        import torch

        if torch.cuda.is_available():
            from ..ci_tests.parcorr_gpu import ParCorrGPU

            return ParCorrGPU
    except ImportError:
        pass
    if not _parcorr_gpu_warned:
        logger.warning("ParCorrGPU requires CUDA; falling back to CPU PartialCorr")
        _parcorr_gpu_warned = True
    from ..ci_tests.parcorr import PartialCorr

    return PartialCorr


def _silence_lightning():
    """Suppress Lightning's repeated GPU/TPU/Tip messages. Called once."""
    global _lightning_silenced
    if _lightning_silenced:
        return
    _lightning_silenced = True
    for name in [
        "pytorch_lightning",  # noqa: E402
        "pytorch_lightning.utilities.rank_zero",  # noqa: E402
        "pytorch_lightning.accelerators.cuda",  # noqa: E402
        "pytorch_lightning.accelerators.mps",  # noqa: E402
        "lightning.pytorch",
        "lightning.fabric",
        "litlogger",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", ".*GPU available.*")
    warnings.filterwarnings("ignore", ".*TPU available.*")
    warnings.filterwarnings("ignore", ".*litlogger.*")
    warnings.filterwarnings("ignore", ".*validation_step.*")
    warnings.filterwarnings("ignore", ".*LeafSpec.*")
    warnings.filterwarnings("ignore", ".*num_workers.*")
    warnings.filterwarnings("ignore", ".*CUBLAS_WORKSPACE_CONFIG.*")
    # Set CUBLAS workspace config for deterministic mode on CUDA >= 10.2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _detect_accelerator() -> str:
    """Detect best available accelerator: mps > gpu > cpu."""
    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


#: Constant of the ``"dT"`` lambda rule, ``LAMBDA_DT_CONSTANT / (d * T)``.
LAMBDA_DT_CONSTANT = 300.0
#: d and T range where the ``"dT"`` rule matched or beat ``"density"`` in calibration
#: (it was fitted on d 5-100, but lost to ``"density"`` at d=5); it warns outside it.
LAMBDA_DT_RANGE = {"d": (10, 100), "T": (250, 1000)}
#: Coefficients of the ``"xcorr"`` lambda rule (see :func:`compute_lambda`).
LAMBDA_XCORR = dict(c=53.0, a=1.12, b=0.58, e=0.16, ref=0.05, floor=0.3, cap=0.5)
#: d and T range of the ``"xcorr"`` calibration; it warns outside it.
LAMBDA_XCORR_RANGE = {"d": (5, 100), "T": (250, 10000)}
#: Default minimum effect of an edge (``effect_floor``): adjusted partial R^2 of
#: 1 - exp(-0.02), i.e. 0.01 nats of predictive gain per observation.
EFFECT_FLOOR = 1.0 - math.exp(-0.02)
#: ``shrink_effect_floor="adaptive"``: per target, drop lagged cross edges whose unique
#: effect (adjusted partial R^2) is below ``max(floor, relative * strongest)``.
SHRINK_EFFECT_FLOOR_ADAPTIVE = dict(floor=0.01, relative=0.2)


def background_xcorr(X, max_lag: int) -> float:
    """Mean ``|corr(x_j(t), x_i(t-k))|`` over ``i != j`` and ``k = 1..max_lag``.

    The background lagged dependence of the data, used by the ``"xcorr"`` lambda
    rule. Rows with a missing value in any lag are dropped.
    """
    X = np.asarray(X, dtype=float)
    T, d = X.shape
    L = max(int(max_lag), 1)
    if d < 2 or T <= L + 2:
        return 0.0
    Z = np.stack([X[L - k : T - k] for k in range(1, L + 1)], axis=2)  # (T-L, d, L)
    Y = X[L:]
    ok = ~(np.isnan(Y).any(axis=1) | np.isnan(Z).any(axis=(1, 2)))
    Y, Z = Y[ok], Z[ok]
    if len(Y) < 3:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        Ys = (Y - Y.mean(0)) / Y.std(0)
        Zs = (Z - Z.mean(0)) / Z.std(0)
        C = np.abs(np.einsum("tj,tik->ijk", Ys, Zs) / len(Y))  # [cause, effect, lag]
    off = C[~np.eye(d, dtype=bool)]
    return float(np.nanmean(off)) if np.isfinite(off).any() else 0.0


def compute_lambda(
    d: int,
    T: int,
    skeleton_density: float,
    rule: str = "density",
    xcorr: Optional[float] = None,
) -> float:
    """Auto-select L0 penalty lambda from problem dimensions and skeleton density.

    ``rule="density"`` (Eq. 11 from arXiv:2606.23880)::

        max(0.007 + 0.16*rho_S,  1/d + 4/T - 0.4*rho_S^0.8)

    The main term adapts to problem scale; the floor ensures sufficient
    regularization for dense skeletons (Lorenz-96, Erdos-Renyi topologies).
    At large d the floor binds, so lambda stops falling with d.

    ``rule="dT"``: ``300 / (d * T)``, with no density term. Motivation: the
    likelihood is averaged over the d modelled variables, so one edge's
    evidence enters the loss scaled by 1/d while its gate cost is not, and a
    false edge's in-sample gain shrinks like 1/T. The constant was fitted on a
    separate calibration grid (GRACE with ``ar_path``, rank data, Student-t
    likelihood; linear, spline and MLP mechanisms; sparse and dense graphs;
    d 5-100, T 250-1000), where the optimal lambda fell with d and T and showed
    no significant dependence on the mechanism or the skeleton density. F1 falls
    steeply once lambda exceeds its optimum on dense graphs, so the constant
    sits just below the fitted one. Refining a loose supplied skeleton (lagged
    precision ~0.65) on sparse graphs, the optimum was ~1.3x larger; the number
    of candidates per target did not predict this. It warns outside
    ``LAMBDA_DT_RANGE``: below it lambda grows like 1/(d*T), much faster than
    under ``"density"``, and can close every gate.

    ``rule="xcorr"``: ``53 * (e / 0.05)^0.16 / (d^1.12 * T^0.58)``, clipped to
    ``[0.3 / d, 0.5]``, with ``e = max(xcorr - 0.8 / sqrt(T), 0.005)`` the mean
    lagged cross-correlation of the data (:func:`background_xcorr`) above its
    level under independence. Background dependence (strong autoregression,
    persistent latent drivers) calls for more pruning, and the floor keeps lambda
    from vanishing on long series, where ``"dT"`` tends to 0. Fitted on GRACE
    calibration runs over d 5-100 and T 250-10000 (the 1/d trend of the
    optimal lambda held up to d=200 in separate runs), with and without latent
    drivers and interactions; on the d 10-100, T 250-1000 grid it is within
    0.005 F1 of ``"dT"``. It warns outside ``LAMBDA_XCORR_RANGE``.

    Parameters
    ----------
    d : int
        Number of variables.
    T : int
        Number of time steps.
    skeleton_density : float
        Fraction of non-zero edges in the skeleton (``"density"`` only).
    rule : {"density", "dT", "xcorr"}, default "density"
    xcorr : float, optional
        Background lagged cross-correlation, required by ``"xcorr"``.

    Returns
    -------
    float
        Selected lambda value.
    """
    if rule == "dT":
        (d_lo, d_hi), (t_lo, t_hi) = LAMBDA_DT_RANGE["d"], LAMBDA_DT_RANGE["T"]
        if not (d_lo <= d <= d_hi and t_lo <= T <= t_hi):
            warnings.warn(
                f"lambda_rule='dT' was calibrated for d in [{d_lo}, {d_hi}] and T in "
                f"[{t_lo}, {t_hi}]; got d={d}, T={T}. Outside that range "
                f"{LAMBDA_DT_CONSTANT:g}/(d*T) is "
                "an extrapolation (below it, lambda can close every gate).",
                UserWarning,
                stacklevel=2,
            )
        return LAMBDA_DT_CONSTANT / (d * T)
    if rule == "xcorr":
        if xcorr is None:
            raise ValueError("rule='xcorr' needs xcorr (see background_xcorr)")
        if not np.isfinite(xcorr):
            raise ValueError(f"xcorr must be finite, got {xcorr!r}")
        (d_lo, d_hi), (t_lo, t_hi) = LAMBDA_XCORR_RANGE["d"], LAMBDA_XCORR_RANGE["T"]
        if not (d_lo <= d <= d_hi and t_lo <= T <= t_hi):
            warnings.warn(
                f"lambda_rule='xcorr' was calibrated for d in [{d_lo}, {d_hi}] and T "
                f"in [{t_lo}, {t_hi}]; got d={d}, T={T}. Outside that range lambda is "
                "an extrapolation, bounded to [0.3/d, 0.5].",
                UserWarning,
                stacklevel=2,
            )
        p = LAMBDA_XCORR
        e = max(xcorr - 0.8 / np.sqrt(T), 0.005)
        lam = p["c"] * (e / p["ref"]) ** p["e"] / (d ** p["a"] * T ** p["b"])
        return float(min(max(lam, p["floor"] / d), p["cap"]))
    if rule != "density":
        raise ValueError(f"rule must be 'density', 'dT' or 'xcorr', got {rule!r}")
    rho = skeleton_density
    floor = 0.007 + 0.16 * rho
    main = 1.0 / d + 4.0 / T - 0.4 * rho**0.8
    return max(floor, main)


def li_threshold(
    values: np.ndarray, floor: float = 0.05, min_per_side: int = 3
) -> Optional[float]:
    """Li minimum cross-entropy threshold on gate values.

    Iterates ``t = (m_hi - m_lo) / (ln m_hi - ln m_lo)`` from the mean, where
    ``m_lo`` / ``m_hi`` are the means below / above ``t`` (Li & Tam 1998).
    Values at or below ``floor`` are dropped first, so the spike of gates
    clamped to zero does not drag the cutoff down.

    Parameters
    ----------
    values : numpy.ndarray
        Gate values of the candidate edges (any shape; flattened).
    floor : float
        Values ``<= floor`` are excluded before thresholding.
    min_per_side : int
        Minimum number of values required on each side of the cutoff.

    Returns
    -------
    float or None
        The cutoff (an edge is kept if its gate is strictly above it), or
        None if fewer than ``min_per_side`` values fall on either side.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[v > floor]
    if len(v) < 2 * min_per_side:
        return None
    # Li needs positive values: shift so the minimum sits just above zero.
    shift = v.min() - 1e-3
    w = v - shift
    t = w.mean()
    for _ in range(200):
        lo, hi = w[w <= t], w[w > t]
        if len(lo) == 0 or len(hi) == 0:
            break
        m_lo, m_hi = lo.mean(), hi.mean()
        t_new = (m_hi - m_lo) / (np.log(m_hi) - np.log(m_lo))
        if abs(t_new - t) < 1e-6:
            break
        t = t_new
    cut = float(t + shift)
    n_hi = int((v > cut).sum())
    if n_hi < min_per_side or len(v) - n_hi < min_per_side:
        return None
    return cut


def gates_bimodal(
    values: np.ndarray,
    floor: float = 0.05,
    min_bic_gain: float = 10.0,
    stretch_gamma: float = -0.1,
    stretch_zeta: float = 1.1,
) -> bool:
    """Whether the candidate gate values form two distinct clusters.

    Fits one- and two-component Gaussian mixtures to the gates above ``floor``
    on the logit scale of the Hard Concrete stretch, and returns True when the
    two-component BIC is lower by more than ``min_bic_gain`` (10 = "very
    strong" evidence, Kass & Raftery 1995). Li's cutoff assumes two clusters;
    on a precise skeleton the true edges form one cluster and Li would split it.

    Parameters
    ----------
    values : numpy.ndarray
        Gate values of the candidate edges (any shape; flattened).
    floor : float
        Values ``<= floor`` are excluded.
    min_bic_gain : float
        Required BIC improvement of two components over one.
    stretch_gamma, stretch_zeta : float
        Hard Concrete stretch limits used for the logit transform.

    Returns
    -------
    bool
        False when fewer than 6 values remain.
    """
    from sklearn.mixture import GaussianMixture

    v = np.asarray(values, dtype=float).ravel()
    v = v[v > floor]
    if len(v) < 6:
        return False
    s = np.clip((v - stretch_gamma) / (stretch_zeta - stretch_gamma), 1e-4, 1 - 1e-4)
    u = np.log(s / (1 - s)).reshape(-1, 1)
    bic1 = GaussianMixture(1, random_state=0).fit(u).bic(u)
    bic2 = GaussianMixture(2, n_init=3, random_state=0).fit(u).bic(u)
    return bool(bic1 - bic2 > min_bic_gain)


def compute_lambda_cv(
    dataset: TensorDataset,
    num_vars: int,
    max_lag: int,
    skeleton: np.ndarray,
    model_kwargs: Optional[dict] = None,
    n_lambdas: int = 8,
    val_frac: float = 0.2,
    max_epochs: int = 100,
    warm_start_epochs: int = 75,
    patience: int = 15,
    batch_size: Optional[int] = None,
    model_seed: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> float:
    """Select L0 penalty lambda via cross-validation on held-out prediction loss.

    Splits data temporally (last val_frac as validation), trains at each lambda
    in a grid, and returns the lambda that minimizes validation NLL.  Models are
    warm-started from the previous lambda (ascending order) for efficiency.
    """
    if model_kwargs is None:
        model_kwargs = {}

    N = len(dataset)
    n_val = max(int(N * val_frac), 1)
    n_train = N - n_val
    train_data = TensorDataset(dataset.tensors[0][:n_train])
    val_data = TensorDataset(dataset.tensors[0][n_train:])

    if batch_size is None:
        batch_size = max(32, min(n_train // 8, 256))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    steps_per_epoch = math.ceil(n_train / batch_size)

    lambda_grid = np.logspace(-2.5, -0.3, n_lambdas)

    accelerator = device if device and device != "auto" else _detect_accelerator()

    best_lambda = float(lambda_grid[len(lambda_grid) // 2])
    best_val_loss = float("inf")
    prev_model = None

    if verbose:
        print(
            f"  CV lambda selection: {n_lambdas} lambdas, "
            f"train={n_train} val={n_val}"
        )

    for lam in sorted(lambda_grid):
        if model_seed is not None:  # before the model, so its random init is seeded
            torch.manual_seed(model_seed)
        model = GatedCausalDiscovery(
            num_vars=num_vars,
            max_lag=max_lag,
            lambda_l0=float(lam),
            normalize_l0=True,
            steps_per_epoch=steps_per_epoch,
            **model_kwargs,
        )
        model.init_from_skeleton(skeleton)
        model.set_skeleton_mask(skeleton)

        if prev_model is not None:
            with torch.no_grad():
                if hasattr(model, "encoder"):
                    model.encoder.load_state_dict(prev_model.encoder.state_dict())
                else:
                    model.W.copy_(prev_model.W)
                    model.bias.copy_(prev_model.bias)
                model.decoders.load_state_dict(prev_model.decoders.state_dict())
                model.log_alpha.copy_(prev_model.log_alpha)
                if model.ar_path:  # the decoders were fitted next to a trained AR head
                    for name in ("ar_W1", "ar_b1", "ar_W2", "ar_lin", "ar_b2"):
                        getattr(model, name).copy_(getattr(prev_model, name))
            run_epochs = warm_start_epochs
        else:
            run_epochs = max_epochs

        if model_seed is not None:
            from pytorch_lightning import seed_everything  # noqa: E402

            seed_everything(model_seed, workers=True)

        _silence_lightning()
        trainer = Trainer(
            max_epochs=run_epochs,
            accelerator=accelerator,
            deterministic="warn" if model_seed is not None else False,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=patience, mode="min")
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        n_edges = int((model.get_gate_values() > 0.5).sum())

        if verbose:
            print(
                f"    lambda={lam:.4f}: val_loss={val_loss:.4f}, edges={n_edges}"
                + (" *" if val_loss < best_val_loss else "")
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lambda = float(lam)

        prev_model = model

    if verbose:
        print(f"  CV selected lambda={best_lambda:.4f} (val_loss={best_val_loss:.4f})")

    return best_lambda


def build_smart_lambda_grid(n_lambdas: int = 10) -> np.ndarray:
    """Build a transition-concentrated lambda grid for stability selection.

    Places ~60% of points in the transition region [10^-2, 10^-0.5] where
    edges actually open/close, with sparser coverage of the extremes.
    This gives better resolution where it matters most.

    Parameters
    ----------
    n_lambdas : int
        Total number of lambda values (default 10).

    Returns
    -------
    numpy.ndarray
        Sorted array of lambda values.
    """
    if n_lambdas <= 3:
        return np.logspace(-3, 0, n_lambdas)

    low, high = -3.0, 0.0

    # Concentrate ~60% of points in the transition region [-2, -0.5]
    n_transition = max(2, int(n_lambdas * 0.6))
    n_explore = n_lambdas - n_transition

    # Exploration points: sparse coverage of full range (exclude endpoints,
    # they're added explicitly)
    explore_pts = np.linspace(low, high, n_explore + 2)[1:-1]

    # Transition points: dense in [-2, -0.5]
    transition_pts = np.linspace(-2.0, -0.5, n_transition)

    all_pts = np.unique(np.concatenate([[low], explore_pts, transition_pts, [high]]))
    # Take up to n_lambdas (unique may give us more or fewer)
    lambdas = np.sort(10.0**all_pts)
    if len(lambdas) > n_lambdas:
        # Subsample uniformly to keep n_lambdas
        idx = np.round(np.linspace(0, len(lambdas) - 1, n_lambdas)).astype(int)
        lambdas = lambdas[idx]
    return lambdas


def prepare_data(
    df: pd.DataFrame,
    max_lag: int,
    normalize: bool = True,
    device: Optional[torch.device] = None,
) -> TensorDataset:
    """Create windowed TensorDataset from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with shape [T, num_vars].
    max_lag : int
        Maximum lag to include.
    normalize : bool
        Z-score per variable.
    device : torch.device, optional
        Torch device to place tensors on.

    Returns
    -------
    torch.utils.data.TensorDataset
        Dataset with tensors of shape [T - max_lag, num_vars, max_lag+1].
    """
    arr = df.values.astype(np.float32)
    if np.isnan(arr).any():
        raise ValueError(
            "NaN found in data passed to prepare_data(). "
            "Use impute='var_em' in run_cdnots_gated() to fill missing values "
            "before neural training."
        )
    if normalize:
        mu = arr.mean(0, keepdims=True)
        std = arr.std(0, keepdims=True) + 1e-8
        arr = (arr - mu) / std

    T, d = arr.shape
    tensor = torch.tensor(arr, dtype=torch.float32)
    # unfold: [T, d] -> [d, T] -> unfold(1, max_lag+1, 1) -> [d, T-max_lag, max_lag+1]
    windows = tensor.T.unfold(1, max_lag + 1, 1)  # [d, T-max_lag, max_lag+1]
    windows = windows.permute(1, 0, 2)  # [T-max_lag, d, max_lag+1]
    if device is not None:
        windows = windows.to(device)
    return TensorDataset(windows)


from ..utils.helpers import evaluate_graph  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Self-contained MLP decoder
# ---------------------------------------------------------------------------


class _MLPDecoder(nn.Module):
    """Simple MLP decoder producing (mu, sigma) for Gaussian output."""

    def __init__(self, in_features: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_features
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_size), nn.SiLU()])
            dim = hidden_size
        layers.append(nn.Linear(dim, 2))  # 2 outputs: mu, raw_sigma
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                # Kaiming normal (fan-in) for SiLU hidden layers
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)
        # Final layer: small Xavier for stable initial mu/sigma
        final = self.net[-1]
        nn.init.xavier_uniform_(final.weight, gain=0.1)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _NonlinearEncoder(nn.Module):
    """Shared MLP encoder conditioned on per-(cause, lag) embeddings.

    A single MLP shared across all (cause, lag) pairs, with a learned embedding
    per pair that lets the network learn different nonlinear transforms (linear,
    quadratic, sinusoidal) for different inputs while sharing weights.
    """

    def __init__(
        self,
        num_vars: int,
        max_lag_plus_1: int,
        hidden_size: int = 64,
        encoder_hidden: int = 32,
        embed_dim: int = 8,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_vars * max_lag_plus_1, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + embed_dim, encoder_hidden),
            nn.SiLU(),
            nn.Linear(encoder_hidden, hidden_size),
        )
        self.num_vars = num_vars
        self.Lp1 = max_lag_plus_1
        # Kaiming for hidden layer (SiLU), Xavier for output projection
        nn.init.kaiming_normal_(self.net[0].weight, nonlinearity="linear")
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, d, Lp1 = x.shape
        idx = torch.arange(d * Lp1, device=x.device)
        emb = self.embed(idx).view(d, Lp1, -1)  # [cause, lag, embed_dim]
        x_in = x.unsqueeze(-1)  # [B, cause, lag, 1]
        emb_b = emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, cause, lag, embed_dim]
        inp = torch.cat([x_in, emb_b], dim=-1)  # [B, cause, lag, 1+embed_dim]
        return self.net(inp)  # [B, cause, lag, hidden_size]


# ---------------------------------------------------------------------------
# GatedCausalDiscovery — LightningModule
# ---------------------------------------------------------------------------


class GatedCausalDiscovery(LightningModule):
    """Causal discovery with Hard Concrete gated inputs and L0 regularization.

    Architecture (per effect variable e)::

        gates[c,l,e] = HardConcrete(log_alpha[c,l,e])
        x_hat[e] = sum_{c,l} gates[c,l,e] * (W[c,l] * x[c,l] + bias[c,l])
        (mu, sigma) = decoder(x_hat[e])
        Loss = NLL + lambda_l0 * sum P(gate != 0)
               + lambda_lag_group * sum max(0, lags_per_pair - max_lags_per_pair)
    """

    def __init__(
        self,
        num_vars: int,
        max_lag: int,
        hidden_size: int = 64,
        decoder_hidden: int = 64,
        decoder_layers: int = 2,
        lambda_l0: float = 0.1,
        lambda_lag_group: float = 0.0,
        max_lags_per_pair: Union[int, np.ndarray] = 1,
        temperature: float = 2 / 3,
        stretch_gamma: float = -0.1,
        stretch_zeta: float = 1.1,
        include_instantaneous: bool = False,
        lr: float = 1e-3,
        wd: float = 1e-4,
        normalize_l0: bool = True,
        steps_per_epoch: int = 1,
        nonlinear_encoder: bool = False,
        encoder_hidden: int = 32,
        embed_dim: int = 8,
        basis_expansion: bool = True,
        noise_model: str = "gaussian",
        ar_path: bool = False,
        ar_hidden: Optional[int] = None,
        ar_exog_cols: int = 0,
        **kwargs,
    ):
        super().__init__()
        # Unknown keyword arguments used to be accepted and ignored, so a misspelled
        # option (e.g. ``lambda_override`` for ``lambda_l0``) silently ran the default.
        if kwargs:
            raise TypeError(
                "GatedCausalDiscovery got unexpected keyword argument(s): "
                + ", ".join(sorted(kwargs))
            )
        # ar_path: an ungated per-target head on the target's own lags carries the
        # autoregression, and self-loops leave the gated part (see run_cdnots_gated).
        if ar_path and max_lag < 1:
            raise ValueError("ar_path needs max_lag >= 1 (self-loops are lagged)")
        if ar_hidden is not None and not ar_path:
            raise ValueError("ar_hidden only applies with ar_path=True")
        # ar_exog_cols: trailing columns (C in the model) that get no AR head; a
        # deterministic C (e.g. a linear trend) is fitted exactly by one, and its
        # vanishing noise scale would then dominate the loss.
        if (
            isinstance(ar_exog_cols, (bool, np.bool_))
            or not isinstance(ar_exog_cols, (int, np.integer))
            or not 0 <= ar_exog_cols < num_vars
        ):
            raise ValueError(
                f"ar_exog_cols must be an integer in [0, num_vars), got {ar_exog_cols!r}"
            )
        if ar_exog_cols and not ar_path:
            raise ValueError("ar_exog_cols only applies with ar_path=True")
        # Saved as passed (None when unset), so a checkpoint reloads.
        self.save_hyperparameters(ignore=["max_lags_per_pair"])
        ar_hidden = 16 if ar_hidden is None else ar_hidden
        _check_ar_hidden(ar_hidden)
        self.ar_path = bool(ar_path)
        if noise_model not in ("gaussian", "student_t"):
            raise ValueError(
                f"noise_model must be 'gaussian' or 'student_t', got {noise_model!r}"
            )
        self.noise_model = noise_model
        if noise_model == "student_t":
            # Per-variable degrees of freedom nu = 2 + softplus(raw), init nu ~ 10.
            self.nu_raw = nn.Parameter(
                torch.full((num_vars,), math.log(math.expm1(8.0)))
            )
        self.num_vars = num_vars
        self.max_lag = max_lag
        self.hidden_size = hidden_size
        self.lambda_l0 = lambda_l0
        self.lambda_lag_group = lambda_lag_group
        self.normalize_l0 = normalize_l0
        self._l0_scale = 1.0 / steps_per_epoch if normalize_l0 else 1.0
        self.nonlinear_encoder = nonlinear_encoder

        # max_lags_per_pair: scalar int or [cause, effect] array
        if isinstance(max_lags_per_pair, (int, float)):
            self._max_lags_scalar = float(max_lags_per_pair)
            self.register_buffer("_max_lags_tensor", None)
        else:
            self._max_lags_scalar = None
            self.register_buffer(
                "_max_lags_tensor",
                torch.tensor(max_lags_per_pair, dtype=torch.float32),
            )
        self.temperature = temperature
        self.stretch_gamma = stretch_gamma
        self.stretch_zeta = stretch_zeta
        self.include_instantaneous = include_instantaneous
        self.lr = lr
        self.wd = wd

        Lp1 = max_lag + 1  # number of lag slots (including lag 0)

        # Gate parameters — shape [cause, lag, effect] (internal: index -1 is lag 0)
        # Init at -0.5: gates start mostly closed (deterministic ~0.36),
        # edges must demonstrate predictive value to open past 0.5 threshold.
        self.log_alpha = nn.Parameter(torch.zeros(num_vars, Lp1, num_vars))
        nn.init.constant_(self.log_alpha, -0.5)

        # Encoder: nonlinear (shared MLP + embeddings) or linear (W * x + bias)
        # With basis_expansion: fixed [x, x², |x|] features before linear projection
        self.basis_expansion = basis_expansion and not nonlinear_encoder
        self.n_basis = 3 if self.basis_expansion else 1

        if nonlinear_encoder:
            self.encoder = _NonlinearEncoder(
                num_vars, Lp1, hidden_size, encoder_hidden, embed_dim
            )
        elif self.basis_expansion:
            # Kaiming-style init: std = sqrt(2 / fan_in) where fan_in = n_basis
            # (each hidden unit receives n_basis inputs from the basis expansion)
            fan_in = self.n_basis
            std = (2.0 / fan_in) ** 0.5
            self.W = nn.Parameter(
                torch.randn(num_vars, Lp1, self.n_basis * hidden_size) * std
            )
            self.bias = nn.Parameter(torch.zeros(num_vars, Lp1, hidden_size))
        else:
            # fan_in = 1 (single scalar input per cause-lag)
            self.W = nn.Parameter(torch.randn(num_vars, Lp1, hidden_size) * (2.0**0.5))
            self.bias = nn.Parameter(torch.zeros(num_vars, Lp1, hidden_size))

        # Decoder: independent MLP per effect variable
        self.decoders = nn.ModuleList(
            [
                _MLPDecoder(hidden_size, decoder_hidden, decoder_layers)
                for _ in range(num_vars)
            ]
        )

        if ar_path:
            L = max_lag
            self.ar_W1 = nn.Parameter(
                torch.randn(num_vars, 3 * L, int(ar_hidden)) * (2.0 / (3 * L)) ** 0.5
            )
            self.ar_b1 = nn.Parameter(torch.zeros(num_vars, int(ar_hidden)))
            self.ar_W2 = nn.Parameter(torch.zeros(num_vars, int(ar_hidden)))
            self.ar_lin = nn.Parameter(torch.zeros(num_vars, L))
            self.ar_b2 = nn.Parameter(torch.zeros(num_vars))
            targets = torch.ones(num_vars)
            targets[num_vars - int(ar_exog_cols) :] = 0.0
            # Derived from ar_exog_cols (saved in hparams), so not in the state dict.
            self.register_buffer("ar_targets", targets, persistent=False)

        # Build masks
        self._build_masks()

        # Optional skeleton mask (None = no mask)
        self._skeleton_mask: Optional[torch.Tensor] = None

        # Store gate values for get_estimated_graph
        self._gate_values: Optional[np.ndarray] = None

    def _build_masks(self):
        """Build mask for self-loops at lag 0 and optionally all instantaneous."""
        m, Lp1 = self.num_vars, self.max_lag + 1
        mask = torch.ones(m, Lp1, m, dtype=torch.bool)
        # Always mask self-loops at lag 0 (last lag index)
        for i in range(m):
            mask[i, -1, i] = False
        # Optionally mask all instantaneous edges
        if not self.include_instantaneous:
            mask[:, -1, :] = False
        if getattr(self, "ar_path", False):
            for i in range(m):
                mask[i, :, i] = False
        self.register_buffer("gate_mask", mask)

    # -- Hard Concrete sampling -------------------------------------------------

    def _sample_hard_concrete(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Sample from Hard Concrete distribution (training mode)."""
        u = torch.empty_like(log_alpha).uniform_(EPS, 1.0 - EPS)
        s = torch.sigmoid(
            (torch.log(u) - torch.log(1.0 - u) + log_alpha) / self.temperature
        )
        z_bar = s * (self.stretch_zeta - self.stretch_gamma) + self.stretch_gamma
        z = z_bar.clamp(0.0, 1.0)
        return z

    def _deterministic_gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Deterministic gate value (eval mode)."""
        s = torch.sigmoid(log_alpha)
        z_bar = s * (self.stretch_zeta - self.stretch_gamma) + self.stretch_gamma
        z = z_bar.clamp(0.0, 1.0)
        return z

    def _l0_penalty(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """Differentiable L0 penalty: P(z != 0) = sigmoid(log_alpha - t*log(-gamma/zeta))."""
        return torch.sigmoid(
            log_alpha
            - self.temperature * np.log(-self.stretch_gamma / self.stretch_zeta)
        )

    # -- Forward ----------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, num_vars, max_lag+1].

        Returns
        -------
        mu : torch.Tensor
            Predicted means with shape [batch, num_vars].
        sigma : torch.Tensor
            Predicted standard deviations with shape
            [batch, num_vars].
        """
        B, m, Lp1 = x.shape

        # Apply mask to log_alpha
        log_alpha = self.log_alpha.clone()
        log_alpha[~self.gate_mask] = -1e10  # force masked gates to zero

        # Apply optional skeleton mask
        if self._skeleton_mask is not None:
            skel_mask = self._skeleton_mask.to(log_alpha.device)
            log_alpha[~skel_mask] = -1e10

        # Compute gates
        if self.training:
            gates = self._sample_hard_concrete(log_alpha)  # [cause, lag, effect]
        else:
            gates = self._deterministic_gate(log_alpha)

        # Force masked positions to exactly zero
        gates = gates * self.gate_mask.float()
        if self._skeleton_mask is not None:
            gates = gates * self._skeleton_mask.float().to(gates.device)

        # Store for get_estimated_graph
        self._gate_values = gates.detach().cpu().numpy()

        # Encode: x -> [batch, cause, lag, hidden]
        if self.nonlinear_encoder:
            encoded = self.encoder(x)
        elif self.basis_expansion:
            # Fixed basis: [x, x², |x|] -> linear projection
            x_basis = torch.stack([x, x**2, x.abs()], dim=-1)  # [B, d, L+1, 3]
            # W: [d, L+1, 3*H] -> [d, L+1, 3, H]
            W_reshaped = self.W.view(m, Lp1, self.n_basis, self.hidden_size)
            # x_basis: [B, d, L+1, 3] -> [B, d, L+1, 3, 1]
            # W_reshaped: [d, L+1, 3, H] -> [1, d, L+1, 3, H]
            encoded = (x_basis.unsqueeze(-1) * W_reshaped.unsqueeze(0)).sum(
                dim=-2
            ) + self.bias.unsqueeze(0)
            # encoded: [B, d, L+1, H]
        else:
            encoded = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.bias.unsqueeze(0)

        # Apply gates and aggregate: sum over (cause, lag) per effect.
        # Done as a batched matmul over the flattened (cause, lag) axis instead
        # of a dense [B, cause, lag, effect, hidden] broadcast-then-sum: at
        # d=300 the dense form allocates ~17 GiB (B * d^2 * (L+1) * H) for a
        # tensor that is >99.5% masked out by the skeleton. matmul only ever
        # materializes O(B * d * (L+1) * H) intermediates.
        K = m * Lp1
        encoded_flat = encoded.reshape(B, K, self.hidden_size)  # [B, K, H]
        gates_flat = gates.reshape(K, m)  # [K, effect]
        x_hat = torch.matmul(encoded_flat.transpose(1, 2), gates_flat)  # [B, H, effect]
        x_hat = x_hat.transpose(1, 2)  # [B, effect, hidden]
        # x_hat: [B, effect, hidden]

        # Decode: per-effect independent decoders
        out_list = []
        for e in range(self.num_vars):
            out_list.append(self.decoders[e](x_hat[:, e, :]))  # [B, 2]
        out = torch.stack(out_list, dim=1)  # [B, num_vars, 2]
        mu = out[..., 0]  # [B, num_vars]
        sig = F.softplus(out[..., 1]) + EPS  # [B, num_vars]
        if self.ar_path:
            own = x[:, :, :-1]  # [B, d, L] own lags (lag 0 excluded)
            feats = torch.cat([own, own**2, own.abs()], dim=-1)  # [B, d, 3L]
            h = F.silu(torch.einsum("bdk,dkh->bdh", feats, self.ar_W1) + self.ar_b1)
            ar = (h * self.ar_W2).sum(-1) + (own * self.ar_lin).sum(-1) + self.ar_b2
            mu = mu + ar * self.ar_targets

        return mu, sig

    # -- Loss -------------------------------------------------------------------

    def get_loss(
        self, mu: torch.Tensor, sigma: torch.Tensor, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Gaussian NLL loss matching existing convention.

        Parameters
        ----------
        mu : torch.Tensor
            Predicted means with shape [batch, num_vars].
        sigma : torch.Tensor
            Predicted standard deviations with shape
            [batch, num_vars].
        obs : torch.Tensor
            Observations with shape [batch, num_vars, max_lag+1].
            Target is obs[..., -1].

        Returns
        -------
        nll : torch.Tensor
            Scalar tensor (includes L0 penalty during training).
        rmse : float
            Root mean squared error.
        """
        target = obs[..., -1]  # [B, num_vars] — current time step
        sq_err = (target - mu) ** 2
        log_sigma = torch.log(sigma)
        if self.noise_model == "student_t":
            # Student-t with scale sigma: a few extreme residuals cannot dominate
            # the fit the way they do under the Gaussian likelihood.
            nu = 2.0 + F.softplus(self.nu_raw)
            log_norm = (
                torch.lgamma((nu + 1) / 2)
                - torch.lgamma(nu / 2)
                - 0.5 * torch.log(nu * np.pi)
            )
            nll_terms = (
                log_sigma
                - log_norm
                + (nu + 1) / 2 * torch.log1p(sq_err / (nu * sigma**2))
            )
        else:
            nll_terms = sq_err / (2 * sigma**2) + log_sigma + 0.5 * np.log(2 * np.pi)
        nll = nll_terms.mean()

        rmse = sq_err.mean(dim=-1).sqrt().mean().item()

        if self.training:
            # L0 penalty
            log_alpha = self.log_alpha.clone()
            log_alpha[~self.gate_mask] = -1e10
            if self._skeleton_mask is not None:
                log_alpha[~self._skeleton_mask.to(log_alpha.device)] = -1e10
            gate_probs = self._l0_penalty(log_alpha)  # P(gate != 0)
            gate_probs = gate_probs * self.gate_mask.float()

            nll = nll + self.lambda_l0 * self._l0_scale * gate_probs.sum()

            # Lag concentration penalty: discourage multiple lags per (cause, effect) pair
            # gate_probs: [cause, lag, effect]
            # Sum P(gate!=0) across lags for each (cause, effect) pair, then penalize
            # the excess above max_lags_per_pair using a soft hinge: max(0, sum - k)
            # Scales with lambda_l0 (tracks sparsity pressure across stability
            # selection sweep) but NOT with _l0_scale (structural, not per-step).
            if self.lambda_lag_group > 0:
                lags_per_pair = gate_probs.sum(dim=1)  # [cause, effect]
                if self._max_lags_tensor is not None:
                    max_lags = self._max_lags_tensor
                else:
                    max_lags = self._max_lags_scalar
                excess = F.relu(lags_per_pair - max_lags)
                nll = nll + self.lambda_l0 * self.lambda_lag_group * excess.sum()

        return nll, rmse

    # -- Lightning hooks --------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Compute training loss and log metrics."""
        (x,) = batch
        mu, sig = self.forward(x)
        loss, rmse = self.get_loss(mu, sig, x)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_rmse", rmse, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation loss and log metrics."""
        (x,) = batch
        mu, sig = self.forward(x)
        loss, rmse = self.get_loss(mu, sig, x)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # noqa: D102
        """Return Adam optimizer with configured lr and weight decay."""
        # Weight decay is a prior on network weights, not on the Student-t
        # degrees of freedom: decaying nu_raw would pull nu toward heavy tails.
        nu = [p for n, p in self.named_parameters() if n == "nu_raw"]
        rest = [p for n, p in self.named_parameters() if n != "nu_raw"]
        groups = [{"params": rest}]
        if nu:
            groups.append({"params": nu, "weight_decay": 0.0})
        return torch.optim.Adam(groups, lr=self.lr, weight_decay=self.wd)

    # -- Graph extraction -------------------------------------------------------

    def get_gate_values(self) -> np.ndarray:
        """Return deterministic gate values in internal shape [cause, lag, effect].

        Lag index -1 is lag 0 (current time).
        """
        self.eval()
        with torch.no_grad():
            log_alpha = self.log_alpha.clone()
            log_alpha[~self.gate_mask] = -1e10
            if self._skeleton_mask is not None:
                log_alpha[~self._skeleton_mask.to(log_alpha.device)] = -1e10
            gates = self._deterministic_gate(log_alpha)
            gates = gates * self.gate_mask.float()
            if self._skeleton_mask is not None:
                gates = gates * self._skeleton_mask.float().to(gates.device)
        return gates.cpu().numpy()

    def get_estimated_graph(
        self, threshold: Union[float, str, None] = 0.5, absval: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Return (G_bin, W) in [cause, effect, lag] with lag 0..max_lag.

        Same output format as existing models in attention_discovery.py.

        Parameters
        ----------
        threshold : float, ``'li'`` or None
            If None, G_bin is None. ``'li'`` keeps gates strictly above
            :func:`li_threshold` of the gate values (0.5 if it falls back).
        absval : bool
            Threshold on absolute value of W if True.

        Returns
        -------
        G_bin : numpy.ndarray or None
            Binary graph with shape [cause, effect, lag] as int8, or None.
        W : numpy.ndarray
            Deterministic gate values with shape [cause, effect, lag].
        """
        arr = self.get_gate_values()  # [cause, lag(L..0), effect]
        # Reverse lag axis (L..0) -> (0..L) then permute to [cause, effect, lag]
        W = np.transpose(arr[:, ::-1, :], (0, 2, 1)).copy()

        if threshold is None:
            return None, W

        comp = np.abs(W) if absval else W
        if threshold == "li":
            cut = li_threshold(comp)
            G_bin = (comp > (0.5 if cut is None else cut)).astype(np.int8)
        else:
            G_bin = (comp >= threshold).astype(np.int8)
        return G_bin, W

    # -- Skeleton initialization ------------------------------------------------

    def init_from_skeleton(
        self, skeleton: np.ndarray, pos_val: float = 0.5, neg_val: float = -2.0
    ):
        """Initialize log_alpha from a CI skeleton.

        Parameters
        ----------
        skeleton : [cause, effect, lag] with lag 0..max_lag — binary
        pos_val : log_alpha value for edges in skeleton
        neg_val : log_alpha value for edges not in skeleton
        """
        # Convert from [cause, effect, lag(0..L)] to internal [cause, lag(L..0), effect]
        skel_internal = np.transpose(skeleton[:, :, ::-1], (0, 2, 1)).copy()
        skel_tensor = torch.tensor(skel_internal, dtype=torch.float32)
        with torch.no_grad():
            self.log_alpha.copy_(torch.where(skel_tensor > 0.5, pos_val, neg_val))

    def set_skeleton_mask(self, skeleton: np.ndarray):
        """Hard-mask non-skeleton edges — they cannot open during training.

        Parameters
        ----------
        skeleton : [cause, effect, lag] with lag 0..max_lag — binary
        """
        # Convert to internal [cause, lag(L..0), effect]
        skel_internal = np.transpose(skeleton[:, :, ::-1], (0, 2, 1)).copy()
        self._skeleton_mask = torch.tensor(skel_internal > 0.5, dtype=torch.bool)


# ---------------------------------------------------------------------------
# StabilitySelector
# ---------------------------------------------------------------------------


class StabilitySelector:
    """Stability Selection over a grid of L0 penalty strengths.

    Runs GatedCausalDiscovery for each lambda and records which gates are active.
    Stability score = fraction of runs where each gate is active.
    """

    def __init__(
        self,
        num_vars: int,
        max_lag: int,
        lambda_grid: Optional[np.ndarray] = None,
        n_subsamples: int = 1,
        subsample_frac: float = 0.8,
        skeleton: Optional[np.ndarray] = None,
        use_skeleton_mask: bool = False,
        model_kwargs: Optional[dict] = None,
        normalize_l0: bool = True,
        warm_start: bool = True,
        warm_start_epochs: Optional[int] = 75,
        seed: Optional[int] = None,
    ):
        self.num_vars = num_vars
        self.max_lag = max_lag
        self.lambda_grid = (
            lambda_grid if lambda_grid is not None else np.logspace(-3, 0, 20)
        )
        self.n_subsamples = n_subsamples
        self.subsample_frac = subsample_frac
        self.skeleton = skeleton
        self.use_skeleton_mask = use_skeleton_mask
        self.model_kwargs = model_kwargs or {}
        self.normalize_l0 = normalize_l0
        self.warm_start = warm_start
        self.warm_start_epochs = warm_start_epochs
        self.seed = seed

        # Extract max_lags_per_pair for hard enforcement (pruning)
        # Default to 1 (single lag per pair) — matches real-world assumption
        mlpp = self.model_kwargs.get("max_lags_per_pair", 1)
        self.max_lags_per_pair = int(mlpp) if isinstance(mlpp, (int, float)) else mlpp

        # Results storage
        self._selections: List[np.ndarray] = (
            []
        )  # list of binary arrays [cause, lag, effect]
        self._models: List[GatedCausalDiscovery] = []
        self._lambda_values: List[float] = []
        self._stability_scores: Optional[np.ndarray] = None

    def _prune_lags_per_pair(
        self, active: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Hard-enforce max_lags_per_pair by keeping top-K lags per (cause, effect).

        Parameters
        ----------
        active : numpy.ndarray
            Binary array with shape [cause, lag, effect] to prune
            in-place.
        values : numpy.ndarray
            Continuous values with shape [cause, lag, effect] used to
            rank lags.

        Returns
        -------
        numpy.ndarray
            The active array with excess lags zeroed out.
        """
        mlpp = self.max_lags_per_pair
        if mlpp is None:
            return active
        d, L, d2 = active.shape
        for i in range(d):
            for j in range(d2):
                lags_on = np.where(active[i, :, j] > 0)[0]
                if isinstance(mlpp, np.ndarray):
                    k = int(mlpp[i, j])
                else:
                    k = int(mlpp)
                if len(lags_on) > k:
                    # Keep top-K by value, zero rest
                    ranked = sorted(
                        lags_on,
                        key=lambda lag: values[i, lag, j],
                        reverse=True,
                    )
                    for lag_idx in ranked[k:]:
                        active[i, lag_idx, j] = 0
        return active

    def run(
        self,
        dataset: TensorDataset,
        max_epochs: int = 200,
        batch_size: int = 64,
        device: Optional[str] = None,
        patience: int = 20,
        verbose: bool = True,
        early_terminate_zero: int = 3,
    ):
        """Run full stability selection.

        Parameters
        ----------
        dataset : torch.utils.data.TensorDataset
            Dataset with shape [N, num_vars, max_lag+1].
        max_epochs : int
            Maximum training epochs per run.
        batch_size : int
            Training batch size.
        device : str or None
            Accelerator: 'cuda', 'cpu', 'mps', 'auto', or None for
            auto-detect.
        patience : int
            Early stopping patience.
        verbose : bool
            Print summary.
        early_terminate_zero : int
            Stop after this many consecutive lambdas with zero active
            edges (higher lambda only makes it sparser). Set 0 to
            disable.
        """
        if device is None or device == "auto":
            accelerator = _detect_accelerator()
        else:
            accelerator = device

        self._selections = []
        self._models = []
        self._lambda_values = []

        if self.seed is not None:
            from pytorch_lightning import seed_everything  # noqa: E402

            seed_everything(self.seed, workers=True)

        full_data = dataset.tensors[0]
        N = full_data.shape[0]
        n_sub = max(int(N * self.subsample_frac), self.max_lag + 2)

        # Sort lambda grid ascending (low → high penalty)
        lambda_sorted = np.sort(self.lambda_grid)

        total_runs = len(lambda_sorted) * self.n_subsamples
        if verbose:
            print(
                f"Stability Selection: {len(lambda_sorted)} lambdas x "
                f"{self.n_subsamples} subsamples = {total_runs} max runs"
            )
            print(
                f"  accelerator={accelerator}, max_epochs={max_epochs}, "
                f"patience={patience}"
            )

        consecutive_zero = 0
        runs_done = 0

        for li, lam in enumerate(lambda_sorted):
            # Early termination: if last N consecutive lambdas gave zero edges, stop
            if early_terminate_zero > 0 and consecutive_zero >= early_terminate_zero:
                skipped = (len(lambda_sorted) - li) * self.n_subsamples
                if verbose:
                    print(
                        f"  Early stop: {consecutive_zero} consecutive lambdas "
                        f"with 0 edges. Skipping {skipped} remaining runs."
                    )
                break

            all_zero_this_lambda = True

            for si in range(self.n_subsamples):
                # Subsample
                if self.n_subsamples > 1 or self.subsample_frac < 1.0:
                    idx = torch.randperm(N)[:n_sub]
                    sub_data = TensorDataset(full_data[idx])
                else:
                    sub_data = dataset

                loader = DataLoader(sub_data, batch_size=batch_size, shuffle=True)

                # Compute steps per epoch for L0 normalization
                steps = math.ceil(len(sub_data) / batch_size)

                # Create model
                model = GatedCausalDiscovery(
                    num_vars=self.num_vars,
                    max_lag=self.max_lag,
                    lambda_l0=float(lam),
                    normalize_l0=self.normalize_l0,
                    steps_per_epoch=steps,
                    **self.model_kwargs,
                )

                # Optionally initialize from skeleton
                if self.skeleton is not None:
                    model.init_from_skeleton(self.skeleton)
                    if self.use_skeleton_mask:
                        model.set_skeleton_mask(self.skeleton)

                # Warm-start: copy encoder + decoders + gates from previous model
                is_warm = False
                if self.warm_start and self._models:
                    prev = self._models[-1]
                    with torch.no_grad():
                        if hasattr(model, "encoder"):
                            model.encoder.load_state_dict(prev.encoder.state_dict())
                        else:
                            model.W.copy_(prev.W)
                            model.bias.copy_(prev.bias)
                        model.decoders.load_state_dict(prev.decoders.state_dict())
                        model.log_alpha.copy_(prev.log_alpha)
                    is_warm = True

                # Determine epochs (reduced for warm-started runs)
                run_epochs = max_epochs
                if is_warm and self.warm_start_epochs is not None:
                    run_epochs = self.warm_start_epochs

                # Train (suppress Lightning's per-model log spam)
                _silence_lightning()
                trainer = Trainer(
                    max_epochs=run_epochs,
                    accelerator=accelerator,
                    deterministic="warn" if self.seed is not None else False,
                    callbacks=[
                        EarlyStopping(
                            monitor="train_loss",
                            patience=patience,
                            mode="min",
                        )
                    ],
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    logger=False,
                    enable_checkpointing=False,
                )
                trainer.fit(model, train_dataloaders=loader)

                # Record active gates
                gates = model.get_gate_values()  # [cause, lag, effect]
                active = (gates > 0.5).astype(np.int8)
                active = self._prune_lags_per_pair(active, gates)
                self._selections.append(active)
                self._lambda_values.append(float(lam))

                n_active = int(active.sum())
                if n_active > 0:
                    all_zero_this_lambda = False

                # Store model on CPU
                model = model.cpu()
                self._models.append(model)

                runs_done += 1
                if verbose:
                    print(
                        f"  [{runs_done}/{total_runs}] λ={lam:.4e}  "
                        f"active_edges={n_active}"
                    )

            if all_zero_this_lambda:
                consecutive_zero += 1
            else:
                consecutive_zero = 0

        # Compute stability scores with informative-run scoring:
        # only count runs where ≥1 edge is active.
        # This prevents high-λ runs with zero edges from diluting true edge scores.
        selections_arr = np.stack(
            self._selections, axis=0
        )  # [runs, cause, lag, effect]
        informative_mask = selections_arr.sum(axis=(1, 2, 3)) > 0
        self._n_informative = int(informative_mask.sum())
        if self._n_informative > 0:
            self._stability_scores = selections_arr[informative_mask].mean(axis=0)
        else:
            self._stability_scores = np.zeros_like(selections_arr[0])

        if verbose:
            print(
                f"\nStability selection complete ({runs_done} runs, "
                f"{self._n_informative} informative)."
            )

    def get_stability_scores(self) -> np.ndarray:
        """Return stability scores in [cause, effect, lag] with lag 0..max_lag."""
        assert self._stability_scores is not None, "Must call run() first"
        arr = self._stability_scores  # [cause, lag(L..0), effect]
        return np.transpose(arr[:, ::-1, :], (0, 2, 1)).copy()

    def get_graph(self, threshold: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        """Return (G_bin, scores) in [cause, effect, lag] with lag 0..max_lag.

        Parameters
        ----------
        threshold : float
            Stability threshold (Meinshausen and Buhlmann recommend
            0.6-0.9).

        Returns
        -------
        G_bin : numpy.ndarray
            Binary graph with shape [cause, effect, lag] as int8.
        scores : numpy.ndarray
            Stability scores with shape [cause, effect, lag].
        """
        scores = self.get_stability_scores()
        G_bin = (scores >= threshold).astype(np.int8)

        # Post-hoc hard enforcement of max_lags_per_pair
        # scores/G_bin are [cause, effect, lag] — transpose to [cause, lag, effect]
        # for _prune_lags_per_pair, then transpose back
        if self.max_lags_per_pair is not None:
            G_t = np.transpose(G_bin, (0, 2, 1))  # [cause, lag, effect]
            s_t = np.transpose(scores, (0, 2, 1))  # [cause, lag, effect]
            G_t = self._prune_lags_per_pair(G_t, s_t)
            G_bin = np.transpose(G_t, (0, 2, 1)).copy()

        return G_bin, scores

    def expected_false_positives(self, threshold: float) -> float:
        """Upper bound on expected false positives (Meinshausen & Buhlmann 2010).

        Bound::

            E[FP] <= q^2 / ((2*pi_thr - 1) * |Lambda|)

        where ``q`` = average number of selected edges per informative run
        and ``Lambda`` = number of informative runs.

        Parameters
        ----------
        threshold : float
            Stability threshold used to compute the bound.

        Returns
        -------
        float
            Upper bound on expected number of false positives.
        """
        assert self._selections, "Must call run() first"
        # Use informative runs only (consistent with scoring)
        selections_arr = np.stack(self._selections, axis=0)
        informative_mask = selections_arr.sum(axis=(1, 2, 3)) > 0
        n_informative = int(informative_mask.sum())
        if n_informative == 0:
            return 0.0
        q = np.mean([s.sum() for s in np.array(self._selections)[informative_mask]])
        denom = (2 * threshold - 1) * n_informative
        if denom <= 0:
            return float("inf")
        return q**2 / denom

    def plot_stability_path(self, var_names: Optional[List[str]] = None):
        """Plot stability path: edge count vs lambda.

        Parameters
        ----------
        var_names : list of str or None
            Variable names for axis labels.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with edge count and stability score plots.
        """
        import matplotlib.pyplot as plt

        assert self._selections, "Must call run() first"

        # Group selections by lambda
        lam_to_counts: dict[float, list[int]] = {}
        for sel, lam in zip(self._selections, self._lambda_values):
            lam_to_counts.setdefault(lam, []).append(int(sel.sum()))

        lambdas_sorted = sorted(lam_to_counts.keys())
        mean_counts = [np.mean(lam_to_counts[l]) for l in lambdas_sorted]  # noqa: E741

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: edge count vs lambda
        ax = axes[0]
        ax.plot(lambdas_sorted, mean_counts, "o-", markersize=4)
        ax.set_xscale("log")
        ax.set_xlabel("L0 penalty (λ)")
        ax.set_ylabel("Number of active edges")
        ax.set_title("Edge Count vs L0 Penalty")
        ax.grid(True, alpha=0.3)

        # Right: stability scores heatmap
        ax = axes[1]
        scores = self.get_stability_scores()  # [cause, effect, lag]
        d, _, L = scores.shape
        # Flatten to [cause*lag, effect] for visualization
        scores_flat = scores.reshape(d, -1).T  # [effect*lag, cause]

        labels = var_names or [f"X{i}" for i in range(d)]
        lag_labels = []
        for e in range(d):
            for lag in range(L):
                lag_labels.append(f"{labels[e]}_L{lag}")

        # Only show edges with score > 0
        nonzero_mask = scores_flat.max(axis=1) > 0
        if nonzero_mask.any():
            scores_show = scores_flat[nonzero_mask]
            labels_show = [
                lag_labels[i] for i in range(len(lag_labels)) if nonzero_mask[i]
            ]
            im = ax.imshow(scores_show, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(d))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels_show)))
            ax.set_yticklabels(labels_show, fontsize=7)
            plt.colorbar(im, ax=ax, label="Stability Score")
        ax.set_title("Stability Scores")
        ax.set_xlabel("Cause Variable")

        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_ci_skeleton(
    df: pd.DataFrame,
    max_lag: int,
    alpha: float = 0.05,
    ci_test_class=None,
    normalize: bool = True,
    max_degree: int = 1,
    include_C: bool = True,
    c_preset: str = "linear",
    cache_dir: str | None = None,
    verbose: bool = True,
    return_pvalues: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Run CI-based skeleton discovery and return binary skeleton.

    Self-contained implementation using causalts directly (no qfr_ml
    dependency).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with shape [T, num_vars].
    max_lag : int
        Maximum lag to include.
    alpha : float
        Significance level for CI tests.
    ci_test_class : class or None
        CI test class. Defaults to PartialCorr.
    normalize : bool
        Z-score normalize data.
    max_degree : int
        Maximum conditioning set size.
    include_C : bool
        Include the C nonstationarity basis as extra variable(s).
    c_preset : str, default ``'linear'``
        C basis preset (same options as ``run_cdnots``). Multi-column presets
        such as ``'linear+sin'`` add one variable per basis column; all of them
        are trimmed from the returned skeleton.
    cache_dir : str
        Directory for caching CI test results.
    verbose : bool
        Print progress.
    return_pvalues : bool
        If True, also return p-value matrix [cause, effect, lag].

    Returns
    -------
    skeleton : numpy.ndarray
        Binary adjacency with shape [cause, effect, lag] as int8,
        with lag 0..max_lag.
    pval_matrix : numpy.ndarray
        P-values per edge with shape [cause, effect, lag]. 1.0 for
        untested/insignificant edges. Only returned if
        return_pvalues=True.
    """
    try:
        from ..cdnots.skeleton_discovery import (
            initialize_graph,
            skeleton_cnst,
            skeleton_discovery,
        )
    except ImportError as e:
        raise ImportError(
            "causalts package required for CI skeleton. "
            "Install with: pip install causalts\n"
            f"Original error: {e}"
        )

    # Normalize
    df_input = df.copy()
    if normalize:
        df_input = pd.DataFrame(
            (df.values - df.values.mean(0)) / (df.values.std(0) + 1e-8),
            columns=df.columns,
            index=df.index,
        )

    if ci_test_class is None:
        ci_test_class = _get_parcorr_class()

    import hashlib

    data_hash = hashlib.md5(str(df.shape).encode()).hexdigest()[:8]
    # Some CI tests (e.g. RCoT, KCI) need numpy arrays, not DataFrames
    ci_data = df_input.values if hasattr(df_input, "values") else df_input
    ci_data = np.ascontiguousarray(ci_data)
    ci_test = ci_test_class(ci_data, cache_dir=cache_dir, data_hash=data_hash)

    # Add the C nonstationarity basis if requested
    if include_C:
        from ..cdnots.phase3_utils import _C_PRESET_LABELS, make_c_array

        C_data = make_c_array(df_input.shape[0], c_preset)
        C = pd.DataFrame(
            C_data,
            columns=_C_PRESET_LABELS.get(
                c_preset, [f"C{i}" for i in range(C_data.shape[1])]
            ),
        )
        df_input = pd.concat([df_input.reset_index(drop=True), C], axis=1)

    # Create lagged data
    data = (
        pd.concat([df_input.shift(i) for i in range(max_lag + 1)], axis=1)
        .dropna()
        .values
    )
    ci_test.data = data

    # Run skeleton discovery
    cg_i = initialize_graph(data, None)
    cg_i_cnst = skeleton_cnst(cg_i, data, max_lag)
    cg_skel, pvals = skeleton_discovery(
        cg=cg_i_cnst,
        ci_test=ci_test,
        alpha=alpha,
        stable=True,
        verbose=False,  # suppress per-pair p-value spam
        show_progress=False,
        return_pvals=True,
        max_degree=max_degree,
        num_lags=max_lag,
    )

    # Remove duplicates from pvals
    pvals = list(dict.fromkeys(pvals))

    # Build M matrix from pvals (same logic as attention_discovery._build_M_matrix_from_pvals)
    num_vars_with_c = df_input.shape[1]
    M = np.full((num_vars_with_c, num_vars_with_c, max_lag + 1), -np.inf, dtype=float)

    def node_to_var_lag(idx):
        idx = int(idx)
        return idx % num_vars_with_c, idx // num_vars_with_c

    for eff_node, cause_node, cond_set, p in pvals:
        p = float(p)
        e_var, e_lag = node_to_var_lag(eff_node)
        c_var, c_lag = node_to_var_lag(cause_node)
        if e_lag != 0:
            continue
        if not (0 <= c_lag <= max_lag):
            continue
        if c_lag < e_lag:
            continue
        k = max_lag - int(c_lag)  # lag_to_axis_index
        if p > M[e_var, c_var, k]:
            M[e_var, c_var, k] = p

    # Set never-filled entries to 1.0 (not significant)
    M[np.isneginf(M)] = 1.0

    # Trim to original variables (exclude C column if added)
    num_vars_orig = df.shape[1]
    M = M[:num_vars_orig, :num_vars_orig, :]

    # M is [effect, cause, lag(L..0)] — significant edges: p-value <= alpha
    skel_internal = (M <= alpha).astype(np.int8)

    # Convert to [cause, effect, lag(0..L)]
    skeleton = np.transpose(skel_internal, (1, 0, 2))[:, :, ::-1].copy()

    n_edges = int(skeleton.sum())
    if verbose:
        print(f"CI skeleton: {n_edges} edges at alpha={alpha}")

    if return_pvalues:
        # Convert M from [effect, cause, lag(L..0)] to [cause, effect, lag(0..L)]
        pval_matrix = np.transpose(M, (1, 0, 2))[:, :, ::-1].copy()
        return skeleton, pval_matrix

    return skeleton


def run_stability_selection(
    df: pd.DataFrame,
    max_lag: int,
    lambda_grid: Optional[np.ndarray] = None,
    n_subsamples: int = 1,
    subsample_frac: float = 0.8,
    skeleton: Optional[np.ndarray] = None,
    use_ci_skeleton: bool = False,
    ci_alpha: float = 0.05,
    ci_test_class=None,
    include_C: bool = True,
    c_preset: str = "linear",
    use_skeleton_mask: bool = False,
    stability_threshold: float = 0.6,
    normalize: bool = True,
    normalize_l0: bool = True,
    max_epochs: int = 150,
    batch_size: Optional[int] = None,
    patience: int = 20,
    device: Optional[str] = None,
    verbose: bool = True,
    ground_truth: Optional[np.ndarray] = None,
    warm_start: bool = True,
    warm_start_epochs: int = 75,
    model_seed: Optional[int] = None,
    impute: Optional[str] = None,
    impute_kwargs: Optional[dict] = None,
    discrete_cols=None,
    ci_test=None,
    **model_kwargs,
) -> "GraceResult":
    """One-line interface for gated causal discovery with stability selection.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with shape [T, num_vars].
    max_lag : int
        Maximum lag to include.
    lambda_grid : numpy.ndarray or None
        Array of L0 penalties (default: logspace(-4, 0, 25)).
    n_subsamples : int
        Number of data subsamples per lambda.
    subsample_frac : float
        Fraction of data per subsample.
    skeleton : numpy.ndarray or None
        Pre-computed CI skeleton with shape [cause, effect, lag],
        binary.
    use_ci_skeleton : bool
        If True and skeleton is None, run CI tests to get skeleton.
    ci_alpha : float
        Significance level for CI skeleton (only used if
        use_ci_skeleton=True).
    ci_test_class : class or None
        CI test class (only used if use_ci_skeleton=True).
    include_C : bool, default True
        Include the C nonstationarity node in the CI skeleton (only used
        if use_ci_skeleton=True). True matches the historical behaviour.
    c_preset : str, default ``'linear'``
        C basis preset for the CI skeleton's nonstationarity node (same
        options as ``run_cdnots``; only used if use_ci_skeleton=True).
    use_skeleton_mask : bool
        If True, hard-mask non-skeleton edges (cannot open). Default
        False: skeleton only used for initialization (recommended for
        stability selection).
    stability_threshold : float
        Threshold for final graph.
    normalize : bool
        Z-score per variable.
    normalize_l0 : bool
        If True, scale L0 penalty by 1/steps_per_epoch so that
        regularization strength is independent of batch size. When
        True, batch_size is a pure compute knob; when False, legacy
        behaviour where batch size affects effective sparsity pressure.
    max_epochs : int
        Maximum training epochs.
    batch_size : int or None
        Training batch size. Auto-selected if None.
    patience : int
        Early stopping patience.
    device : str or None
        Accelerator device.
    verbose : bool
        Print progress.
    ground_truth : numpy.ndarray or None
        Ground truth with shape [cause, effect, lag], binary. If
        provided, print evaluation.
    impute : str or None
        Missing-value strategy applied before discovery.
        ``None`` — no imputation. ``'var_em'`` — iterative VAR-EM.
    impute_kwargs : dict or None
        Extra keyword arguments forwarded to the imputer.
    discrete_cols : list or None
        Column names or indices of discrete variables.
    ci_test : object, optional
        Accepted so ``causalts.algorithms.run_algorithm`` can pass its CI-test
        instance; not used (a warning says so). GRACE builds its CI test on its
        transformed data; choose the test with ``ci_test_class``.
    **model_kwargs : dict
        Passed to GatedCausalDiscovery.

    Returns
    -------
    GraceResult
        Result object with attributes ``cg_tig`` (binary graph),
        ``stability_scores`` (edge frequencies in [0, 1]), and
        inherited plotting / DoWhy bridge methods.
    """
    _warn_unused_ci_test(ci_test, "run_stability_selection")
    _check_model_kwargs(
        model_kwargs,
        "run_stability_selection",
        reserved=(
            "num_vars",
            "max_lag",
            "lambda_l0",
            "normalize_l0",
            "steps_per_epoch",
        ),
    )
    if (
        model_kwargs.get("ar_path")
        or model_kwargs.get("ar_hidden") is not None
        or model_kwargs.get("ar_exog_cols")
    ):
        # The AR head would be trained but no self-loop test run, so self-loops
        # (masked out of the gates) could never be selected.
        raise ValueError(
            "ar_path / ar_hidden / ar_exog_cols are supported by run_cdnots_gated only"
        )
    num_vars = df.shape[1]

    # ---- Pre-impute df so both skeleton and neural model use clean data ----
    if impute is not None and df.isna().any().any():
        import warnings as _w

        if impute in ("var_em", "causal_iterative"):
            if impute == "causal_iterative":
                _w.warn(
                    "impute='causal_iterative' is not supported in GRACE "
                    "(circular dependency with cdnots). Falling back to 'var_em'.",
                    UserWarning,
                    stacklevel=2,
                )
            from ..imputation import var_em_impute

            df = var_em_impute(df, **(impute_kwargs or {}))

    dataset = prepare_data(df, max_lag, normalize=normalize)
    N = len(dataset)

    # Batch size selection.
    # With normalize_l0=True the L0 penalty is divided by steps_per_epoch,
    # making regularization strength independent of batch size.  Batch size
    # becomes a pure compute/noise knob so we can use larger batches.
    # With normalize_l0=False (legacy), smaller batches = more steps = stronger
    # effective sparsity, so we target ~15-20 steps/epoch.
    if batch_size is None:
        if normalize_l0:
            # L0 normalization decouples regularization from batch size,
            # but we still want enough steps per epoch for good training
            # dynamics (~8+ steps).  Cap at N//8 so small datasets don't
            # degenerate to near-full-batch.
            batch_size = max(32, min(N // 8, 256))
        else:
            batch_size = max(16, min(N // 15, 64))
        if verbose:
            steps_per_epoch = math.ceil(N / batch_size)
            print(
                f"Auto batch_size={batch_size} (N={N}, ~{steps_per_epoch} steps/epoch)"
            )

    # Check if problem is underdetermined and warn (but don't auto-enable)
    num_candidates = num_vars * num_vars * (max_lag + 1) - num_vars  # exclude self@lag0
    samples_per_edge = N / num_candidates
    if skeleton is None and not use_ci_skeleton and samples_per_edge < 2.0:
        if verbose:
            print(
                f"WARNING: Underdetermined problem — {N} samples for "
                f"{num_candidates} candidate edges ({samples_per_edge:.1f} "
                f"samples/edge). Consider use_ci_skeleton=True for better results."
            )

    # Run CI skeleton if requested (or auto-enabled)
    ci_pval_matrix = None
    if skeleton is None and use_ci_skeleton:
        if verbose:
            print("Running CI-based skeleton discovery...")
        skeleton, ci_pval_matrix = run_ci_skeleton(
            df,
            max_lag=max_lag,
            alpha=ci_alpha,
            ci_test_class=ci_test_class,
            include_C=include_C,
            c_preset=c_preset,
            normalize=normalize,
            verbose=verbose,
            return_pvalues=True,
        )

    # Default: 1 lag per (cause, effect) pair — real-world systems rarely
    # have a cause acting at multiple lags.  User can override via model_kwargs.
    default_mlpp = model_kwargs.pop("max_lags_per_pair", 1)

    # Auto-set per-pair max_lags_per_pair from CI p-values when available.
    # The skeleton at alpha=0.05 is loose (high recall). Use a strict alpha
    # to count only confidently significant lags, then cap at default_mlpp.
    if ci_pval_matrix is not None and model_kwargs.get("lambda_lag_group", 0) > 0:
        strict_alpha = ci_alpha / 10  # e.g., 0.005 when ci_alpha=0.05
        confident_edges = (ci_pval_matrix <= strict_alpha).astype(np.int8)
        per_pair_max = np.clip(
            confident_edges.sum(axis=2), 1, int(default_mlpp)
        )  # [cause, effect]
        if verbose:
            skel_lags = skeleton.sum(axis=2)
            n_reduced = int((per_pair_max < skel_lags).sum())
            print(
                f"Per-pair lag budget (strict α={strict_alpha:.4f}, max={default_mlpp}): "
                f"{n_reduced} pairs tightened vs skeleton"
            )
        model_kwargs["max_lags_per_pair"] = per_pair_max
    else:
        model_kwargs["max_lags_per_pair"] = default_mlpp

    selector = StabilitySelector(
        num_vars=num_vars,
        max_lag=max_lag,
        lambda_grid=lambda_grid,
        n_subsamples=n_subsamples,
        subsample_frac=subsample_frac,
        skeleton=skeleton,
        use_skeleton_mask=use_skeleton_mask,
        model_kwargs=model_kwargs,
        normalize_l0=normalize_l0,
        warm_start=warm_start,
        warm_start_epochs=warm_start_epochs,
        seed=model_seed,
    )

    selector.run(
        dataset,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        patience=patience,
        verbose=verbose,
    )

    G_hat, scores = selector.get_graph(threshold=stability_threshold)

    if verbose:
        n_edges = int(G_hat.sum())
        efp = selector.expected_false_positives(stability_threshold)
        print(f"\nResult: {n_edges} edges at threshold={stability_threshold}")
        print(f"Expected false positives (upper bound): {efp:.2f}")

    if ground_truth is not None:
        metrics = evaluate_graph(G_hat, ground_truth)
        if verbose:
            print("\n--- Evaluation vs Ground Truth ---")
            print(
                f"  TP={metrics['TP']}  FP={metrics['FP']}  "
                f"FN={metrics['FN']}  TN={metrics['TN']}"
            )
            print(
                f"  TPR={metrics['TPR']:.3f}  FPR={metrics['FPR']:.3f}  "
                f"Precision={metrics['Precision']:.3f}  F1={metrics['F1']:.3f}"
            )
            print(f"  SHD={metrics['SHD']}  SHD_pair={metrics['SHD_pair']}")

    from .result import GraceResult

    return GraceResult(
        graph=G_hat,
        df=df,
        var_names=list(df.columns),
        stability_scores=scores,
    )


# ---------------------------------------------------------------------------
# CDNOTS + Gated Refinement (single-run, no stability selection)
# ---------------------------------------------------------------------------


def _check_ar_hidden(ar_hidden) -> None:
    if (
        isinstance(ar_hidden, (bool, np.bool_))
        or not isinstance(ar_hidden, (int, np.integer))
        or ar_hidden < 1
    ):
        raise ValueError(f"ar_hidden must be a positive integer, got {ar_hidden!r}")


def _warn_unused_ci_test(ci_test, caller: str) -> None:
    # causalts.algorithms.run_algorithm passes every algorithm a CI-test instance.
    # GRACE builds its own on the transformed data (``ci_test_class``), so an
    # instance is not used; say so rather than ignore it silently.
    if ci_test is not None:
        warnings.warn(
            f"{caller}() does not use the ci_test instance; GRACE builds its CI "
            "test on its transformed data. Pass ci_test_class to choose the test.",
            UserWarning,
            stacklevel=3,
        )


def _check_model_kwargs(model_kwargs: dict, caller: str, reserved=()) -> None:
    """Fail fast on keyword arguments ``GatedCausalDiscovery`` does not take, or that
    ``caller`` sets itself, before any expensive skeleton or training run."""
    allowed = set(inspect.signature(GatedCausalDiscovery.__init__).parameters) - {
        "self",
        "kwargs",
    }
    unknown = sorted(set(model_kwargs) - allowed)
    if unknown:
        raise TypeError(
            f"{caller}() got unexpected keyword argument(s): {', '.join(unknown)}"
        )
    clash = sorted(set(model_kwargs) & set(reserved))
    if clash:
        raise TypeError(f"{caller}() sets {', '.join(clash)} itself; do not pass it")


def _cdnots_graph_to_binary(cg_tig, n_vars: int, max_lag: int) -> np.ndarray:
    """Convert tigramite/cdnots graph array to binary [cause, effect, lag]."""
    # Trim to original vars (cdnots may append a C variable)
    cg_trimmed = cg_tig[:n_vars, :n_vars, :]
    if cg_trimmed.dtype.kind in ("U", "S", "O"):
        # String graph (e.g. '-->', '<--')
        G = np.zeros((n_vars, n_vars, max_lag + 1), dtype=np.int8)
        for i in range(cg_trimmed.shape[0]):
            for j in range(cg_trimmed.shape[1]):
                for lag in range(min(cg_trimmed.shape[2], max_lag + 1)):
                    s = str(cg_trimmed[i, j, lag]).strip()
                    if s == "-->":
                        G[i, j, lag] = 1
                    elif s == "<--":
                        G[j, i, lag] = 1
        return G
    else:
        # Already binary int array — align lag dimension
        G = cg_trimmed.astype(np.int8)
        _, _, L = G.shape
        target_L = max_lag + 1
        if L < target_L:
            pad = np.zeros((n_vars, n_vars, target_L - L), dtype=np.int8)
            G = np.concatenate([G, pad], axis=2)
        elif L > target_L:
            G = G[:, :, :target_L]
        return G


def run_cdnots_gated(
    df: pd.DataFrame,
    max_lag: int,
    alpha: float = 0.05,
    ci_test_class=None,
    skeleton: Optional[np.ndarray] = None,
    skeleton_runtime: float = 0.0,
    gate_threshold: Union[float, str] = "auto",
    normalize: bool = True,
    max_epochs: int = 150,
    batch_size: Optional[int] = None,
    patience: int = 20,
    device: Optional[str] = None,
    verbose: bool = True,
    ground_truth: Optional[np.ndarray] = None,
    model_seed: Optional[int] = None,
    include_C: bool = True,
    include_C_in_model: bool = False,
    c_preset: str = "linear",
    lambda_cv: bool = False,
    lambda_rule: str = "xcorr",
    impute: Optional[str] = None,
    impute_kwargs: Optional[dict] = None,
    discrete_cols=None,
    grow_rounds: Optional[int] = None,
    grow_q: float = 0.1,
    grow_transform: Optional[str] = "auto",
    data_transform: Optional[str] = "rank",
    noise_model: str = "student_t",
    refine: str = "gates",
    shrink_alpha: float = 0.001,
    shrink_basis: str = "spline",
    shrink_robust: bool = True,
    shrink_interactions: bool = False,
    *,
    grow_variance: Optional[bool] = None,
    grow_nuisance_basis: Optional[str] = "spline",
    effect_floor: float = EFFECT_FLOOR,
    shrink_effect_floor: Union[None, str, float] = None,
    ar_path: Optional[bool] = None,
    self_loop_basis: str = "union",
    contemporaneous: bool = False,
    contemporaneous_test="spline",
    contemporaneous_alpha: float = 0.001,
    ci_test=None,
    **model_kwargs,
) -> "GraceResult":
    """CDNOTS skeleton + refinement (no stability selection).

    1. Run CDNOTS to get a high-recall binary skeleton (or use ``skeleton``).
    2. Refine it: train a gated model on the candidates to get continuous gate
       values (``refine="gates"``), or prune each target's candidate parents by
       regression (``refine="shrink"``). Grow rounds test the lagged pairs
       outside the candidates, add the hits and refine again (by default one
       round under ``"gates"``, three under ``"shrink"``).
    3. Cut the gates into a graph (gates only; the shrink's survivors are the
       graph). Under the default ``ar_path`` self-loops are decided by an F-test,
       not by gates, and the effect floor then removes cross edges with a
       negligible effect, so ``gate_values`` alone do not determine ``cg_tig``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with shape [T, num_vars].
    max_lag : int
        Maximum lag to include.
    alpha : float
        CDNOTS significance level.
    ci_test_class : class or None
        CI test class (default: PartialCorr).
    skeleton : numpy.ndarray or None
        Optional pre-computed binary skeleton array with shape
        (d, d, max_lag+1); with ``include_C_in_model=True`` and
        ``refine="gates"``, (d + n_C, d + n_C, max_lag+1), the C columns last.
        If provided, skips CDNOTS and uses this skeleton directly. Under the
        default ``grow_rounds`` (1 with ``refine="gates"``) the grow round can add
        lagged edges outside it, which repairs edges the supplied graph missed,
        and under the default ``ar_path`` the self-loop F-test can add self-loops
        it lacks; pass ``grow_rounds=0, ar_path=False`` to keep only edges inside
        the skeleton.
    skeleton_runtime : float
        Wall-clock time of the pre-computed skeleton (for info dict).
    gate_threshold : float, ``'auto'`` or ``'li'``, default ``'auto'``
        Cutoff on gate values for the final graph. A float keeps edges with
        gate ``>=`` the value (``0.5`` reproduces the old default). ``'li'``
        picks the cutoff from the data with :func:`li_threshold` on the
        candidate gates and keeps edges strictly above it (0.5 if too few
        gates sit on either side). ``'auto'`` uses ``'li'`` when GRACE builds
        its own skeleton, and for a supplied ``skeleton`` only when
        :func:`gates_bimodal` finds two clusters; otherwise 0.5, so a precise
        skeleton is not split in half. With ``ar_path`` (the default under
        ``refine="gates"``) a Li cutoff is never below 0.5 (see ``ar_path``). The
        applied cutoff is stored in ``result.gate_threshold`` and the rule in
        ``result.gate_threshold_rule``: ``'fixed'``, ``'li'``, ``'li_fallback'``
        or ``'auto_unimodal'``; with ``ar_path`` ``'ar_fixed'``,
        ``'ar_li_floor'`` (Li, floored at 0.5), ``'ar_li_fallback'`` or
        ``'ar_auto_unimodal'``.
    normalize : bool
        Z-score normalize data.
    max_epochs : int
        Training epochs.
    batch_size : int or None
        Training batch size (auto if None).
    patience : int
        Early stopping patience.
    device : str or None
        Accelerator ('mps', 'cuda', 'cpu', or None for auto).
    verbose : bool
        Print progress.
    ground_truth : numpy.ndarray or None
        Optional ground truth for evaluation.
    model_seed : int or None
        Seed for reproducibility.
    include_C : bool, default True
        Include the C nonstationarity node in the CDNOTS skeleton. True
        matches the historical behaviour (the skeleton always had a C
        node); set False for data you have already detrended or
        differenced. Ignored when ``skeleton`` is supplied.
    include_C_in_model : bool
        If True, append the C nonstationarity basis to the data before
        training the gated model (mirrors CDNOTS nonstationarity
        handling). C-related edges are stripped from the output graph.
        With ``refine="gates"`` it requires ``include_C=True``, since the
        model's C columns are masked by the skeleton's. It conditions the grow
        tests (and the ``ar_path`` self-loop test) on the C basis, also for a
        supplied skeleton; with ``refine="shrink"`` every shrink regression too,
        independently of ``include_C``.
    c_preset : str, default ``'linear'``
        C basis preset for the nonstationarity node, applied to the CDNOTS
        skeleton and, when ``include_C_in_model`` is True, to the columns
        appended for gated training. Same options as ``run_cdnots``: ``'linear'``,
        ``'linear+quad'``, ``'linear+sin'``, ``'linear+exp'``, ``'step'``,
        ``'step+linear'``. Presets that expand to more than one basis
        column (e.g. ``'linear+sin'``) add that many C columns; all of them
        are stripped from the returned graph.
    lambda_cv : bool
        If True, select lambda via cross-validation instead of the
        closed-form formula (Eq. 11, arXiv:2606.23880). Slower but
        data-driven.
    lambda_rule : {"density", "dT", "xcorr"}, default "xcorr"
        Closed-form rule for lambda when neither ``lambda_cv`` nor a
        ``lambda_l0`` override is given (see :func:`compute_lambda`); no effect
        under ``refine="shrink"``.
        ``"density"`` is the formula above. ``"dT"`` is ``300 / (d * T)``,
        with d the number of modelled variables and T the series length: it
        keeps the per-edge evidence needed to open a gate from growing with d,
        which under ``"density"`` drops true nonlinear edges at large d and T.
        ``"xcorr"`` also scales with the data's background lagged
        cross-correlation and is bounded to ``[0.3 / d, 0.5]``, so it does not
        vanish on long series (see :func:`compute_lambda`).
    impute : str or None
        Missing-value strategy applied before discovery.
        ``None`` — no imputation. ``'var_em'`` — iterative VAR-EM.
    impute_kwargs : dict or None
        Extra keyword arguments forwarded to the imputer.
    discrete_cols : list or None
        Column names or indices of discrete variables.
    grow_rounds : int or None, default None
        Maximum grow-shrink rounds after the first fit (0 = off); None means 1
        under ``refine="gates"`` and 3 under ``refine="shrink"`` (the REAP
        configuration). Each round
        tests every lagged pair outside the candidate set with
        :func:`~causalts.grace.grow.grow_candidates`, adds the hits to the
        candidates, and refines again on the enlarged set (retrains the gates,
        or re-runs the shrink with ``refine="shrink"``). Stops early once a
        round adds nothing. Repairs edges the skeleton missed, which pruning
        alone cannot recover; a missing parent that acts only through an
        interaction with other parents is not reliably recovered. Under
        ``"gates"`` each round costs one more training run.
    grow_q : float, default 0.1
        Benjamini-Hochberg level for the grow tests.
    grow_transform : {"auto", None, "rank"}, default "auto"
        Passed to :func:`~causalts.grace.grow.grow_candidates`; ``"rank"``
        runs the grow tests on rank-Gaussianized data. ``"auto"`` means None
        under ``refine="gates"`` and ``"rank"`` under ``refine="shrink"``.
    data_transform : {"rank", None}, default "rank"
        ``"rank"`` maps every continuous column to normal scores of its ranks
        (missing values stay missing) before the skeleton, training and grow
        steps, which tames heavy-tailed noise; discrete columns keep their codes,
        except that under ``refine="shrink"`` with the spline basis the shrink and
        grow see them rank-transformed too. ``None`` uses the raw data.
    noise_model : {"student_t", "gaussian"}, default "student_t"
        Likelihood of the gated model's prediction errors. ``"student_t"``
        learns a per-variable degrees-of-freedom, so a few extreme residuals
        cannot dominate the fit; ``"gaussian"`` is the previous default.
    refine : {"gates", "shrink"}, default "gates"
        How the candidate skeleton is refined. ``"gates"`` trains the neural
        gated model (Hard Concrete gates with an L0 penalty) and cuts the gates.
        ``"shrink"`` trains nothing: each target's candidate parents are pruned
        by regression-based backward elimination
        (:func:`~causalts.grace.shrink.shrink_parents`), and ``grow_rounds``
        alternate grow and shrink. Every shrink and grow regression conditions on
        the C basis on GRACE's own skeleton (``include_C``), and on a supplied
        skeleton when ``include_C_in_model=True``. Typically an order of magnitude faster than
        ``"gates"``. Under ``"shrink"`` the gate and training options
        (``gate_threshold``, ``noise_model``, ``lambda_cv``, ``lambda_rule``, ``max_epochs``,
        ``batch_size``, ``patience``, ``device``, ``normalize``, ``model_seed``,
        ``**model_kwargs``) have no effect; lagged edges only (lag >= 1) unless
        ``contemporaneous=True``.
    shrink_alpha : float, default 0.001
        ``refine="shrink"``: significance level for keeping a parent. The shrink's
        p-values are approximate (fixed-basis F-tests) and become liberal at large
        T when effects are strongly curved; see
        :func:`~causalts.grace.shrink.shrink_parents`.
    shrink_basis : {"spline", "rff"}, default "spline"
        ``refine="shrink"``: feature block per lagged parent, used by both the
        shrink and the grow tests. ``"spline"`` (natural cubic spline, knots from
        the sample minimum to maximum), on rank-transformed data, gives extreme
        rows much lower leverage than ``"rff"``. Variables with at most 8 distinct
        values get level indicators, after levels seen in fewer than 5 rows are
        merged into a neighbour; a variable whose levels all merge into one (e.g.
        a binary one whose rare level has fewer than 5 rows) gets none (with a
        warning), so it cannot be selected.
    shrink_robust : bool, default True
        ``refine="shrink"``: Huber-clip residuals before each shrink test, which
        limits the influence of heavy-tailed noise.
    shrink_interactions : bool, default False
        ``refine="shrink"``: add pairwise products of each target's parents to the
        shrink and grow regressions (see
        :func:`~causalts.grace.shrink.shrink_parents`). Keeps precision when
        parents interact (non-additive mechanisms), where the additive fit lets
        correlated non-parents in as T grows; costs some power when targets
        have many parents.
    grow_variance : bool or None, default None
        None means True under ``refine="gates"`` with ``grow_rounds >= 1``, else
        False. Each grow round also proposes lagged edges that change a
        target's noise scale rather than its mean
        (:func:`~causalts.grace.grow.grow_variance_candidates`); the gated model's
        likelihood can keep them, a mean-based test cannot see them. Needs
        ``refine="gates"`` and ``grow_rounds >= 1``. Its test always runs on
        rank-Gaussianized data, whatever ``grow_transform``.
    grow_nuisance_basis : {"spline", "rff", None}, default "spline"
        Basis of the conditioning (parent) terms in the grow tests and, with
        ``ar_path``, the self-loop test; the tested candidate keeps its own block.
        ``"spline"``: a natural cubic spline with boundary knots at the sample
        minimum and maximum. ``None``: the tested block's basis (the RFF block
        under ``refine="gates"``) and a linear C, as before this option (the
        conditioning on kept lag-0 parents and, with ``include_C_in_model``, on
        C for a supplied skeleton still applies). With a fixed
        RFF block the part of a persistent parent's effect it cannot fit becomes
        significant at large T, and a candidate that proxies the same drivers
        absorbs it: on null configurations (T=1000, driver autocorrelation 0.9)
        the RFF version rejected up to 32% of true nulls at level 0.001, the
        spline version about 2% or less in the tested settings. Any fixed basis
        can still over-reject at large T when a parent's effect is more wiggly
        than it can fit. A discrete parent enters the spline nuisance through one
        indicator per level, without merging rare levels.
        Under the gates the tests also take each C column through a 4-knot spline
        (with ``data_transform="rank"`` a linear trend is curved on the
        transformed scale), so they remove smooth trends C describes, at some
        cost in self-loop power at small T; a trend C does not describe, or a
        nearly noise-free one, can still make lagged proxies look causal, so
        detrend first or use ``data_transform=None`` there. Under
        ``refine="shrink"`` with the spline basis, ``"spline"`` and ``None`` give
        the same results on continuous data and differ only for a discrete
        parent with a level seen in fewer than 5 rows.
    effect_floor : float, default ``EFFECT_FLOOR`` (= 1 - exp(-0.02), about 0.0198)
        Minimum effect of an edge, as an adjusted partial R^2: the share of its
        target's unexplained variance (after the target's own lags and other parents)
        the edge explains, corrected for chance fit. Free of units and of T. It is
        applied twice: the grow tests add a BH-rejected candidate only if its effect
        reaches the floor, and after the gates (and the lag-0 step) every lagged cross
        edge whose unique effect on the mean or the scale of its target is below it is
        pruned (:func:`~causalts.grace.grow.prune_effect_floor`; self-loops and lag-0
        edges stay). The default is 0.01 nats of predictive gain per observation.
        Significance tests and the L0 gates keep any real dependence once T is large,
        so on long real series (only approximately sparse) the graph otherwise fills
        with real but negligible edges. ``0`` turns both off (the behaviour before
        this option). Applies with ``refine="gates"``.
        ``gate_values`` and ``gate_threshold`` of the result describe the gates
        before this pruning; ``effect_pruned`` marks the edges it removed, so a
        gate-based score that should agree with ``cg_tig`` must zero those.
    shrink_effect_floor : None, 'adaptive' or float, default None
        ``refine="shrink"`` only. After the shrink and grow rounds (before the lag-0
        step), prune lagged cross edges whose unique effect on their target is
        small (:func:`~causalts.grace.grow.prune_effect_floor` with the shrink's
        blocks and C conditioning; self-loops stay). ``'adaptive'``: per target,
        backward elimination of edges below ``max(0.01, 0.2 * strongest)``,
        ``strongest`` the largest effect among the target's cross edges
        (``SHRINK_EFFECT_FLOOR_ADAPTIVE``), so the threshold scales with each
        target's signal; a float is a fixed floor. It also drops real parents
        weaker than a fifth of the target's strongest (or below 0.01), so leave it
        off when such weak parents matter more than precision. Not available with
        ``shrink_interactions=True`` (it scores additive blocks). ``None`` keeps the
        shrink's graph (the behaviour before this option). On long series (T in the
        thousands) the shrink and grow tests keep real but negligible
        dependencies; on held-out CausalRivers river graphs (T=7304) ``'adaptive'``
        cut SHD per graph from 11.4 to 6.4 and raised F1 from 0.40 to 0.44, and on
        simulated graphs with T up to 1000 it changed lagged F1 by +0.004 on
        average (no setting lost more than 0.007). ``edge_pvalue`` and
        ``edge_strength`` still describe the shrink's tests; ``effect_pruned``
        marks the edges this step removed.
    ar_path : bool or None, default None
        None means True under ``refine="gates"`` with ``max_lag >= 1``, else
        False. ``refine="gates"`` only. The gated model gets an ungated
        per-target head on the target's own lags (linear plus a small MLP), and
        self-loops leave the gated part. Cross edges are cut at the larger of
        the Li cutoff of the cross gates and 0.5 under ``gate_threshold`` 'li'
        and, for GRACE's own skeleton, 'auto'; a supplied skeleton under 'auto'
        gets that cut only when its cross gates form two clusters, else 0.5
        (as without ``ar_path``). A float cuts at that value. Each self-loop lag
        is then kept if a partial F-test of adding it (the lag alone and a
        4-knot spline of it, Bonferroni over the tests run;
        ``self_loop_test(test_basis="union")``) to an additive regression of the target on its kept cross parents (lag-0
        ones under ``include_instantaneous`` if not kept in both directions), its
        other own lags and the C basis (when the grow tests use it or the model
        includes C) gives p < 0.001 (per test, not family-wise: nominally
        0.001 * d * max_lag false self-loops when none are real, more where the
        null is misspecified, e.g. a trend C does not fully remove;
        :func:`~causalts.grace.grow.self_loop_test`). Strongly autoregressive
        targets otherwise make the gates of real cross edges collapse, and the
        gated self-loop absorbs signal. The test always runs on rank-Gaussianized
        data, whatever ``data_transform`` (on raw heavy-tailed data the F-test
        over-rejects), and warns when a test is undefined (too few rows for the
        kept parents and lags); that self-loop is then not kept. It is on the
        conditional mean, so an own lag that acts only on the noise scale
        (ARCH-type) is not found under ``ar_path``; grow rounds leave self-loops
        to the test. Self-loop entries of ``gate_values`` are 0 under
        ``ar_path``: they are not gated, and the formula lambda's density counts
        the cross candidates only.
    self_loop_basis : {"union", "rff"}, default "union"
        Tested block of the ``ar_path`` self-loop test
        (``self_loop_test(test_basis=...)``). ``"rff"`` reproduces ``ar_path``
        output from before the union test. Ignored without ``ar_path``.
    contemporaneous : bool, default False
        ``refine="shrink"`` only. After the shrink and grow rounds, find
        contemporaneous (lag-0) edges with CDNOTS+
        (:func:`~causalts.cdnots.phase3_utils.run_cdnots_plus`) on the shrink's
        data, with every lagged edge pinned to the shrink's selection (kept edges
        required, all others forbidden) and C included exactly when the shrink
        conditioned on it: CDNOTS+ tests lag-0 adjacencies conditioning on the
        selected lagged parents, and orients them by its collider and Meek rules
        (which use the selected lagged arrows) and, with C, its nonstationarity
        rule. Oriented lag-0 edges go into ``cg_tig[:, :, 0]``;
        unoriented ones are dropped there, as in CDNOTS+, and kept in
        ``to_marks()`` (``"o-o"``) and in ``contemporaneous_result``. The lagged
        edges are unchanged. With the default False, the shrink stays lagged-only and
        its output is unchanged.
    contemporaneous_test : str or CIT_Base, default "spline"
        CI test for the lag-0 step. ``"spline"`` is the natural-spline block
        partial F with Huber-clipped residuals (the shrink's own test under
        ``shrink_basis="spline"``), both directions Cauchy-combined, on every
        column rank-transformed (with ``data_transform="rank"``); its p-values
        share the shrink's approximations: liberal when effects are wiggly. Otherwise a registered
        name (:func:`~causalts.ci_tests.create_ci_test`), e.g. ``"raft"`` for a
        calibrated test at some cost in power at small T, or an instance.
    contemporaneous_alpha : float, default 0.001
        Significance level of the lag-0 step.
    ci_test : object, optional
        Accepted so ``causalts.algorithms.run_algorithm`` can pass its CI-test
        instance; not used (a warning says so). GRACE builds its CI test on its
        transformed data; choose the test with ``ci_test_class``.
    **model_kwargs : dict
        Forwarded to GatedCausalDiscovery.

    Returns
    -------
    GraceResult
        Result object with attributes ``cg_tig`` (binary graph),
        ``gate_values`` (continuous gate values; ``None`` under
        ``refine="shrink"``, which sets ``edge_pvalue`` and ``edge_strength``
        instead), ``skeleton``, ``lambda_l0``, ``gate_threshold``, ``refine``,
        ``runtime``, and inherited plotting / DoWhy bridge methods.
    """
    import time as _time
    import warnings

    from ..cdnots.phase3_utils import _C_PRESET_LABELS, make_c_array, run_cdnots

    # Defaults that depend on the refinement: the gated configuration of the GRACE
    # paper (AR head, one grow round with scale candidates) under "gates"; the REAP
    # configuration (three grow rounds on rank-transformed data) under "shrink".
    gates = refine == "gates"
    if grow_rounds is None:
        grow_rounds = 1 if gates else 3
    if grow_transform == "auto":
        grow_transform = None if gates else "rank"
    if grow_variance is None:
        grow_variance = gates and grow_rounds >= 1
    if ar_path is None:
        ar_path = gates and max_lag >= 1

    t0 = _time.time()
    num_vars = df.shape[1]
    _warn_unused_ci_test(ci_test, "run_cdnots_gated")
    _check_model_kwargs(
        model_kwargs,
        "run_cdnots_gated",
        reserved=(
            "num_vars",
            "max_lag",
            "noise_model",
            "normalize_l0",
            "steps_per_epoch",
            "ar_exog_cols",
        ),
    )
    if isinstance(gate_threshold, str) and gate_threshold not in ("auto", "li"):
        raise ValueError(
            f"gate_threshold must be a float, 'auto' or 'li', got {gate_threshold!r}"
        )
    own_skeleton = skeleton is None
    if lambda_rule not in ("density", "dT", "xcorr"):
        raise ValueError(
            f"lambda_rule must be 'density', 'dT' or 'xcorr', got {lambda_rule!r}"
        )
    if refine not in ("gates", "shrink"):
        raise ValueError(f"refine must be 'gates' or 'shrink', got {refine!r}")
    if refine == "shrink":
        if model_kwargs.get("include_instantaneous"):
            raise ValueError("refine='shrink' handles lagged edges only (lag >= 1)")
        if max_lag < 1:
            raise ValueError("refine='shrink' needs max_lag >= 1 (lagged edges only)")
        if shrink_basis not in ("spline", "rff"):
            raise ValueError(
                f"shrink_basis must be 'spline' or 'rff', got {shrink_basis!r}"
            )
        if not 0 < shrink_alpha < 1:
            raise ValueError(f"shrink_alpha must be in (0, 1), got {shrink_alpha!r}")
        if impute not in ("var_em", "causal_iterative") and df.isna().any().any():
            raise ValueError(
                "refine='shrink' needs data without missing values; impute first "
                "(e.g. impute='var_em')"
            )
    if contemporaneous:
        if refine != "shrink":
            raise ValueError("contemporaneous=True needs refine='shrink'")
        if not 0 < contemporaneous_alpha < 1:
            raise ValueError(
                "contemporaneous_alpha must be in (0, 1), got "
                f"{contemporaneous_alpha!r}"
            )
        from .contemporaneous import check_ci_test

        check_ci_test(contemporaneous_test)
    if not 0 <= effect_floor < 1:
        raise ValueError(f"effect_floor must be in [0, 1), got {effect_floor!r}")
    if shrink_effect_floor is not None:
        if refine != "shrink":
            raise ValueError("shrink_effect_floor applies with refine='shrink' only")
        if shrink_interactions:
            # the floor scores each parent's additive block alone, so a parent that
            # acts only through an interaction would be pruned
            raise ValueError(
                "shrink_effect_floor does not support shrink_interactions=True"
            )
        if isinstance(shrink_effect_floor, str):
            if shrink_effect_floor != "adaptive":
                raise ValueError(
                    "shrink_effect_floor must be None, 'adaptive' or a float in "
                    f"[0, 1), got {shrink_effect_floor!r}"
                )
        elif isinstance(shrink_effect_floor, bool) or not (
            isinstance(shrink_effect_floor, numbers.Real)
            and 0 <= shrink_effect_floor < 1
        ):
            raise ValueError(
                "shrink_effect_floor must be None, 'adaptive' or a float in [0, 1), "
                f"got {shrink_effect_floor!r}"
            )
    if grow_rounds > 0:  # checked here so a typo fails before the skeleton run
        if not 0 < grow_q < 1:
            raise ValueError(f"grow_q must be in (0, 1), got {grow_q!r}")
        if grow_transform not in (None, "rank"):
            raise ValueError(
                f"grow_transform must be None or 'rank', got {grow_transform!r}"
            )
    if grow_nuisance_basis not in (None, "rff", "spline"):
        raise ValueError(
            "grow_nuisance_basis must be None, 'rff' or 'spline', got "
            f"{grow_nuisance_basis!r}"
        )
    if self_loop_basis not in ("union", "rff"):
        raise ValueError(
            f"self_loop_basis must be 'union' or 'rff', got {self_loop_basis!r}"
        )
    if grow_variance and (refine != "gates" or grow_rounds < 1):
        raise ValueError(
            "grow_variance needs refine='gates' and grow_rounds >= 1, got "
            f"refine={refine!r}, grow_rounds={grow_rounds!r}"
        )
    if model_kwargs.get("ar_hidden") is not None:
        if not ar_path:
            raise ValueError("ar_hidden only applies with ar_path=True")
        _check_ar_hidden(model_kwargs["ar_hidden"])
    if ar_path:
        if refine != "gates":
            raise ValueError("ar_path is a gated-model option; it needs refine='gates'")
        if max_lag < 1:
            raise ValueError("ar_path needs max_lag >= 1 (self-loops are lagged)")
        model_kwargs["ar_path"] = True
    if data_transform not in (None, "rank"):
        raise ValueError(
            f"data_transform must be None or 'rank', got {data_transform!r}"
        )
    # Validate the preset up front (and learn its width) so a typo fails before
    # the expensive skeleton run rather than after it.
    if include_C or include_C_in_model:
        c_width = make_c_array(len(df), c_preset).shape[1]
    n_c_cols = c_width if include_C else 0
    if include_C_in_model and not include_C and refine == "gates":
        raise ValueError(
            "include_C_in_model=True requires include_C=True: the model's C "
            "columns are masked by the skeleton's, so a skeleton without C "
            "leaves them permanently closed."
        )

    # ---- Pre-impute df so both skeleton and neural model use clean data ----
    if impute is not None and df.isna().any().any():
        if impute in ("var_em", "causal_iterative"):
            if impute == "causal_iterative":
                warnings.warn(
                    "impute='causal_iterative' is not supported in GRACE "
                    "(circular dependency with cdnots). Falling back to 'var_em'.",
                    UserWarning,
                    stacklevel=2,
                )
            from ..imputation import var_em_impute

            df = var_em_impute(df, **(impute_kwargs or {}))
            if verbose:
                print(
                    f"  Imputed {df.isna().sum().sum()} remaining NaN values via VAR-EM"
                )

    df_input = df  # the result keeps the caller's data, not the transformed copy
    # "xcorr" lambda rule: background dependence of the caller's (imputed) data,
    # measured before any rank transform. The rule was calibrated on Pearson
    # correlations of the untransformed series, also for runs whose model saw ranks.
    xcorr = (
        background_xcorr(df.to_numpy(dtype=float), max_lag)
        if lambda_rule == "xcorr"
        and refine == "gates"
        and not lambda_cv
        and model_kwargs.get("lambda_l0") is None
        else None
    )
    spline_refine = refine == "shrink" and shrink_basis == "spline"
    X_refine = None  # shrink and grow data; None means use df as is
    lag0_spline = contemporaneous and isinstance(contemporaneous_test, str)
    lag0_spline = lag0_spline and contemporaneous_test == "spline"
    X_lag0 = None  # lag-0 step data when it differs from X_refine / df
    if lag0_spline and data_transform is None and not spline_refine:
        warnings.warn(
            "contemporaneous_test='spline' with data_transform=None builds spline "
            "features on raw values, whose boundary knots sit at the sample minimum "
            "and maximum, so an outlier can get leverage near 1 and decide a test. "
            "Use data_transform='rank' unless the data have no outliers.",
            UserWarning,
            stacklevel=2,
        )
    if spline_refine and data_transform is None:
        warnings.warn(
            "refine='shrink' with data_transform=None feeds raw values to the shrink "
            "and grow spline features, whose boundary knots sit at the sample "
            "minimum and maximum, so an outlier (or a rare extreme level of an "
            "integer column) can get leverage near 1 and decide its tests. Use "
            "data_transform='rank' unless the data have no outliers.",
            UserWarning,
            stacklevel=2,
        )
    if data_transform == "rank":
        from scipy.stats import norm, rankdata

        from ..cdnots.phase3_utils import _detect_discrete_cols

        # Resolve discrete columns on the raw data: ranking would turn their
        # integer codes into Gaussian scores and hide them from CDNOTS's own
        # detection. They stay unranked and are passed on explicitly.
        if discrete_cols is None:
            discrete_cols = _detect_discrete_cols(df, len(df))
            if discrete_cols and skeleton is None:
                # run_cdnots only warns when it detects them itself; keep the hint.
                warnings.warn(
                    f"Detected discrete columns: {discrete_cols}. CDNOTS will wrap "
                    "its CI test in StratifiedCIT for them (unless the test handles "
                    "discrete data natively). Pass discrete_cols=[] to disable.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # A selector is a column label first; an integer that is not a label
            # is a position.
            discrete_cols = [
                (
                    df.columns[c]
                    if c not in df.columns and isinstance(c, (int, np.integer))
                    else c
                )
                for c in discrete_cols
            ]
        X = df.to_numpy(dtype=float)
        X_all = X.copy()  # every column rank-transformed
        for j in range(X.shape[1]):
            ok = ~np.isnan(X[:, j])
            X_all[ok, j] = norm.ppf((rankdata(X[ok, j]) - 0.5) / ok.sum())
        disc = [j for j, col in enumerate(df.columns) if col in discrete_cols]
        Xr = X_all.copy()
        Xr[:, disc] = X[:, disc]
        if spline_refine:
            # The shrink's and grow's spline features need bounded tails (a rare
            # extreme value of a raw integer column would get leverage near 1), so
            # both see every column rank-transformed; only CDNOTS keeps the codes.
            X_refine = X_all
        elif lag0_spline:
            X_lag0 = X_all  # the lag-0 spline test needs the same bounded tails
        df = pd.DataFrame(Xr, index=df.index, columns=df.columns)

    # ---- Step 1: CDNOTS Skeleton ----
    if skeleton is not None and refine == "gates":
        want = num_vars + (n_c_cols if include_C_in_model else 0)
        if np.shape(skeleton) != (want, want, max_lag + 1):
            raise ValueError(
                f"skeleton must have shape {(want, want, max_lag + 1)} for "
                "refine='gates'"
                + (
                    f" (the {num_vars} variables followed by the {n_c_cols} C "
                    "columns, since include_C_in_model=True)"
                    if include_C_in_model
                    else ""
                )
                + f", got {np.shape(skeleton)}"
            )
    if skeleton is not None:
        t_skeleton = skeleton_runtime
        if verbose:
            print("Step 1: Using pre-computed skeleton")
    else:
        if verbose:
            print("Step 1: CDNOTS skeleton discovery...")

        if ci_test_class is None:
            ci_test_class = _get_parcorr_class()

        skeleton_test = ci_test_class(np.ascontiguousarray(df.values))
        cdnots_res = run_cdnots(
            df=df,
            indep_test=skeleton_test,
            num_lags=max_lag,
            include_C=include_C,
            c_preset=c_preset,
            alpha=alpha,
            stable=True,
            show_progress=verbose,
            discrete_cols=discrete_cols,
        )

        skel_vars = num_vars + n_c_cols if include_C_in_model else num_vars
        skeleton = _cdnots_graph_to_binary(cdnots_res.cg_tig, skel_vars, max_lag)
        t_skeleton = _time.time() - t0

    # Optionally append the C basis columns to the model data
    if include_C_in_model:
        C_cols = pd.DataFrame(
            make_c_array(len(df), c_preset),
            columns=_C_PRESET_LABELS.get(c_preset, [f"C{i}" for i in range(n_c_cols)]),
        )
        df_model = pd.concat([df.reset_index(drop=True), C_cols], axis=1)
        model_num_vars = num_vars + n_c_cols
        if ar_path:  # C is an input, not a target with its own autoregression
            model_kwargs["ar_exog_cols"] = n_c_cols
    else:
        df_model = df
        model_num_vars = num_vars

    n_possible = model_num_vars * model_num_vars * (max_lag + 1) - model_num_vars

    if (
        refine == "gates"
    ):  # neural training state; the regression-based shrink needs none
        # Prepare data (before lambda selection — CV needs the dataset)
        dataset = prepare_data(df_model, max_lag, normalize=normalize)
        N = len(dataset)
        if batch_size is None:
            batch_size = max(32, min(N // 8, 256))
        steps_per_epoch = math.ceil(N / batch_size)

        # No lag concentration penalty by default — the skeleton already
        # determines lag structure.  Set lambda_lag_group > 0 to enforce
        # at most max_lags_per_pair active lags per pair.
        model_kwargs.setdefault("lambda_lag_group", 0.0)
        # Allow callers to override the computed lambda via model_kwargs
        lambda_override = model_kwargs.pop("lambda_l0", None)

        if device is None or device == "auto":
            accelerator = _detect_accelerator()
        else:
            accelerator = device

    def _fit_and_cut(skeleton):
        """Train the gated model on ``skeleton``; return gates, lambda and graph."""
        if ar_path:  # density over the gates the model trains: cross, lag-0 masked
            m = skeleton.shape[0]  # unless include_instantaneous
            lags = slice(0 if model_kwargs.get("include_instantaneous") else 1, None)
            trained = (np.asarray(skeleton) != 0)[:, :, lags] & ~np.eye(m, dtype=bool)[
                :, :, None
            ]
            n_cand = m * (m - 1) * trained.shape[2]
            density = int(trained.sum()) / n_cand if n_cand > 0 else 0.0
        else:
            density = int(skeleton.sum()) / n_possible if n_possible > 0 else 0.0
        # Select lambda: cross-validation or closed-form formula (Eq. 11, arXiv:2606.23880)
        if lambda_override is not None:
            lambda_l0, lambda_src = lambda_override, "override"
        elif lambda_cv:
            lambda_src = "CV"
            lambda_l0 = compute_lambda_cv(
                dataset,
                model_num_vars,
                max_lag,
                skeleton,
                model_kwargs=dict(model_kwargs, noise_model=noise_model),
                batch_size=batch_size,
                model_seed=model_seed,
                device=device,
                verbose=verbose,
            )
        else:
            lambda_src = "formula"
            lambda_l0 = compute_lambda(
                model_num_vars, len(df_model), density, rule=lambda_rule, xcorr=xcorr
            )
        if verbose:
            print(f"  lambda_L0={lambda_l0:.4f} ({lambda_src})")

        if model_seed is not None:
            from pytorch_lightning import seed_everything  # noqa: E402

            seed_everything(model_seed, workers=True)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = GatedCausalDiscovery(
            num_vars=model_num_vars,
            noise_model=noise_model,
            max_lag=max_lag,
            lambda_l0=float(lambda_l0),
            normalize_l0=True,
            steps_per_epoch=steps_per_epoch,
            **model_kwargs,
        )
        model.init_from_skeleton(skeleton)
        model.set_skeleton_mask(skeleton)

        _silence_lightning()
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            deterministic="warn" if model_seed is not None else False,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",
                    patience=patience,
                    mode="min",
                )
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(model, train_dataloaders=loader)

        _, gate_values = model.get_estimated_graph(threshold=None)
        # Strip C columns if include_C_in_model — keep only the original d variables
        if include_C_in_model:
            gate_values = gate_values[:num_vars, :num_vars, :]

        if ar_path:
            # Self-loops are not gated. Cross edges are cut at max(Li of the cross
            # gates, 0.5) under 'auto'/'li' (0.5 is the gate's decision boundary;
            # with the AR head, gates of edges without signal fall below it),
            # self-loops by a partial F-test given them.
            from .grow import self_loop_test

            n_gated = gate_values.shape[0]
            off_diag = ~np.eye(n_gated, dtype=bool)[:, :, None]
            cand = off_diag & (np.asarray(skeleton)[:n_gated, :n_gated] != 0)
            cross_g = np.where(cand, gate_values, 0.0)
            # As below: a supplied skeleton under 'auto' gets Li only when the
            # cross gates form two clusters, else the plain 0.5 cut.
            unimodal = (
                gate_threshold == "auto"
                and not own_skeleton
                and not gates_bimodal(
                    cross_g,
                    stretch_gamma=model_kwargs.get("stretch_gamma", -0.1),
                    stretch_zeta=model_kwargs.get("stretch_zeta", 1.1),
                )
            )
            if unimodal:  # at or above, as for 'auto_unimodal'
                cut, rule = 0.5, "ar_auto_unimodal"
                G_hat = ((cross_g >= cut) & cand).astype(np.int8)
            elif gate_threshold in ("auto", "li"):  # strictly above, as for Li
                c_li = li_threshold(cross_g)
                cut = max(0.5 if c_li is None else c_li, 0.5)
                rule = "ar_li_fallback" if c_li is None else "ar_li_floor"
                G_hat = ((cross_g > cut) & cand).astype(np.int8)
            else:  # at or above, as for the 'fixed' rule
                cut, rule = float(gate_threshold), "ar_fixed"
                G_hat = ((cross_g >= cut) & cand).astype(np.int8)
            if not model_kwargs.get("include_instantaneous"):
                G_hat[:, :, 0] = 0
            self_loops = self_loop_test(
                # Always on rank-Gaussianized data: the F-test is not robust to
                # heavy tails, and re-ranking data_transform="rank" output is a no-op.
                df.to_numpy(dtype=float),
                G_hat[:num_vars, :num_vars].astype(bool),
                max_lag,
                transform="rank",
                exog=c_tests,
                nuisance_basis=grow_nuisance_basis,
                test_basis=self_loop_basis,
            )
            G_hat = (G_hat.astype(bool) | self_loops).astype(np.int8)
            np.fill_diagonal(G_hat[:, :, 0], 0)
            return gate_values, lambda_l0, G_hat, cut, rule

        # Gates outside the skeleton are masked to 0, so Li sees only candidates.
        # 'auto': GRACE's own CDNOTS skeleton is loose by construction, so Li;
        # a supplied skeleton may already be precise (one cluster of open
        # gates), so Li only when the gates form two clusters, else 0.5.
        use_li = gate_threshold == "li" or (
            gate_threshold == "auto"
            and (
                own_skeleton
                or gates_bimodal(
                    gate_values,
                    stretch_gamma=model_kwargs.get("stretch_gamma", -0.1),
                    stretch_zeta=model_kwargs.get("stretch_zeta", 1.1),
                )
            )
        )
        if use_li:
            cut = li_threshold(gate_values)
            rule = "li" if cut is not None else "li_fallback"
            if cut is None:
                cut = 0.5
            G_hat = (gate_values > cut).astype(np.int8)
        elif gate_threshold == "auto":
            cut, rule = 0.5, "auto_unimodal"
            G_hat = (gate_values >= cut).astype(np.int8)
        else:
            cut = float(gate_threshold)
            rule = "fixed"
            G_hat = (gate_values >= cut).astype(np.int8)
        # Zero self-loops at lag 0
        np.fill_diagonal(G_hat[:, :, 0], 0)
        return gate_values, lambda_l0, G_hat, cut, rule

    # The C basis CDNOTS used conditions the grow tests, the AR path's self-loop
    # test and the regression-based shrink on GRACE's own skeleton. A supplied skeleton
    # never used C, unless include_C_in_model puts C in the model (in the gated
    # model, or in every shrink regression); then those tests condition on it too.
    use_c = (include_C and own_skeleton) or include_C_in_model
    c_exog = make_c_array(len(df), c_preset) if use_c else None
    # Under the gates the grow and self-loop tests take C through the spline
    # nuisance basis too: with data_transform="rank" a linear trend is curved on
    # the transformed scale, which a linear C column cannot remove, and a lagged
    # proxy of the trend then looks causal. The shrink keeps C linear, as in its
    # own regressions.
    # Four knots: a smooth function of time also absorbs the slow part of an
    # autoregression, so more flexibility costs self-loop power at small T (pure
    # AR(1), phi 0.5, T=100: 0.49 with linear C, 0.36 with 4 knots, 0.10 with 8)
    # for little extra trend protection (false self-loops on a ranked quadratic
    # trend, noise sd 0.1: 95% linear C, 4.4% with 4 knots, 1.9% with 8).
    c_tests = c_exog
    if c_exog is not None and refine == "gates" and grow_nuisance_basis == "spline":
        from .grow import _is_discrete, _level_indicators, spline_basis

        c_tests = np.column_stack(
            [
                _level_indicators(c) if _is_discrete(c) else spline_basis(c, n_knots=4)
                for c in np.asarray(c_exog, float).T
            ]
        )
    edge_pvalue = edge_strength = None

    def _refine(skeleton):
        """Refine ``skeleton``: gates (train + cut) or regression-based shrink."""
        nonlocal edge_pvalue, edge_strength
        if refine == "gates":
            return _fit_and_cut(skeleton)
        from .shrink import shrink_parents

        G, edge_pvalue, edge_strength = shrink_parents(
            X_refine if X_refine is not None else df.to_numpy(dtype=float),
            skeleton[:num_vars, :num_vars, :],
            max_lag,
            alpha=shrink_alpha,
            robust=shrink_robust,
            basis=shrink_basis,
            exog=c_exog,
            interactions=shrink_interactions,
        )
        return None, None, G.astype(np.int8), None, None

    if verbose:
        print(
            f"  Skeleton: {int(skeleton.sum())} edges, "
            f"density={int(skeleton.sum()) / max(n_possible, 1):.3f} "
            f"({t_skeleton:.1f}s)"
        )
        if refine == "gates":
            print("\nStep 2: Single gated training with skeleton mask...")
        else:
            print("\nStep 2: Regression-based shrink of the skeleton...")

    # ---- Steps 2-3: gated training then cutoff, or regression-based shrink ----
    gate_values, lambda_l0, G_hat, cut, threshold_rule = _refine(skeleton)
    t_training = _time.time() - t0 - t_skeleton
    if verbose:
        if refine == "gates":
            print(
                f"  Training done: {int(G_hat.sum())} edges kept at cutoff {cut:.3f} "
                f"[{threshold_rule}] ({t_training:.1f}s)"
            )
        else:
            print(f"  Shrink done: {int(G_hat.sum())} edges kept ({t_training:.1f}s)")

    # ---- Step 4 (optional): grow-shrink recall repair ----
    grow_history = []
    if grow_rounds > 0:
        from .grow import grow_candidates, grow_variance_candidates

        X_grow = X_refine if X_refine is not None else df.to_numpy(dtype=float)
        if np.isnan(X_grow).any():
            raise ValueError("grow_rounds > 0 needs data without missing values")
        # Condition the grow tests on the same C basis as the refinement (``c_exog``,
        # see ``use_c`` above), so series that share only a trend are not re-added.
        for rnd in range(1, grow_rounds + 1):
            sk_d = skeleton[:num_vars, :num_vars, :].astype(bool)
            if ar_path:  # self-loops are decided by the F-test, never grown
                sk_d = sk_d | np.eye(num_vars, dtype=bool)[:, :, None]
            add, n_tests = grow_candidates(
                X_grow,
                G_hat.astype(bool),
                sk_d,
                max_lag,
                grow_q,
                grow_transform,
                exog=c_tests,
                basis=shrink_basis if refine == "shrink" else "rff",
                interactions=refine == "shrink" and shrink_interactions,
                nuisance_basis=grow_nuisance_basis,
                effect_floor=effect_floor if refine == "gates" else 0.0,
            )
            n_tests_var = n_added_var = None
            if grow_variance:
                add_v, n_tests_var = grow_variance_candidates(
                    X_grow,
                    G_hat.astype(bool) | add,  # fresh mean parents enter the fit
                    sk_d | add,
                    max_lag,
                    grow_q,
                    "rank",
                    exog=c_tests,
                    nuisance_basis=grow_nuisance_basis,
                    effect_floor=effect_floor if refine == "gates" else 0.0,
                )
                n_added_var = int((add_v & ~add).sum())
                add = add | add_v
            n_added = int(add.sum())
            grow_history.append({"round": rnd, "n_tests": n_tests, "n_added": n_added})
            if n_tests_var is not None:  # a separate BH family, also at grow_q
                grow_history[-1]["n_tests_variance"] = n_tests_var
                grow_history[-1]["n_added_variance"] = n_added_var
            if verbose and n_tests_var is None:
                print(f"  Grow round {rnd}: {n_added} of {n_tests} tested edges added")
            elif verbose:
                print(
                    f"  Grow round {rnd}: {n_added - n_added_var} of {n_tests} mean "
                    f"tests and {n_added_var} of {n_tests_var} variance tests added"
                )
            if n_added == 0:
                break
            skeleton = skeleton.copy()
            block = skeleton[:num_vars, :num_vars, :]
            skeleton[:num_vars, :num_vars, :] = ((block != 0) | add).astype(
                skeleton.dtype
            )
            gate_values, lambda_l0, G_hat, cut, threshold_rule = _refine(skeleton)
            grow_history[-1]["n_kept"] = int((G_hat.astype(bool) & add).sum())
            if verbose and refine == "gates":
                print(
                    f"    retrained: {int(G_hat.sum())} edges at cutoff {cut:.3f} "
                    f"[{threshold_rule}], {grow_history[-1]['n_kept']} grown kept"
                )
            elif verbose:
                print(
                    f"    shrunk: {int(G_hat.sum())} edges, "
                    f"{grow_history[-1]['n_kept']} grown kept"
                )
        t_training = _time.time() - t0 - t_skeleton

    n_effect_pruned = 0
    effect_pruned = None
    if shrink_effect_floor is not None:
        from .grow import prune_effect_floor

        if shrink_effect_floor == "adaptive":
            floor = SHRINK_EFFECT_FLOOR_ADAPTIVE["floor"]
            relative = SHRINK_EFFECT_FLOOR_ADAPTIVE["relative"]
        else:
            floor, relative = float(shrink_effect_floor), 0.0
        # Before the lag-0 step, so CDNOTS+ pins the pruned lagged graph. Same blocks
        # and C conditioning as the shrink (which already rejected missing values).
        X_prune = X_refine if X_refine is not None else df.to_numpy(dtype=float)
        G_pruned = prune_effect_floor(
            X_prune,
            G_hat[:num_vars, :num_vars],
            max_lag,
            floor,
            exog=c_exog,
            basis=shrink_basis,
            nuisance_basis=grow_nuisance_basis,
            relative=relative,
        )
        effect_pruned = (np.asarray(G_hat[:num_vars, :num_vars]) != 0) & (G_pruned == 0)
        n_effect_pruned = int(effect_pruned.sum())
        G_hat = np.array(G_hat, copy=True)
        G_hat[:num_vars, :num_vars] = G_pruned
        if verbose:
            print(
                f"  Effect floor ({shrink_effect_floor}): {n_effect_pruned} edges pruned"
            )

    contemporaneous_result = None
    if contemporaneous:
        from .contemporaneous import contemporaneous_edges

        if verbose:
            print("Step 3: CDNOTS+ contemporaneous edges (lagged edges pinned)...")
        t_lag0 = _time.time()
        # The shrink's data and C conditioning: every column rank-transformed
        # under the spline basis (see X_refine; X_lag0 for the spline test under
        # shrink_basis="rff"), C exactly when the shrink used it.
        X0 = X_refine if X_refine is not None else X_lag0
        lag0_df = (
            df if X0 is None else pd.DataFrame(X0, index=df.index, columns=df.columns)
        )
        contemporaneous_result = contemporaneous_edges(
            lag0_df,
            G_hat,
            max_lag,
            ci_test=contemporaneous_test,
            alpha=contemporaneous_alpha,
            include_C=use_c,
            c_preset=c_preset,
            discrete_cols=discrete_cols,
            show_progress=verbose,
        )
        G_hat = np.array(G_hat, copy=True)
        G_hat[:, :, 0] = np.asarray(contemporaneous_result.cg_tig)[
            :num_vars, :num_vars, 0
        ]
        if verbose:
            print(
                f"  Lag-0: {int(G_hat[:, :, 0].sum())} oriented edges "
                f"({_time.time() - t_lag0:.1f}s)"
            )

    if refine == "gates" and effect_floor > 0:
        from .grow import prune_effect_floor

        X_prune = df.to_numpy(dtype=float)
        if np.isnan(X_prune).any():
            warnings.warn(
                "effect_floor: the data have missing values, so the final edges are "
                "not pruned (impute first, e.g. impute='var_em').",
                UserWarning,
                stacklevel=2,
            )
        else:
            G_pruned = prune_effect_floor(
                X_prune,
                G_hat[:num_vars, :num_vars],
                max_lag,
                effect_floor,
                exog=c_tests,
                nuisance_basis=grow_nuisance_basis,
            )
            effect_pruned = (np.asarray(G_hat[:num_vars, :num_vars]) != 0) & (
                G_pruned == 0
            )
            n_effect_pruned = int(effect_pruned.sum())
            G_hat = np.array(G_hat, copy=True)
            G_hat[:num_vars, :num_vars] = G_pruned
            if verbose:
                print(
                    f"  Effect floor {effect_floor:.4f}: {n_effect_pruned} edges pruned"
                )

    n_skeleton_edges = int(skeleton.sum())
    skeleton_density = n_skeleton_edges / n_possible if n_possible > 0 else 0.0

    total_runtime = _time.time() - t0
    n_edges = int(G_hat.sum())

    if verbose:
        if refine == "gates":
            print(
                f"\nResult: {n_edges} edges at threshold={cut:.3f} "
                f"({total_runtime:.1f}s total)"
            )
            print(f"  Skeleton: {t_skeleton:.1f}s, Training: {t_training:.1f}s")
        else:
            print(
                f"\nResult: {n_edges} edges after shrink"
                + (" and the lag-0 step" if contemporaneous else "")
                + f" ({total_runtime:.1f}s total)"
            )
            print(f"  Skeleton: {t_skeleton:.1f}s, Shrink: {t_training:.1f}s")

    if ground_truth is not None:
        metrics = evaluate_graph(G_hat, ground_truth)
        if verbose:
            print("\n--- Evaluation vs Ground Truth ---")
            print(
                f"  TP={metrics['TP']}  FP={metrics['FP']}  "
                f"FN={metrics['FN']}  TN={metrics['TN']}"
            )
            print(
                f"  TPR={metrics['TPR']:.3f}  FPR={metrics['FPR']:.3f}  "
                f"Precision={metrics['Precision']:.3f}  F1={metrics['F1']:.3f}"
            )
            print(f"  SHD={metrics['SHD']}  SHD_pair={metrics['SHD_pair']}")

    gated_or_floored = refine == "gates" or shrink_effect_floor is not None
    info = {
        # Under shrink the C rows/columns (include_C_in_model) were only
        # conditioning; report the skeleton over the d variables, like cg_tig.
        "skeleton": (
            skeleton[:num_vars, :num_vars, :] if refine == "shrink" else skeleton
        ),
        "grow_history": grow_history,
        # gates: the floor applied (0 = off); shrink: shrink_effect_floor (None = off)
        "effect_floor": effect_floor if refine == "gates" else shrink_effect_floor,
        "n_effect_pruned": n_effect_pruned,
        "skeleton_density": skeleton_density,
        "n_skeleton_edges": n_skeleton_edges,
        "lambda_l0": lambda_l0,
        "gate_threshold": cut,
        "gate_threshold_rule": threshold_rule,
        "runtime": total_runtime,
        "runtime_skeleton": t_skeleton,
        "runtime_training": t_training,
    }

    from .result import GraceResult

    return GraceResult(
        graph=G_hat,
        df=df_input,
        var_names=list(df.columns),
        gate_values=gate_values,
        skeleton=info["skeleton"],
        lambda_l0=info.get("lambda_l0"),
        gate_threshold=cut,
        gate_threshold_rule=threshold_rule,
        grow_history=grow_history if grow_rounds > 0 else None,
        effect_floor=info["effect_floor"],
        n_effect_pruned=info["n_effect_pruned"] if gated_or_floored else None,
        effect_pruned=effect_pruned,
        ar_path=bool(ar_path),
        lambda_rule=lambda_rule if refine == "gates" else None,
        runtime=info.get("runtime"),
        refine=refine,
        edge_pvalue=edge_pvalue,
        edge_strength=edge_strength,
        contemporaneous_result=contemporaneous_result,
    )
