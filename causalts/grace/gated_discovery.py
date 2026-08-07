# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Gated Causal Discovery with Stability Selection.

Replaces softmax attention with independent Hard Concrete gates (Louizos et al. 2018)
and uses Stability Selection (Meinshausen & Bühlmann 2010) across L0 penalty strengths
to determine the causal graph without requiring a manual threshold.
"""

import logging
import math
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


def compute_lambda(d: int, T: int, skeleton_density: float) -> float:
    """Auto-select L0 penalty lambda from problem dimensions and skeleton density.

    Formula (Eq. 11 from arXiv:2507.07898)::

        max(0.007 + 0.16*rho_S,  1/d + 4/T - 0.4*rho_S^0.8)

    The main term adapts to problem scale; the floor ensures sufficient
    regularization for dense skeletons (Lorenz-96, Erdos-Renyi topologies).

    Parameters
    ----------
    d : int
        Number of variables.
    T : int
        Number of time steps.
    skeleton_density : float
        Fraction of non-zero edges in the skeleton.

    Returns
    -------
    float
        Selected lambda value.
    """
    rho = skeleton_density
    floor = 0.007 + 0.16 * rho
    main = 1.0 / d + 4.0 / T - 0.4 * rho**0.8
    return max(floor, main)


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
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["max_lags_per_pair"])
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

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
        self, threshold: Optional[float] = 0.5, absval: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Return (G_bin, W) in [cause, effect, lag] with lag 0..max_lag.

        Same output format as existing models in attention_discovery.py.

        Parameters
        ----------
        threshold : float or None
            If None, G_bin is None.
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
        Include time index as a variable.
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

    # Add time index if requested
    if include_C:
        T = df_input.shape[0]
        C = pd.DataFrame(np.arange(T).reshape(-1, 1), columns=["C"])
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
    **model_kwargs : dict
        Passed to GatedCausalDiscovery.

    Returns
    -------
    GraceResult
        Result object with attributes ``cg_tig`` (binary graph),
        ``stability_scores`` (edge frequencies in [0, 1]), and
        inherited plotting / DoWhy bridge methods.
    """
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
    gate_threshold: float = 0.5,
    normalize: bool = True,
    max_epochs: int = 150,
    batch_size: Optional[int] = None,
    patience: int = 20,
    device: Optional[str] = None,
    verbose: bool = True,
    ground_truth: Optional[np.ndarray] = None,
    model_seed: Optional[int] = None,
    include_C_in_model: bool = False,
    lambda_cv: bool = False,
    impute: Optional[str] = None,
    impute_kwargs: Optional[dict] = None,
    discrete_cols=None,
    **model_kwargs,
) -> "GraceResult":
    """CDNOTS skeleton + single gated refinement (no stability selection).

    1. Run CDNOTS to get a high-recall binary skeleton.
    2. Train single gated model with skeleton mask to get continuous
       gate values.
    3. Threshold gate values to produce the final graph.

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
        (d, d, max_lag+1). If provided, skips CDNOTS and uses this
        skeleton directly.
    skeleton_runtime : float
        Wall-clock time of the pre-computed skeleton (for info dict).
    gate_threshold : float
        Threshold on gate values for final graph.
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
    include_C_in_model : bool
        If True, append a time-index column C to the data before
        training the gated model (mirrors CDNOTS nonstationarity
        handling). C-related edges are stripped from the output graph.
    lambda_cv : bool
        If True, select lambda via cross-validation instead of the
        closed-form formula (Eq. 11, arXiv:2507.07898). Slower but
        data-driven.
    impute : str or None
        Missing-value strategy applied before discovery.
        ``None`` — no imputation. ``'var_em'`` — iterative VAR-EM.
    impute_kwargs : dict or None
        Extra keyword arguments forwarded to the imputer.
    discrete_cols : list or None
        Column names or indices of discrete variables.
    **model_kwargs : dict
        Forwarded to GatedCausalDiscovery.

    Returns
    -------
    GraceResult
        Result object with attributes ``cg_tig`` (binary graph),
        ``gate_values`` (continuous gate values), ``skeleton``,
        ``lambda_l0``, ``gate_threshold``, ``runtime``, and inherited
        plotting / DoWhy bridge methods.
    """
    import time as _time
    import warnings

    from ..cdnots.phase3_utils import run_cdnots

    t0 = _time.time()
    num_vars = df.shape[1]

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

    # ---- Step 1: CDNOTS Skeleton ----
    if skeleton is not None:
        t_skeleton = skeleton_runtime
        if verbose:
            print("Step 1: Using pre-computed skeleton")
    else:
        if verbose:
            print("Step 1: CDNOTS skeleton discovery...")

        if ci_test_class is None:
            ci_test_class = _get_parcorr_class()

        ci_test = ci_test_class(np.ascontiguousarray(df.values))
        cdnots_res = run_cdnots(
            df=df,
            indep_test=ci_test,
            num_lags=max_lag,
            include_C=True,
            alpha=alpha,
            stable=True,
            show_progress=verbose,
            discrete_cols=discrete_cols,
        )

        skel_vars = num_vars + 1 if include_C_in_model else num_vars
        skeleton = _cdnots_graph_to_binary(cdnots_res.cg_tig, skel_vars, max_lag)
        t_skeleton = _time.time() - t0

    # Optionally append a time-index column C to the model data
    if include_C_in_model:
        T_len = len(df)
        C_col = pd.DataFrame(
            np.arange(T_len, dtype=np.float64).reshape(-1, 1), columns=["C"]
        )
        df_model = pd.concat([df.reset_index(drop=True), C_col], axis=1)
        model_num_vars = num_vars + 1
    else:
        df_model = df
        model_num_vars = num_vars

    n_skeleton_edges = int(skeleton.sum())
    n_possible = model_num_vars * model_num_vars * (max_lag + 1) - model_num_vars
    skeleton_density = n_skeleton_edges / n_possible if n_possible > 0 else 0.0

    if verbose:
        print(
            f"  Skeleton: {n_skeleton_edges} edges, "
            f"density={skeleton_density:.3f} ({t_skeleton:.1f}s)"
        )

    # ---- Step 2: Single Gated Training ----
    if verbose:
        print("\nStep 2: Single gated training with skeleton mask...")

    # Prepare data (before lambda selection — CV needs the dataset)
    dataset = prepare_data(df_model, max_lag, normalize=normalize)
    N = len(dataset)

    # Select lambda: cross-validation or closed-form formula (Eq. 11, arXiv:2507.07898)
    if lambda_cv:
        lambda_l0 = compute_lambda_cv(
            dataset,
            model_num_vars,
            max_lag,
            skeleton,
            model_kwargs=dict(model_kwargs),
            batch_size=batch_size,
            model_seed=model_seed,
            device=device,
            verbose=verbose,
        )
    else:
        lambda_l0 = compute_lambda(model_num_vars, len(df_model), skeleton_density)

    if batch_size is None:
        batch_size = max(32, min(N // 8, 256))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    steps_per_epoch = math.ceil(N / batch_size)

    if model_seed is not None:
        from pytorch_lightning import seed_everything  # noqa: E402

        seed_everything(model_seed, workers=True)

    # No lag concentration penalty by default — the skeleton already
    # determines lag structure.  Set lambda_lag_group > 0 to enforce
    # at most max_lags_per_pair active lags per pair.
    model_kwargs.setdefault("lambda_lag_group", 0.0)
    # Allow callers to override the computed lambda via model_kwargs
    lambda_src = "CV" if lambda_cv else "formula"
    if "lambda_l0" in model_kwargs:
        lambda_l0 = model_kwargs.pop("lambda_l0")
        lambda_src = "override"

    if verbose:
        print(f"  lambda_L0={lambda_l0:.4f} ({lambda_src})")

    model = GatedCausalDiscovery(
        num_vars=model_num_vars,
        max_lag=max_lag,
        lambda_l0=float(lambda_l0),
        normalize_l0=True,
        steps_per_epoch=steps_per_epoch,
        **model_kwargs,
    )
    model.init_from_skeleton(skeleton)
    model.set_skeleton_mask(skeleton)

    # Train
    if device is None or device == "auto":
        accelerator = _detect_accelerator()
    else:
        accelerator = device

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

    # Extract gate values
    _, gate_values = model.get_estimated_graph(threshold=None)
    t_training = _time.time() - t0 - t_skeleton

    # Strip C column if include_C_in_model — return only the original d variables
    if include_C_in_model:
        gate_values = gate_values[:num_vars, :num_vars, :]

    if verbose:
        n_open = int((gate_values > gate_threshold).sum())
        print(f"  Training done: {n_open} gates > {gate_threshold} ({t_training:.1f}s)")

    # ---- Step 3: Threshold ----
    G_hat = (gate_values >= gate_threshold).astype(np.int8)
    # Zero self-loops at lag 0
    np.fill_diagonal(G_hat[:, :, 0], 0)

    total_runtime = _time.time() - t0
    n_edges = int(G_hat.sum())

    if verbose:
        print(
            f"\nResult: {n_edges} edges at threshold={gate_threshold} "
            f"({total_runtime:.1f}s total)"
        )
        print(f"  Skeleton: {t_skeleton:.1f}s, Training: {t_training:.1f}s")

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

    info = {
        "skeleton": skeleton,
        "skeleton_density": skeleton_density,
        "n_skeleton_edges": n_skeleton_edges,
        "lambda_l0": lambda_l0,
        "gate_threshold": gate_threshold,
        "runtime": total_runtime,
        "runtime_skeleton": t_skeleton,
        "runtime_training": t_training,
    }

    from .result import GraceResult

    return GraceResult(
        graph=G_hat,
        df=df,
        var_names=list(df.columns),
        gate_values=gate_values,
        skeleton=info["skeleton"],
        lambda_l0=info.get("lambda_l0"),
        gate_threshold=gate_threshold,
        runtime=info.get("runtime"),
    )
