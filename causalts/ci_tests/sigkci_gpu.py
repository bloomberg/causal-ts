# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deprecated: ``SigKCIGPU`` was renamed to :class:`LinSigCIGPU`.

The test always used a linear static kernel with a closed-form recursion, not
the RBF signature-kernel PDE of Manten et al. (2024), so the old name overstated
it. Import :class:`causalts.ci_tests.linsig_gpu.LinSigCIGPU` instead.
"""

import warnings

from .linsig_gpu import LinSigCIGPU, linsig_cind, linsig_uind

_DEPRECATED = (
    "SigKCIGPU / 'sigkci' is deprecated and will be removed in a future "
    "release; use LinSigCIGPU / 'linsig'."
)


def _warn_lamda(lamda):
    if lamda is not None and lamda != 0.001:
        warnings.warn(
            f"lamda={lamda!r} is ignored: LinSig conditions with a ridge scaled "
            "to the kernel, set with ridge= (default 1e-3).",
            UserWarning,
            stacklevel=3,
        )


def sigkci_uind(
    x,
    y,
    sigma="median",
    dyadic_order=0,
    path_transform="time_delay",
    time_delay_dim=3,
    approx="gamma",
    nperms=200,
):
    """Deprecated alias for :func:`linsig_uind`."""
    warnings.warn(_DEPRECATED, DeprecationWarning, stacklevel=2)
    return linsig_uind(
        x, y, sigma, dyadic_order, path_transform, time_delay_dim, approx, nperms
    )


def sigkci_cind(
    x,
    y,
    z,
    sigma="median",
    dyadic_order=0,
    path_transform="time_delay",
    time_delay_dim=3,
    approx="gamma",
    nperms=200,
    lamda=0.001,
):
    """Deprecated alias for :func:`linsig_cind`; ``lamda`` is ignored."""
    warnings.warn(_DEPRECATED, DeprecationWarning, stacklevel=2)
    _warn_lamda(lamda)
    return linsig_cind(
        x, y, z, sigma, dyadic_order, path_transform, time_delay_dim, approx, nperms
    )


class SigKCIGPU(LinSigCIGPU):
    """Deprecated alias for :class:`LinSigCIGPU`; will be removed."""

    def __init__(
        self,
        data,
        device=None,
        sigma="median",
        dyadic_order=0,
        path_transform="time_delay",
        time_delay_dim=3,
        approx="hbe",
        nperms=200,
        lamda=0.001,
        normalize=True,
        seed=42,
        **kwargs,
    ):
        # The 0.27.0 signature, so positional calls keep their meaning; forward
        # by keyword because LinSig's slot after ``nperms`` is ``ridge``.
        warnings.warn(_DEPRECATED, DeprecationWarning, stacklevel=2)
        # lamda: the old absolute ridge, replaced by LinSig's kernel-scaled ridge.
        _warn_lamda(lamda)
        super().__init__(
            data,
            device=device,
            sigma=sigma,
            dyadic_order=dyadic_order,
            path_transform=path_transform,
            time_delay_dim=time_delay_dim,
            approx=approx,
            nperms=nperms,
            normalize=normalize,
            seed=seed,
            **kwargs,
        )

    def __setstate__(self, state):
        # An object pickled by 0.27.0 has ``lamda`` and no ``ridge``. Give it
        # LinSig's default ridge and cache identity, and drop p-values the old
        # conditioning computed so they are not served as LinSig results.
        self.__dict__.update(state)
        if "ridge" not in state:
            _warn_lamda(state.get("lamda"))
            self.ridge = 1e-3
            self.__dict__.pop("lamda", None)
            self.param_hash = f"{self.param_hash}-rr{self.ridge}"
            self.pvalue_cache = {}
            self.pvalue_local = {}
