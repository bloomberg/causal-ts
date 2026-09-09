# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exposes causal-ts's GPU CI tests through causal-learn's CIT registry.

Importing this module registers every causal-ts GPU CI test with
``causallearn.utils.cit.register_ci_test``, so they become selectable by
name from causal-learn's own factory and algorithms::

    import causalts.ci_tests.causal_learn_bridge  # registers on import
    from causallearn.utils.cit import CIT
    from causallearn.search.ConstraintBased.PC import pc

    cit = CIT(data, method="parcorr_gpu")
    g = pc(data, indep_test=cit)

causal-learn's ``CIT_Base.__call__`` contract returns a single p-value,
while causal-ts CI tests return ``(p_value, statistic)``; the adapter below
unwraps that tuple.

``CMIknnMixedGPU`` and ``StratifiedCIT`` are deliberately excluded: both
have required constructor arguments (``discrete_cols``, and for
``StratifiedCIT`` also a pre-built ``inner_cit``) with no defaults, so they
cannot be selected by bare name the way the tests below are — a caller would
still need to pass those through ``CIT(..., **kwargs)`` by hand, at which
point there is no benefit over instantiating them directly. ``StratifiedCIT``
is also documented as an internal wrapper not meant for factory selection.
"""

from .cmiknn_gpu import CMIknnGPU
from .dfcit_gpu import DFCITGPU
from .gcmi_gpu import GCMIGPU
from .kci_gpu import KCIGPU
from .parcorr_gpu import ParCorrGPU
from .pyRcot_gpu import RCOTGPU
from .sigkci_gpu import SigKCIGPU
from .splitkci_gpu import SplitKCIGPU

_REGISTRY = {
    "parcorr_gpu": ParCorrGPU,
    "kci_gpu": KCIGPU,
    "splitkci_gpu": SplitKCIGPU,
    "dfcit_gpu": DFCITGPU,
    "rcot_gpu": RCOTGPU,
    "cmiknn_gpu": CMIknnGPU,
    "gcmi_gpu": GCMIGPU,
    "sigkci_gpu": SigKCIGPU,
}


def _make_adapter(name, causalts_cls):
    from causallearn.utils.cit import CIT_Base as CLCITBase

    class _Adapter(CLCITBase):
        def __init__(self, data, cache_path=None, **kwargs):
            super().__init__(data, cache_path=cache_path)
            self.method = name
            self._inner = causalts_cls(data=data, **kwargs)

        def __call__(self, X, Y, condition_set=None):
            pval, _stat = self._inner(X, Y, condition_set)
            return float(pval)

    _Adapter.__name__ = f"CausalTS_{causalts_cls.__name__}"
    _Adapter.__qualname__ = _Adapter.__name__
    return _Adapter


def register_all():
    """Register every causal-ts GPU CI test with causal-learn's CIT factory.

    Safe to call more than once; later calls just re-register the same
    classes under the same names.
    """
    from causallearn.utils.cit import register_ci_test

    for name, cls in _REGISTRY.items():
        register_ci_test(name, _make_adapter(name, cls))


register_all()
