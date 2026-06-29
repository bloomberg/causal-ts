# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib

_REGISTRY = {}

_LAZY_REGISTRY = {
    "parcorr-gpu": ("causalts.ci_tests.parcorr_gpu", "ParCorrGPU"),
    "kci": ("causalts.ci_tests.kci_gpu", "KCIGPU"),
    "rcot": ("causalts.ci_tests.pyRcot_gpu", "RCOTGPU"),
    "cmiknn": ("causalts.ci_tests.cmiknn", "CMIKNN"),
    "cmiknn-gpu": ("causalts.ci_tests.cmiknn_gpu", "CMIknnGPU"),
    "parcorr": ("causalts.ci_tests.parcorr", "PartialCorr"),
    "splitkci": ("causalts.ci_tests.splitkci_gpu", "SplitKCIGPU"),
    "dfcit": ("causalts.ci_tests.dfcit_gpu", "DFCITGPU"),
    "sigkci": ("causalts.ci_tests.sigkci_gpu", "SigKCIGPU"),
    "gcmi": ("causalts.ci_tests.gcmi_gpu", "GCMIGPU"),
    "cmiknn-mixed-gpu": ("causalts.ci_tests.cmiknn_mixed_gpu", "CMIknnMixedGPU"),
    "gsquared": ("causalts.ci_tests.causallearn_tests", "CausalLearnGsq"),
    "fisherz": ("causalts.ci_tests.causallearn_tests", "CausalLearnFisherZ"),
    "chisq": ("causalts.ci_tests.causallearn_tests", "CausalLearnChisq"),
}


def register_ci_test(name):
    """Decorator to register a custom CI test class by name.

    Example::

        from causalts.ci_tests import register_ci_test
        from causalts.ci_tests.cit_test import CIT_Base

        @register_ci_test("my_test")
        class MyCITest(CIT_Base):
            def __init__(self, data, **kwargs):
                super().__init__(data=data, method="my_test", **kwargs)

            def __call__(self, X, Y, condition_set=None):
                # Return (p_value, test_statistic)
                ...
    """

    def decorator(cls):
        _REGISTRY[name] = cls
        return cls

    return decorator


def create_ci_test(name, data, **kwargs):
    """Instantiate a CI test by its registered name.

    Parameters
    ----------
    name : str
        Registered name (e.g. ``"splitkci"``, ``"parcorr-gpu"``).
    data : np.ndarray
        Data array of shape ``(T, d)``.
    **kwargs
        Passed to the test constructor (e.g. ``device="cuda"``).

    Returns
    -------
    CIT_Base instance
    """
    if name in _REGISTRY:
        return _REGISTRY[name](data=data, **kwargs)
    if name in _LAZY_REGISTRY:
        mod_path, cls_name = _LAZY_REGISTRY[name]
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        _REGISTRY[name] = cls
        return cls(data=data, **kwargs)
    raise ValueError(f"Unknown CI test '{name}'. Available: {sorted(list_ci_tests())}")


def list_ci_tests():
    """Return sorted list of all registered CI test names."""
    return sorted(set(_REGISTRY.keys()) | set(_LAZY_REGISTRY.keys()))


_CLASS_TO_MODULE = {
    cls_name: mod_path for mod_path, cls_name in _LAZY_REGISTRY.values()
}


def __getattr__(name):
    """Lazy import CI test classes by name.

    Allows ``from causalts.ci_tests import SplitKCIGPU`` without eagerly
    loading torch or other heavy dependencies.
    """
    if name in _CLASS_TO_MODULE:
        mod = importlib.import_module(_CLASS_TO_MODULE[name])
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'causalts.ci_tests' has no attribute {name!r}")
