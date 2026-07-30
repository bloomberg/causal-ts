# Causal-TS

[![Release](https://img.shields.io/github/v/release/bloomberg/causal-ts?display_name=tag)](CHANGELOG.md)
[![Documentation](https://readthedocs.org/projects/causal-ts/badge/?version=latest)](https://causal-ts.readthedocs.io/en/latest/)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/bloomberg/causal-ts/badge)](https://scorecard.dev/viewer/?uri=github.com/bloomberg/causal-ts)
[![Lint](https://github.com/bloomberg/causal-ts/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/marketplace/actions/super-linter)
[![Tests](https://github.com/bloomberg/causal-ts/actions/workflows/test.yml/badge.svg)](https://github.com/bloomberg/causal-ts/actions/workflows/test.yml)
[![Contributor-Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-fbab2c.svg)](CODE_OF_CONDUCT.md)

## About the Project

📚 **[Full documentation is on Read the Docs](https://causal-ts.readthedocs.io/en/latest/)** — algorithm deep-dives, CI test guide, CLI reference, and worked example notebooks.

Causal-TS is a Python framework for causal discovery in time series data. It implements four discovery algorithms and eight GPU-accelerated conditional independence tests, along with built-in visualization and evaluation tools.

**Algorithms:**

- **CD-NOTS** — Constraint-based discovery handling nonstationarity via a time-index node.
- **CD-NOTS+** — PCMCI+-style two-phase skeleton (MCI conditioning) for improved precision on dense graphs.
- **CEDAR** — Scalable pairwise discovery using minimum-lag selection. O(d²) complexity.
- **GRACE** — Hybrid: CD-NOTS skeleton + neural gated refinement with L0 regularization for high-dimensional data.

**CI Tests:** Run `causal-ts ci-test-info` for a full selection guide.

| Test | Type | Speed |
|------|------|-------|
| `parcorr-gpu` | Linear | instant |
| `gcmi` | Monotone nonlinear | instant |
| `splitkci` | Nonlinear (kernel) | fast |
| `rcot` | Nonlinear (RFF) | fast |
| `linsig` | Path-space (signature) | moderate |
| `kci` | Nonlinear (kernel) | slow |
| `dfcit` | Distribution-free | moderate |
| `cmiknn-gpu` | Nonparametric (k-NN) | slow |


## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/bloomberg/causal-ts.git
   cd causal-ts
   ```

2. Install dependencies:

   ```bash
   pip install -e .
   ```

   PyTorch is installed automatically. CUDA and Apple MPS are auto-detected at runtime; CPU is the fallback.

## Quick Start

```python
import numpy as np
from causalts.synthetic_data.synthetic_datasets import load_dataset
from causalts.ci_tests import SplitKCIGPU
from causalts import run_cdnots
from causalts.utils import evaluate_graph
from causalts.plotting import compare_graphs

# 1. Load a built-in dataset (ex1: 5-var nonlinear)
data = load_dataset("ex1", seed=42, T=500)
df, ground_truth = data["df"], data["ground_truth"]

# 2. Run CD-NOTS causal discovery
ci_test = SplitKCIGPU(np.zeros((2, 2)), device="cpu")
res = run_cdnots(
    df=df, indep_test=ci_test, num_lags=data["max_lag"],
    include_C=True, alpha=0.05, stable=True,
)

# 3. Evaluate (exclude C dimension for shape match with ground truth)
d = ground_truth.shape[0]
metrics = evaluate_graph(res.cg_tig[:d, :d, :], ground_truth)
print(f"F1={metrics['F1']:.3f}, SHD={metrics['SHD']}")

# 4. Visualize
res.plot()
compare_graphs(ground_truth, res.cg_tig[:d, :d, :], var_names=list(df.columns))
```

## Roadmap

See the [open issues](https://github.com/bloomberg/causal-ts/issues) for a list
of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to
learn, inspire, and create. Any contributions you make are **greatly
appreciated**. For detailed contributing guidelines, please see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Distributed under the `GPL-3.0-or-later` License. See [LICENSE](LICENSE) for
more information.

This project includes modified code from third-party packages. See
[NOTICE](NOTICE) for details on original authorship and licensing.

## Contact

Mohammad Fesanghary - [@fesanghary](https://twitter.com/fesanghary)

Project Link:
[https://github.com/bloomberg/causal-ts](https://github.com/bloomberg/causal-ts)

## Acknowledgements

We thank the open source contributors to Tigramite and causal-learn whose implementations informed parts of Causal-TS’s design. This template was adapted from [Best-README-Template](https://github.com/othneildrew/Best-README-Template).

## Citations

If you use causal-ts in your research, please cite the following papers:

1. **Causal-TS: A Python Library for Causal Discovery in High-Dimensional and Nonstationary Time Series**
   Mohammad Fesanghary
   arXiv preprint [arXiv:2607.24673](https://arxiv.org/abs/2607.24673), 2026.

2. **CEDAR: Causal Edge Discovery for Autoregressive Processes**
   Mohammad Fesanghary, Achintya Gopal
   arXiv preprint [arXiv:2607.20696](https://arxiv.org/abs/2607.20696), 2026.

3. **GRACE: Gated Refinement for Accurate Causal Edge Discovery in High-Dimensional Time Series**
   Mohammad Fesanghary, Abhinav Havaldar
   arXiv preprint [arXiv:2606.23880](https://arxiv.org/abs/2606.23880), 2026.

4. **Causal Discovery from Nonstationary Time Series**
   Agathe Sadeghi, Achintya Gopal, Mohammad Fesanghary
   *International Journal of Data Science and Analytics*, 19, pp. 33–59, 2025.
   [doi:10.1007/s41060-024-00679-7](https://doi.org/10.1007/s41060-024-00679-7)

<details>
<summary>BibTeX entries</summary>

```bibtex
@article{fesanghary2026causalts,
  title={Causal-TS: A Python Library for Causal Discovery in High-Dimensional and Nonstationary Time Series},
  author={Fesanghary, Mohammad},
  journal={arXiv preprint arXiv:2607.24673},
  year={2026}
}

@article{fesanghary2026cedar,
  title={CEDAR: Causal Edge Discovery for Autoregressive Processes},
  author={Fesanghary, Mohammad and Gopal, Achintya},
  journal={arXiv preprint arXiv:2607.20696},
  year={2026}
}

@article{fesanghary2026grace,
  title={GRACE: Gated Refinement for Accurate Causal Edge Discovery in High-Dimensional Time Series},
  author={Fesanghary, Mohammad and Havaldar, Abhinav},
  journal={arXiv preprint arXiv:2606.23880},
  year={2026}
}

@article{sadeghi2025cdnots,
  title={Causal Discovery from Nonstationary Time Series},
  author={Sadeghi, Agathe and Gopal, Achintya and Fesanghary, Mohammad},
  journal={International Journal of Data Science and Analytics},
  volume={19},
  pages={33--59},
  year={2025},
  publisher={Springer},
  doi={10.1007/s41060-024-00679-7}
}
```

</details>
