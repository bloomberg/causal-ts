# Data Generation API

## Loading Built-in Datasets

```{eval-rst}
.. autofunction:: causalts.synthetic_data.synthetic_datasets.load_dataset
```

## Parametric Generator

```{eval-rst}
.. autoclass:: causalts.synthetic_data.synthetic_datasets.SCPGraphGenerator
   :members:
   :show-inheritance:
```

## Built-in Datasets

```{eval-rst}
.. autofunction:: causalts.synthetic_data.synthetic_datasets.ex1_nl

.. autofunction:: causalts.synthetic_data.synthetic_datasets.ex2

.. autofunction:: causalts.synthetic_data.synthetic_datasets.ex3

.. autofunction:: causalts.synthetic_data.synthetic_datasets.henon_chain

.. autofunction:: causalts.synthetic_data.synthetic_datasets.ex1_nonstationary

.. autofunction:: causalts.synthetic_data.synthetic_datasets.confounded_trends

.. autofunction:: causalts.synthetic_data.synthetic_datasets.lorenz96
```

## Topology Generators

```{eval-rst}
.. autofunction:: causalts.synthetic_data.synthetic_datasets.topology_scp

.. autofunction:: causalts.synthetic_data.synthetic_datasets.scale_free

.. autofunction:: causalts.synthetic_data.synthetic_datasets.erdos_renyi

.. autofunction:: causalts.synthetic_data.synthetic_datasets.small_world
```

## Real-world Benchmark and Confounding

```{eval-rst}
.. autofunction:: causalts.synthetic_data.ercot_weather

.. autofunction:: causalts.synthetic_data.apply_confounding
```

## Mixed Datasets

```{eval-rst}
.. autofunction:: causalts.synthetic_data.mixed_datasets.mixed_a

.. autofunction:: causalts.synthetic_data.mixed_datasets.mixed_b

.. autofunction:: causalts.synthetic_data.mixed_datasets.mixed_c
```

## Utilities

```{eval-rst}
.. autofunction:: causalts.utils.linearity.check_linearity

.. autofunction:: causalts.imputation.var_em_impute

.. autofunction:: causalts.imputation.iterative_causal_impute

.. autofunction:: causalts.utils.heteroskedasticity.check_variance_dependence

.. autofunction:: causalts.utils.subsampling.detect_subsampling

.. autoclass:: causalts.utils.subsampling.DetectionResult
   :members:
   :undoc-members:
```
