# CI Tests API

Class reference for every registered test. For choosing a test, see the
[CI Test Selection Guide](../ci_tests.md). Tests can also be created by name:

```{eval-rst}
.. autofunction:: causalts.ci_tests.create_ci_test

.. autofunction:: causalts.ci_tests.list_ci_tests

.. autofunction:: causalts.ci_tests.register_ci_test
```

## ParCorrGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.parcorr_gpu.ParCorrGPU
   :members:
   :show-inheritance:
```

## GCMIGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.gcmi_gpu.GCMIGPU
   :members:
   :show-inheritance:
```

## SplitKCIGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.splitkci_gpu.SplitKCIGPU
   :members:
   :show-inheritance:
```

## KCIGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.kci_gpu.KCIGPU
   :members:
   :show-inheritance:
```

## DFCITGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.dfcit_gpu.DFCITGPU
   :members:
   :show-inheritance:
```

## RCOTGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.pyRcot_gpu.RCOTGPU
   :members:
   :show-inheritance:
```

## CMIknnGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.cmiknn_gpu.CMIknnGPU
   :members:
   :show-inheritance:
```

## LinSigCIGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.linsig_gpu.LinSigCIGPU
   :members:
   :show-inheritance:
```

## StratifiedCIT

```{eval-rst}
.. autoclass:: causalts.ci_tests.stratified_cit.StratifiedCIT
   :members:
   :show-inheritance:
```

## CMIknnMixedGPU

```{eval-rst}
.. autoclass:: causalts.ci_tests.cmiknn_mixed_gpu.CMIknnMixedGPU
   :members:
   :show-inheritance:
```

## GCM

```{eval-rst}
.. autoclass:: causalts.ci_tests.gcm.GCM
   :members:
   :show-inheritance:
```

## WGCM

```{eval-rst}
.. autoclass:: causalts.ci_tests.wgcm.WGCM
   :members:
   :show-inheritance:
```

## PCM

```{eval-rst}
.. autoclass:: causalts.ci_tests.pcm.PCM
   :members:
   :show-inheritance:
```

## RAFT / RIFT

```{eval-rst}
.. autoclass:: causalts.ci_tests.raft.RAFTCI
   :members:
   :show-inheritance:

.. autoclass:: causalts.ci_tests.raft.RIFTCI
   :show-inheritance:
```

## CPU implementations

```{eval-rst}
.. autoclass:: causalts.ci_tests.parcorr.PartialCorr
   :members:
   :show-inheritance:

.. autoclass:: causalts.ci_tests.cmiknn.CMIKNN
   :members:
   :show-inheritance:
```

## causal-learn adapters

```{eval-rst}
.. autoclass:: causalts.ci_tests.causallearn_tests.CausalLearnFisherZ
   :show-inheritance:

.. autoclass:: causalts.ci_tests.causallearn_tests.CausalLearnChisq
   :show-inheritance:

.. autoclass:: causalts.ci_tests.causallearn_tests.CausalLearnGsq
   :show-inheritance:
```

`SigKCIGPU` (registry name `sigkci`) is a deprecated alias of `LinSigCIGPU`.

