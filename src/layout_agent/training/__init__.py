"""Training framework for the learned evaluation dimension (reward model).

Modules:
- data:    pair dataset from evalset manifests; ICM -> tensor encoding
- model:   ResNet18 backbone -> scalar quality logit (sigmoid*100 for display)
- losses:  Bradley-Terry pairwise ranking loss (+ grade regression hook)
- metrics: spearman / pairwise agreement (same protocol as the formula baseline)
- train:   DDP training loop (torchrun / 8×A800) with early stop on val agreement
- eval:    checkpoint evaluation on labeled T1/T2
- score:   single-layout quality (formula + residual)
- export:  ONNX export + onnxruntime parity check
- env:     LayoutEnv (Executor+Evaluator) for the later decision-RL phase
"""
