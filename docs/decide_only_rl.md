# Decide-only RL pipeline

The learned policy replaces only `DecisionEngine`. `Executor`, `Evaluator`, and
the deterministic rule `Schedule` remain active safety layers.

## 1. Freeze human gold

```powershell
python scripts/evalset_human_gold.py freeze
python scripts/evalset_human_gold.py validate `
  --t1-labels <t1 exports...> --t2-labels <t2 exports...>
```

This freezes 15 T1 tasks and 80 T2 pairs under
`data/evalsys/human_gold/`. The larger annotation packages are not rewritten.
Conflicting duplicate annotations fail validation and require adjudication.
Val pairs without a live official URL are placed in the separate
`data/evalsys/annotate_t2_gold_supplement/` package; colleague challenge pairs
are never used to fill the gold gate.

## 2. Validate the simulator

```powershell
python scripts/rl_validate_env.py `
  --states "data/datasets/*/snapshots/*.state.json" `
  --episodes 20 --out runs/rl_env_validation.json
```

`LayoutEnv` uses Gymnasium's `(observation, info)` reset and
`(observation, reward, terminated, truncated, info)` step APIs. Observations
contain the 3-channel ICM, global features, padded candidate features, and an
action mask. `legal_bbox` is immutable; only `render_bbox` may expand.

For simulator-to-drawing-2d checks, pass `--real-replays` with JSONL rows:

```json
{"initial_state":"before.state.json","actions":[{"type":"SetLinearSide","target_id":"L1","side":"R"}],"expected_state":"drawing2d-after.state.json"}
```

## 3. Collect experts and train behavior cloning

```powershell
python -m layout_agent.training.collect `
  --states "data/datasets/*/snapshots/*.state.json" `
  --out runs/rl_trajectories

python -m layout_agent.training.train_bc `
  --trajectories runs/rl_trajectories `
  --out runs/rl_bc

python -m layout_agent.training.rollout_eval `
  --states "data/datasets/*/snapshots/*.state.json" `
  --checkpoint runs/rl_bc/best.pt `
  --out runs/rl_bc/rollout_gate.json
```

Collection mixes clean/light/medium/heavy starts and rules/greedy/beam/random
rollouts. Splitting happens by `view_key` before perturbation. Frozen human and
colleague-gold views are excluded. Random trajectories are stored as a
baseline but excluded from BC by default.

The rollout gate compares hard-clear rate, final formula `P`, and steps against
both rules and greedy. PPO refuses a missing or failed gate.

## 4. Train masked PPO

```powershell
python -m layout_agent.training.train_ppo `
  --states "data/datasets/*/snapshots/*.state.json" `
  --bc-checkpoint runs/rl_bc/best.pt `
  --bc-gate runs/rl_bc/rollout_gate.json `
  --out runs/rl_ppo
```

Sampling uses spawned subprocesses by default. Reports include action-family
coverage, no-op/rollback rates, per-step time, hard-clear rate, and final `P`.

## 5. Gate and ablate the human reward

First evaluate the reward model and formula on the frozen human labels, then:

```powershell
python scripts/rl_build_human_reward_gate.py `
  --model-report runs/reward/human_eval.json `
  --formula-report runs/formula/human_eval.json `
  --reward-checkpoint runs/reward/best.pt `
  --out runs/reward/human_gate.json

python scripts/rl_reward_ablation.py `
  --states "data/datasets/*/snapshots/*.state.json" `
  --bc-checkpoint runs/rl_bc/best.pt `
  --bc-gate runs/rl_bc/rollout_gate.json `
  --reward-checkpoint runs/reward/best.pt `
  --reward-gate runs/reward/human_gate.json `
  --out runs/rl_reward_ablation
```

The learned term is `beta * (Q(next) - Q(current))`, applies only when both
states are hard-clean, and cannot offset any hard violation. Default ablations
use beta 0, 0.01, and 0.03.
