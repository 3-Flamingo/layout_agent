"""Dataset tooling for the learned evaluation dimension (reward model).

- perturb: graded perturbation operators (weak-supervised negative samples)
- preview: human-readable renders for annotation
- pairs:   preference-pair dataset IO (manifest.jsonl + states/renders/icm)
"""

from layout_agent.evalset.isolation import (
    colleague_challenge_views,
    forbidden_training_views,
    gold_views,
    human_pairs_test_views,
    human_pairs_train_views,
    write_forbidden_train_views,
)
from layout_agent.evalset.pairs import PairWriter, make_pair_record, split_for, view_key
from layout_agent.evalset.perturb import (
    LEVELS,
    measurement_signature,
    perturb_state,
    perturb_state_cmu_complex,
    perturb_state_dense,
)
from layout_agent.evalset.preview import render_preview

__all__ = [
    "PairWriter",
    "make_pair_record",
    "split_for",
    "view_key",
    "LEVELS",
    "perturb_state",
    "perturb_state_cmu_complex",
    "perturb_state_dense",
    "measurement_signature",
    "render_preview",
    "colleague_challenge_views",
    "forbidden_training_views",
    "gold_views",
    "human_pairs_test_views",
    "human_pairs_train_views",
    "write_forbidden_train_views",
]
