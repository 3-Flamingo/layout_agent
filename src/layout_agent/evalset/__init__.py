"""Dataset tooling for the learned evaluation dimension (reward model).

- perturb: graded perturbation operators (weak-supervised negative samples)
- preview: human-readable renders for annotation
- pairs:   preference-pair dataset IO (manifest.jsonl + states/renders/icm)
"""

from layout_agent.evalset.pairs import PairWriter, make_pair_record, split_for, view_key
from layout_agent.evalset.perturb import LEVELS, perturb_state
from layout_agent.evalset.preview import render_preview

__all__ = [
    "PairWriter",
    "make_pair_record",
    "split_for",
    "view_key",
    "LEVELS",
    "perturb_state",
    "render_preview",
]
