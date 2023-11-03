from typing import Dict
from logics.utils import Disjunction


class FlatEstimator:

    def __init__(self, transition_condition_map: Dict[str, Disjunction]):
        self.transition_condition_map = transition_condition_map


class InteractionEstimator:

    def __init__(self, condition: Disjunction):
        self.condition = condition


class TransitionGuard:

    def __init__(self, flat_estimators: Dict[str, FlatEstimator], interaction_estimator: InteractionEstimator):
        self.flat_estimators = flat_estimators
        self.interaction_estimator = interaction_estimator