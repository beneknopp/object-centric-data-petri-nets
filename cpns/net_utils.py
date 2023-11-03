from pm4py.objects.petri_net.obj import PetriNet

from rule_mining.guard_utils import FlatEstimator, TransitionGuard


class Place:

    def __init__(self, place: PetriNet.Place, function_estimator: FlatEstimator = None, color = None):
        self.place = place
        self.function_estimator = function_estimator
        self.color = color


class Transition:

    def __init__(self, transition: PetriNet.Transition, guard: TransitionGuard):
        self.transition = transition
        self.guard = guard