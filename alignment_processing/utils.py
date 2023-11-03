from typing import Union

class CaseAlignment:
    '''
    The alignment of a case, consisting of the case_id and the alignment.
    '''

    def __init__(self, case_id, alignment):
        """
        Construct a CaseAlignment.

        :param case_id: The case id.
        :param alignment: The alignment.

        """
        self.case_id = case_id
        self.alignment = alignment

    def is_model_move_at_index(self, index):
        return self.alignment[index][1][1] is None

    def get_transition_name_at_index(self, index):
        return self.alignment[index][0][1]


class EnrichedMove:
    '''
    An alignment move also storing information about relations that this object has and object attributes
    '''

    def __init__(self, transition_name: Union[str, None], event_id, relations, attributes):
        """
        Construct an EnrichedMove.

        :param transition_name: The name (identifier) of the transition, if synchronous move or model move.
        :param event_id: The event id, if synchronous move or log move.
        :param relations: (Qualified) object-to-object relationships of the object (at time of the state).
        :param attributes: Object attribute value assignments of the object at time of the state.

        """
        self.transition_name = transition_name
        self.event_id = event_id
        self.relations = relations
        self.attributes = attributes

class EnrichedCaseAlignment:
    '''
    EnrichedAlignments per Case.
    '''

    def __init__(self, case_id, enriched_moves):
        """
        Construct an EnrichedCaseAlignment.

        :param case_id: The case (object) id to which the alignment belongs.
        :param enriched_moves: A list of EnrichedMoves.

        """
        self.case_id = case_id
        self.enriched_moves = enriched_moves
