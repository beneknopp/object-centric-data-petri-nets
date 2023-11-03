import os
import pickle
import pandas as pd
from pandas import DataFrame
from pm4py.objects.petri_net.obj import PetriNet
from sklearn.tree import DecisionTreeClassifier
from p_decision_tree.DecisionTree import DecisionTree
from event_log_processing.event_log_manager import EventLogManager
from rule_mining.decision_tree_utils import DecisionTree
from logics.utils import Conjunction, Disjunction, AtomicExpression


class DecisionMiner:
    '''
    Mining guards (rules) for transitions in an object-centric Petri net.
    '''

    @classmethod
    def load(cls, name):
        path = os.getcwd()
        path = os.path.join(path, "rule_mining")
        path = os.path.join(path, name + ".pkl")
        with open(path, "rb") as rf:
            return pickle.load(rf)

    def __init__(self, name, object_types, petri_nets, alignments, event_log_manager: EventLogManager):
        """
        Create an object to conduct decision mining.

        :param name: The name of this DecisionMiner (for persistence purposes)
        :param object_types: The object types found in the object-centric Petri net.
        :param petri_nets: A dictionary storing for each object type the corresponding simple Petri net.
        :param alignments: A list of CaseAlignments storing for each object type the alignments of trace variants found in the
        passed enriched event frame (event_objects_attributes_frame).
        :param event_log_manager: For accessing the required data (frames).

        """
        self.decision_trees = None
        if not set(object_types) == set(petri_nets.keys()):
            raise Exception("Petri nets do not describe the passed object types")
        if not set(object_types) == set(alignments.keys()):
            raise Exception("Alignments do not describe the passed object types")
        self.name = name
        self.__object_types = object_types
        self.__petri_nets = petri_nets
        self.__alignments = alignments
        self.__event_log_manager = event_log_manager
        self.__attributes = list(
            filter(lambda x: x.startswith("ocim:attr:"),
                   self.__event_log_manager.get_event_object_attributes_frame().columns)
        )

    def save(self):
        path = os.getcwd()
        path = os.path.join(path, "rule_mining")
        path = os.path.join(path, self.name + ".pkl")
        with open(path, "wb") as wf:
            pickle.dump(self, wf)

    def create_analytics_base_tables(self):
        """
        Enriches the alignments with information about the object attributes at each state.
        """
        analytics_base_tables = {}
        decision_pairs = {}
        objects_frame = self.__event_log_manager.get_objects_frame()
        event_objects_attributes_frame: DataFrame = self.__event_log_manager.get_event_object_attributes_frame()
        for object_type in self.__object_types:
            ot_alignments = self.__alignments[object_type]
            exploded_ot_alignments = map(lambda alm:
                    list(map(lambda move: [alm.case_id, move[0][1], move[1][0]], alm.alignment)),
                    ot_alignments)
            exploded_ot_alignments = [move for case_list in exploded_ot_alignments for move in case_list]
            ot_alignments_frame = pd.DataFrame(exploded_ot_alignments,
                columns= ["ocel:oid", "transition_name", "event_activity"]
            )
            ot_event_objects = event_objects_attributes_frame[event_objects_attributes_frame["ocel:type"] == object_type]
            ot_event_objects["has_event"] = 1
            ot_event_objects["event_counter"] = ot_event_objects.groupby('ocel:oid')['has_event'].cumsum()
            ot_event_objects["event_counter"] = ot_event_objects.apply(lambda row: row["event_counter"] - row["has_event"], axis=1)
            ot_event_objects.drop("has_event", axis=1, inplace=True)
            ot_alignments_frame["has_event"] = ot_alignments_frame.apply(lambda row: 0 if row["event_activity"] == ">>" else 1, axis=1)
            ot_alignments_frame['event_counter'] = ot_alignments_frame.groupby('ocel:oid')['has_event'].cumsum()
            ot_alignments_frame["event_counter"] = ot_alignments_frame.apply(
                lambda row: row["event_counter"] - row["has_event"], axis=1)
            ot_alignments_frame.drop("has_event", axis=1, inplace=True)
            ot_enriched_alignments = ot_event_objects.merge(ot_alignments_frame, on = ["ocel:oid", "event_counter"])
            ot_enriched_alignments.drop(["event_counter"], axis= 1, inplace=True)
            ###
            # Associate moves with decisions at places
            ###
            ot_petri_net = self.__petri_nets[object_type]
            arcs = ot_petri_net.arcs
            decision_pairs_ot = pd.DataFrame(list(map(
                lambda arc: [arc.source.name, arc.target.name],
                    filter(lambda arc: type(arc.target) == PetriNet.Transition and len(arc.source.out_arcs) > 1, arcs))),
                columns = ["place_id", "transition_name"]
            )
            decision_pairs[object_type] = decision_pairs_ot
            analytics_base_table = ot_enriched_alignments.merge(decision_pairs_ot, on=["transition_name"])
            analytics_base_tables[object_type] = analytics_base_table
        self.__decision_pairs = decision_pairs
        self.__analytics_base_tables = analytics_base_tables

    def build_decision_trees(self):
        decision_trees = {}
        for object_type in self.__object_types:
            analytics_base_table = self.__analytics_base_tables[object_type]
            decision_point_groups = analytics_base_table.groupby("place_id")
            decision_trees_ot = {}
            for place_id, group in decision_point_groups:
                features = self.__attributes
                # TODO: Robustness for missing / unclean data
                features = list(group[features].dropna(axis=1).columns)
                column_types = group.dtypes.to_dict()
                feature_types = {f: column_types[f] for f in column_types if f in features}
                decision_tree = DecisionTree(min_information_gain=0.01)
                target_class = "transition_name"
                decision_tree.fit(group, features, feature_types, target_class)
                decision_trees_ot[place_id] = decision_tree
            decision_trees[object_type] = decision_trees_ot
        self.decision_trees = decision_trees

    def mine_decision_trees(self):
        self.create_analytics_base_tables()
        self.build_decision_trees()

    def get_decision_rule(self, object_type, transition_name, min_weight=0.05):
        decision_trees = self.decision_trees[object_type]
        decision_pairs = self.__decision_pairs[object_type]
        transition_preset = set(decision_pairs[decision_pairs["transition_name"] == transition_name]["place_id"] \
                                .values)
        condition = Conjunction()
        for place_id in transition_preset:
            if place_id not in decision_trees:
                condition = AtomicExpression("False")
                break
            decision_tree: DecisionTree = decision_trees[place_id]
            disjunction: Disjunction = decision_tree.reverse_function(transition_name, min_weight)
            condition.add_operand(disjunction)
        return condition