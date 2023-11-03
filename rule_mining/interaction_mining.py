import os
import pickle
import pandas as pd

from pandas import DataFrame
from mlxtend.frequent_patterns import apriori

from event_log_processing.event_log_manager import EventLogManager
from rule_mining.interaction_utils import ObjectTypeTransitionMultiplicity
from logics.utils import Disjunction, Conjunction, AtomicExpression

class InteractionMiner:
    '''
    Mining interaction (binding) patterns for object type interfaces in object-centric Petri nets.
    '''

    @classmethod
    def load(cls, name):
        path = os.getcwd()
        path = os.path.join(path, "rule_mining")
        path = os.path.join(path, name + ".pkl")
        with open(path, "rb") as rf:
            return pickle.load(rf)

    def __init__(self, name, activities, object_types, ocpn, event_log_manager: EventLogManager):
        """
        Create an object to conduct decision mining.

        :param name: The name of this DecisionMiner (for persistence purposes)
        :param activities: The labels of the transitions which form interaction points.
        :param object_types: The object types in the process.
        :param ocpn: The underlying object-centric Petri net.
        :param event_log_manager: For accessing the required data (frames).
        """
        self.name = name
        self.__activities = activities
        self.__object_types = object_types
        self.__ocpn = ocpn
        self.__ot2ot = self.__create_ot2ot_frame(event_log_manager)
        self.__event_frame = event_log_manager.get_event_frame()
        self.__o2o_frame = event_log_manager.get_o2o_frame()
        self.__e2o_frame = event_log_manager.get_e2o_frame()
        self.__object_evolutions_frame = event_log_manager.get_object_evolutions_frame()

    def __create_ot2ot_frame(self, event_log_manager: EventLogManager):
        o2o_frame = event_log_manager.get_o2o_frame()
        objects = event_log_manager.get_objects_frame()
        o2o_with_types = o2o_frame.merge(objects[["ocel:oid", "ocel:type"]], on="ocel:oid", how='inner') \
            .rename(columns={"ocel:oid": "ocel:oid_1", "ocel:type": "ocel:type_1"}) \
            .merge(objects[["ocel:oid", "ocel:type"]], left_on="ocel:oid_2", right_on="ocel:oid") \
            .rename(columns={"ocel:type": "ocel:type_2"}) \
            .drop("ocel:oid", axis=1)
        otype_relations_frame = o2o_with_types.groupby(['ocel:type_1', 'ocel:type_2'])['ocel:qualifier'].agg(
            lambda x: list(set(x))).reset_index()
        otype_relations = otype_relations_frame.groupby('ocel:type_1').apply(
            lambda x: dict(zip(x['ocel:type_2'], x['ocel:qualifier']))).to_dict()
        return otype_relations

    def save(self):
        path = os.getcwd()
        path = os.path.join(path, "rule_mining")
        path = os.path.join(path, self.name + ".pkl")
        with open(path, "wb") as wf:
            pickle.dump(self, wf)

    def get_interaction_base_table(self, activity, skip_many_to_many=False):
        df = self.__event_frame[self.__event_frame["ocel:activity"] == activity]
        df.drop("ocel:activity", axis=1)
        # extend with information about related objects
        df = pd.merge(df, self.__e2o_frame, on='ocel:eid', how='inner') \
            .drop("ocel:activity", axis=1) \
            .drop("ocel:qualifier", axis=1)
        # create rows corresponding to pairs of objects that interact
        df = pd.merge(df, df, on="ocel:eid", how="inner")
        df = df \
            .rename(columns={'ocel:timestamp_x': 'ocel:timestamp'}) \
            .drop("ocel:timestamp_y", axis=1)
        df = df \
            .merge(self.__o2o_frame, left_on=['ocel:oid_x', 'ocel:oid_y'], right_on=["ocel:oid", "ocel:oid_2"], how='left') \
            .drop(["ocel:oid", "ocel:oid_2"], axis=1)
        if skip_many_to_many:
            ### !!! strong assumption (start) !!!
            #   We do not want to inspect many-to-many interactions,
            #   i.e., relations between objects that both occur with variable multiplicity.
            ### !!! strong assumption (end) !!!
            df = df[df['ocel:type_x'].map(self.__ot_trans_multiplicities[activity]).
                        eq(ObjectTypeTransitionMultiplicity.SINGLE) |
                    df['ocel:type_y'].map(self.__ot_trans_multiplicities[activity])
                        .eq(ObjectTypeTransitionMultiplicity.SINGLE)]

        attributes = [col for col in self.__object_evolutions_frame.columns if col not in ["ocel:oid", "ocim:from", "ocim:to"]]
        df = df.merge(self.__object_evolutions_frame, left_on=["ocel:oid_x"], right_on=["ocel:oid"], how="inner")
        df = df[(df['ocel:timestamp'] >= df['ocim:from']) & (df['ocel:timestamp'] < df['ocim:to'])]
        df = df.rename(columns={key: key + "_x" for key in attributes})
        df = df.drop(["ocim:from", "ocim:to", "ocel:oid"], axis=1)
        df = df.merge(self.__object_evolutions_frame, left_on=["ocel:oid_y"], right_on=["ocel:oid"], how="inner")
        df = df[(df['ocel:timestamp'] >= df['ocim:from']) & (df['ocel:timestamp'] < df['ocim:to'])]
        df = df.rename(columns={key: key + "_y" for key in attributes})
        df = df.drop(["ocim:from", "ocim:to", "ocel:oid"], axis=1)
        return df

    def create_interaction_table(self, activity, skip_many_to_many=False):
        interaction_base_table = self.get_interaction_base_table(activity, skip_many_to_many)
        interaction_groups = interaction_base_table.groupby("ocel:eid")
        interaction_table = pd.DataFrame({'ocel:eid': interaction_groups.groups.keys()})
        ot_trans_mults_at_act = self.__ot_trans_multiplicities[activity]
        for ot1 in ot_trans_mults_at_act:
            for ot2 in ot_trans_mults_at_act:
                if ot1 == ot2:
                    continue
                ot_mult1 = ot_trans_mults_at_act[ot1]
                ot_mult2 = ot_trans_mults_at_act[ot2]
                if ot_mult1 is ObjectTypeTransitionMultiplicity.SINGLE and ot_mult2 is ObjectTypeTransitionMultiplicity.SINGLE:
                    self.__create_one_to_many_exists_interaction_features(ot1, ot2, interaction_groups, interaction_table)
                if ot_mult1 is ObjectTypeTransitionMultiplicity.SINGLE and ot_mult2 is ObjectTypeTransitionMultiplicity.VARIABLE:
                    self.__create_one_to_many_exists_interaction_features(ot1, ot2, interaction_groups, interaction_table)
                    self.__create_one_to_many_forall_interaction_features(ot1, ot2, interaction_groups, interaction_table)
                    self.__create_one_to_many_complete_interaction_features(ot1, ot2, interaction_groups, interaction_table)
        interaction_table.drop("ocel:eid", axis=1, inplace=True)
        return interaction_table

    def extract_ot_trans_multiplicities(self):
        ot_trans_multiplicities = {}
        ocpn = self.__ocpn
        e2o_frame = self.__e2o_frame
        event_frame = self.__event_frame
        act_otype_counts = e2o_frame.merge(event_frame, on="ocel:eid").\
            groupby(['ocel:eid', 'ocel:activity', 'ocel:type']).size().reset_index(name="count")
        act_otype_counts["exists"] = act_otype_counts.apply(lambda row: row["count"] > 0, axis=1)
        act_otype_counts["unique"] = act_otype_counts.apply(lambda row: row["count"] == 1, axis=1)
        for activity in self.__activities:
            ot_trans_multiplicities[activity] = {}
            total_events = len(event_frame[event_frame["ocel:activity"] == activity])
            for object_type in self.__object_types:
                unique_occurrences = act_otype_counts[
                    (act_otype_counts["ocel:activity"] == activity) &
                    (act_otype_counts["ocel:type"] == object_type)]["unique"].sum()
                is_single = float(unique_occurrences) / total_events > 0.98
                if is_single:
                    ot_trans_multiplicities[activity][object_type] = ObjectTypeTransitionMultiplicity.SINGLE
                    continue
                occurrences = act_otype_counts[
                    (act_otype_counts["ocel:activity"] == activity) &
                    (act_otype_counts["ocel:type"] == object_type)]["exists"].sum()
                is_variable = float(occurrences) / total_events > 0.98
                if is_variable:
                    ot_trans_multiplicities[activity][object_type] = ObjectTypeTransitionMultiplicity.VARIABLE
                else:
                    ot_trans_multiplicities[activity][object_type] = ObjectTypeTransitionMultiplicity.NONE
        self.__ot_trans_multiplicities = ot_trans_multiplicities

    def create_interaction_tables(self):
        interaction_tables = {}
        print("Creating interaction tables...")
        for activity in self.__activities:
            print("Creating interaction table for '{}'".format(activity))
            interaction_table = self.create_interaction_table(activity)
            interaction_tables[activity] = interaction_table
        self.__interaction_tables = interaction_tables

    def mine_interaction_patterns(self):
        interaction_patterns = {}
        print("Mining interaction patterns...")
        for activity in self.__activities:
            print("Mining interaction patterns for '{}'".format(activity))
            interaction_table = self.__interaction_tables[activity]
            interaction_patterns_act = apriori(interaction_table, min_support=0.05, use_colnames=True)
            interaction_patterns[activity] = interaction_patterns_act
        self.interaction_patterns = interaction_patterns

    def reduce_interaction_patterns(self, heuristics = None):
        reduced_interaction_patterns = {}
        for activity in self.__activities:
            interaction_patterns: DataFrame = self.interaction_patterns[activity]
            reduced_patterns = interaction_patterns.copy()
            reduced_patterns['to_delete'] = False
            for i, row in reduced_patterns.iterrows():
                if not row['to_delete']:
                    for j, other_row in reduced_patterns.iterrows():
                        if i != j and not other_row['to_delete']:
                            if set(row['itemsets']).issubset(other_row['itemsets']):
                                if other_row['support'] >= row['support']:
                                    reduced_patterns.at[i, 'to_delete'] = True
                                    break
            reduced_patterns = reduced_patterns[~reduced_patterns['to_delete']]
            reduced_patterns = reduced_patterns.drop(columns=['to_delete'])
            reduced_interaction_patterns[activity] = reduced_patterns
        self.reduced_interaction_patterns = reduced_interaction_patterns

    def transform_patterns_to_conditions(self):
        interaction_conditions = dict()
        for activity, interaction_patterns in self.reduced_interaction_patterns.items():
            outer_clauses = list(interaction_patterns["itemsets"].values)
            outer_condition = Disjunction()
            if len(outer_clauses) == 0:
                outer_condition.add_operand(AtomicExpression("True"))
            for outer_clause in outer_clauses:
                inner_clauses = list(outer_clause)
                inner_condition = Conjunction()
                for inner_clause in inner_clauses:
                    expression = AtomicExpression(inner_clause)
                    inner_condition.add_operand(expression)
                outer_condition.add_operand(inner_condition)
            interaction_conditions[activity] = outer_condition
        self.__interaction_conditions = interaction_conditions

    def run(self):
        self.extract_ot_trans_multiplicities()
        self.create_interaction_tables()
        self.mine_interaction_patterns()
        self.reduce_interaction_patterns()
        self.transform_patterns_to_conditions()

    def get_interaction_condition(self, activity):
        return self.__interaction_conditions[activity]

    ###
    # utils
    ###

    def __one_to_one_unique_condition(self, ot1, ot2, relation, group):
        ot1_unique = len(group["ocel:type_x"]) == len(group["ocel:oid_x"])
        ot2_unique = len(group["ocel:type_y"]) == len(group["ocel:oid_y"])
        nof_ot1_ot2_rel = ((group['ocel:qualifier'] == relation) & (group['ocel:type_x'] == ot1) & (group['ocel:type_y'] == ot2)).sum().astype(int)
        return ot1_unique and ot2_unique and nof_ot1_ot2_rel == 1

    def __one_to_many_exists_condition(self, ot1, ot2, relation, group):
        ot1_unique = len(group["ocel:type_x"]) == len(group["ocel:oid_x"])
        ot2_exists = (((group['ocel:qualifier'] == relation) & (group['ocel:type_x'] == ot1) & (group['ocel:type_y'] == ot2)).sum().astype(int) >= 1)
        return ot1_unique and ot2_exists

    def __one_to_many_forall_condition(self, ot1, ot2, relation, group):
        ot1_unique = len(group["ocel:type_x"]) == len(group["ocel:oid_x"])
        nof_ot2_event = ((group['ocel:type_x'] == ot1) & (group['ocel:type_y'] == ot2)).sum().astype(int)
        nof_ot2_rels_event = ((group['ocel:qualifier'] == relation) & (group['ocel:type_x'] == ot1) & (group['ocel:type_y'] == ot2)).sum().astype(int)
        return ot1_unique and (nof_ot2_event == nof_ot2_rels_event)

    def __one_to_many_complete_condition(self, ot1, ot2, relation, group):
        o2o_frame = self.__o2o_frame
        leading_object = group[group["ocel:type_x"] == ot1]["ocel:oid_x"].values[0]
        ot1_unique = len(group["ocel:type_x"]) == len(group["ocel:oid_x"])
        nof_ot2_rels_event = ((group['ocel:qualifier'] == relation) & (group['ocel:type_x'] == ot1) & (group['ocel:type_y'] == ot2)).sum().astype(int)
        nof_ot2_rels_total = len(o2o_frame[(o2o_frame["ocel:oid"] == leading_object) & (o2o_frame["ocel:qualifier"] == relation)])
        return ot1_unique and nof_ot2_rels_event == nof_ot2_rels_total

    def create_one_to_one_interaction_features(self, ot1, ot2, interaction_groups,  interaction_table):
        ot2ot = self.__ot2ot
        if ot1 not in ot2ot or ot2 not in ot2ot[ot1]:
            return
        relations = ot2ot[ot1][ot2]
        for relation in relations:
            filtered_groups = interaction_groups.filter(lambda group: self.__one_to_one_unique_condition(ot1, ot2, relation, group))
            retained_event_ids = set(filtered_groups['ocel:eid'].unique())
            interaction = relation + "(" + ot1 + "," + ot2 + ")_UNIQUE"
            interaction_table[interaction] = interaction_table['ocel:eid'].apply(lambda x: x in retained_event_ids)

    def __create_one_to_many_exists_interaction_features(self, ot1, ot2, interaction_groups, interaction_table):
        ot2ot = self.__ot2ot
        if ot1 not in ot2ot or ot2 not in ot2ot[ot1]:
            return
        relations = ot2ot[ot1][ot2]
        for relation in relations:
            filtered_groups = interaction_groups.filter(lambda group: self.__one_to_many_exists_condition(ot1, ot2, relation, group))
            retained_event_ids = set(filtered_groups['ocel:eid'].unique())
            interaction = relation + "(" + ot1 + "," + ot2 + ")_EXISTS"
            interaction_table[interaction] = interaction_table['ocel:eid'].apply(lambda x: x in retained_event_ids)

    def __create_one_to_many_forall_interaction_features(self, ot1, ot2, interaction_groups, interaction_table):
        ot2ot = self.__ot2ot
        if ot1 not in ot2ot or ot2 not in ot2ot[ot1]:
            return
        relations = ot2ot[ot1][ot2]
        for relation in relations:
            interaction = "(" + ot1 + "," + ot2 + ")_FORALL"
            filtered_groups = interaction_groups.filter(
                lambda group: self.__one_to_many_forall_condition(ot1, ot2, relation, group)
            )
            retained_event_ids = set(filtered_groups['ocel:eid'].unique())
            interaction = relation + "(" + ot1 + "," + ot2 + ")_FORALL"
            interaction_table[interaction] = interaction_table['ocel:eid'].apply(lambda x: x in retained_event_ids)

    def __create_one_to_many_complete_interaction_features(self, ot1, ot2, interaction_groups, interaction_table):
        ot2ot = self.__ot2ot
        if ot1 not in ot2ot or ot2 not in ot2ot[ot1]:
            return
        relations = ot2ot[ot1][ot2]
        for relation in relations:
            interaction = relation + "(" + ot1 + "," + ot2 + ")_COMPLETE"
            # TODO: elegant and safe condition check for uniqueness of leading type object
            filtered_groups = interaction_groups.filter(
                lambda group: self.__one_to_many_complete_condition(ot1, ot2, relation, group)
            )
            retained_event_ids = set(filtered_groups['ocel:eid'].unique())
            interaction_table[interaction] = interaction_table['ocel:eid'].apply(lambda x: x in retained_event_ids)