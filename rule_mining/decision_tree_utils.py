import math

import numpy as np
import pandas as pd

from pandas import Interval, DataFrame

from logics.utils import Conjunction, Disjunction, \
    Expression, BinaryExpression, BinaryOperator, VariableExpression, ValueExpression


class DecisionTreeNode:

    def __init__(self, instances, parent = None):
        self.parent = parent
        self.instances = instances
        self.children = []
        self.is_leaf = False

    def turn_to_leaf(self, target_feature):
        self.children = None
        total = len(self.instances)
        self.classes = set(self.instances[target_feature].values)
        self.weighted_classes = {
            target_class: float(len(self.instances[self.instances[target_feature] == target_class])) / total
            for target_class in self.classes
        }
        self.is_leaf = True

    def delete_training_data(self):
        self.instances = None
        if self.children is not None:
            for arc in self.children:
                arc.child.delete_training_data()

    def set_split_feature(self, split_feature):
        self.split_feature = split_feature


class DecisionTreeArc:

    def __init__(self, parent, child, condition: Expression):
        self.parent = parent
        self.child = child
        self.condition = condition


class DecisionTreeLeaf:

    def __init__(self, instances, classes, weighted_classes):
        self.instances = instances,
        self.classes = classes
        self.weighted_classes = weighted_classes


class DecisionTree:

    def __init__(self, min_information_gain: float = 0.0):
        self.min_information_gain = min_information_gain

    def fit(self, instances, feature_names, feature_types, target_class):
        self.target_class = target_class
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.root = DecisionTreeNode(instances)
        self.target_class_values = instances[target_class].values
        self.expand(self.root)
        self.collect_garbage()

    def get_entropy(self, array):
        unique_elements, counts = np.unique(array, return_counts=True)
        total_count = len(array)
        relative_frequencies = counts / total_count
        entropy = -sum(map(lambda rf: rf*math.log(rf)/math.log(2), relative_frequencies))
        return entropy

    def expand(self, node):
        instances = node.instances
        target_labels = instances[self.target_class].values
        if len(set(target_labels)) < 2:
            node.turn_to_leaf(self.target_class)
        entropy = self.get_entropy(target_labels)
        total = len(instances)
        feature_split_entropies = {}
        numerical_features_best_thresholds = {}
        for feature in self.feature_names:
            feature_type = self.feature_types[feature]
            is_numerical = feature_type == float or feature_type == int
            if is_numerical:
                best_threshold, entropy = self.get_numerical_feature_best_threshold(feature, instances)
                numerical_features_best_thresholds[feature] = best_threshold
                feature_split_entropies[feature] = entropy
            else:
                feature_values = set(instances[feature].values)
                feature_split_entropy = 0
                for value in feature_values:
                    value_share = instances[instances[feature] == value]
                    value_share_entropy = self.get_entropy(value_share[self.target_class].values)
                    value_share_total = len(value_share)
                    feature_split_entropy = feature_split_entropy + value_share_entropy * value_share_total / total
                feature_split_entropies[feature] = feature_split_entropy
        best_split_feature = min(feature_split_entropies, key=feature_split_entropies.get)
        information_gain = entropy - feature_split_entropies[best_split_feature]
        if information_gain < self.min_information_gain:
            node.turn_to_leaf(self.target_class)
        else:
            split_feature_type = self.feature_types[best_split_feature]
            threshold = None
            if split_feature_type  == float or split_feature_type == int:
                threshold = numerical_features_best_thresholds[best_split_feature]
            self.split_node(node, best_split_feature, threshold)

    def split_node(self, node, split_feature, threshold = None):
        node.set_split_feature(split_feature)
        instances = node.instances
        feature_values = set(instances[split_feature].values)
        if threshold is None:
            for value in feature_values:
                value_share = instances[instances[split_feature] == value]
                child = DecisionTreeNode(value_share, node)
                variable_expression = AtomicExpression(split_feature)
                value_expression = AtomicExpression(value)
                operator = BinaryOperator.EQUALS
                condition = BinaryExpression(variable_expression, value_expression, operator)
                arc = DecisionTreeArc(node, child, condition)
                node.children.append(arc)
                self.expand(child)
        else:
            left_value_share = instances[instances[split_feature] <= threshold]
            right_value_share = instances[instances[split_feature] > threshold]
            child_a = DecisionTreeNode(left_value_share, node)
            child_b = DecisionTreeNode(right_value_share, node)
            variable_expression = VariableExpression(split_feature)
            value_expression = ValueExpression(threshold)
            operator_a = BinaryOperator.SMALLER_EQUALS
            operator_b = BinaryOperator.GREATER
            condition_a = BinaryExpression(variable_expression, value_expression, operator_a)
            condition_b = BinaryExpression(variable_expression, value_expression, operator_b)
            arc_a = DecisionTreeArc(node, child_a, condition_a)
            arc_b = DecisionTreeArc(node, child_b, condition_b)
            node.children.append(arc_a)
            node.children.append(arc_b)
            self.expand(child_a)
            self.expand(child_b)

    def get_numerical_feature_best_threshold(self, feature, instances: DataFrame):
        df = instances.copy()
        df.sort_values(feature, ascending=True)
        df['target_class_shifted'] = df[self.target_class].shift(1)
        df['feature_shifted'] = df[feature].shift(1)
        df["target_class_changes"] = ~(df[self.target_class] == df["target_class_shifted"])
        df.at[0, "target_class_changes"] = False
        df["candidate_thresholds"] = df.apply(
            lambda row: (row[feature] + row["feature_shifted"]) / 2 if row["target_class_changes"] else None,
            axis=1)
        candidate_thresholds = set(df["candidate_thresholds"].dropna().values)
        best_threshold = None
        entropy = None
        for th in candidate_thresholds:
            left_value_share = df[df[feature] <= th]
            right_value_share = df[df[feature] > th]
            left_value_share_entropy = self.get_entropy(left_value_share[self.target_class].values)
            right_value_share_entropy = self.get_entropy(right_value_share[self.target_class].values)
            n = len(left_value_share)
            m = len(right_value_share)
            th_entropy = (n*left_value_share_entropy + m*right_value_share_entropy) / (n+m)
            if entropy is None or th_entropy < entropy:
                entropy = th_entropy
                best_threshold = th
        return best_threshold, entropy

    def collect_garbage(self):
        self.root.delete_training_data()

    def reverse_function(self, target_class, min_weight):
        node = self.root
        inner_conjunction = Conjunction()
        outer_disjunction = Disjunction()
        self.__construct_reverse_function_recursive(outer_disjunction, node, inner_conjunction, target_class, min_weight)
        return outer_disjunction

    def __construct_reverse_function_recursive(self, outer_disjunction: Disjunction, node: DecisionTreeNode,
                                               inner_conjunction: Conjunction, target_class, min_weight):
        if node.is_leaf:
            if target_class in node.classes:
                if node.weighted_classes[target_class] >= min_weight:
                    outer_disjunction.add_operand(inner_conjunction)
            return
        arc: DecisionTreeArc
        for arc in node.children:
            new_inner_conjunction = Conjunction(
                operands= inner_conjunction.operands
            )
            new_inner_conjunction.add_operand(arc.condition)
            child = arc.child
            self.__construct_reverse_function_recursive(outer_disjunction, child, new_inner_conjunction, target_class, min_weight)


