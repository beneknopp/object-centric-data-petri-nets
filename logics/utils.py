from enum import Enum


class Operator(Enum):

    def print_sml(self):
        raise Exception("Abstract parent class method called")


class UnaryOperator(Operator):
    NOT = "NOT"

    def print_sml(self):
        return "not"


class BinaryOperator(Operator):

    SMALLER = "SMALLER"
    SMALLER_EQUALS = "SMALLER_EQUALS"
    GREATER = "GREATER"
    GREATER_EQUALS = "GREATER_EQUALS"
    EQUALS = "EQUALS"

    def print_sml(self):
        if self == BinaryOperator.SMALLER:
            return "<"
        if self == BinaryOperator.SMALLER_EQUALS:
            return "<="
        if self == BinaryOperator.GREATER:
            return ">"
        if self == BinaryOperator.GREATER_EQUALS:
            return ">="
        if self == BinaryOperator.EQUALS:
            return "="


class NAryOperator(Operator):

    AND = "AND"
    OR = "OR"

    def print_sml(self):
        if self == NAryOperator.AND:
            return "andalso"
        if self == NAryOperator.OR:
            return "orelse"


class Expression:

    def __init__(self):
        pass

    def print_sml(self):
        raise Exception("Abstract parent class method called")


class AtomicExpression(Expression):

    def __init__(self, value):
        self.value = value

    def print_sml(self):
        return str(self.value)


class UnaryExpression(Expression):

    def __init__(self, operand, operator: UnaryOperator):
        super().__init__()
        self.operand = operand
        self.operator = operator

    def print_sml(self):
        operator_str = self.operator.print_sml()
        operand_str = self.operand.print_sml()
        if operand_str.startswith("(") and operand_str.endswith(")"):
            operand_str = operand_str[1:-1]
        return "{} ({})".format(operator_str, operand_str)


class VariableExpression(AtomicExpression):

    def __init__(self, value):
        super().__init__(value)


class ValueExpression(AtomicExpression):

    def __init__(self, value):
        super().__init__(value)


class BinaryExpression(Expression):

    def __init__(self, variable: VariableExpression, value: ValueExpression, operator: Operator):
        super().__init__()
        self.variable = variable
        self.value = value
        self.operator = operator

    def print_sml(self):
        operand_a_str = self.variable.print_sml()
        operator_str = self.operator.print_sml()
        operand_b_str = self.value.print_sml()
        if operand_a_str.startswith("(") and operand_a_str.endswith(")"):
            operand_a_str = operand_a_str[1:-1]
        if operand_b_str.startswith("(") and operand_b_str.endswith(")"):
            operand_b_str = operand_b_str[1:-1]
        return "({}) {} ({})".format(operand_a_str, operator_str, operand_b_str)

    def subsumes(self, other_expression):
        operator = self.operator
        other_operator = other_expression.operator
        value = self.value.value
        other_value = other_expression.value.value
        # self: x > 3 subsumes x > 4
        if operator is BinaryOperator.GREATER:
            if other_operator is BinaryOperator.GREATER:
                return value <= other_value
            if other_operator is BinaryOperator.GREATER_EQUALS:
                return value < other_value
            return False
        if operator is BinaryOperator.GREATER_EQUALS:
            if other_operator is BinaryOperator.GREATER_EQUALS:
                return value <= other_value
            if other_operator is BinaryOperator.GREATER:
                return value < other_value
            return False
        if operator is BinaryOperator.SMALLER:
            # self: x < 2 subsumes x < 1
            if other_operator is BinaryOperator.SMALLER:
                return value >= other_value
            if other_operator is BinaryOperator.SMALLER_EQUALS:
                return value > other_value
            return False
        if operator is BinaryOperator.SMALLER_EQUALS:
            if other_operator is BinaryOperator.SMALLER_EQUALS:
                return value >= other_value
            if other_operator is BinaryOperator.SMALLER:
                return value > other_value
            return False

    def contradicts(self, other_expression):
        operator = self.operator
        other_operator = other_expression.operator
        value = self.value.value
        other_value = other_expression.value.value
        # self: x > 3 contradicts x < 3 and x <= 2 but not x <= 3
        if operator is BinaryOperator.GREATER:
            if other_operator is BinaryOperator.SMALLER:
                return value >= other_value
            if other_operator is BinaryOperator.SMALLER_EQUALS:
                return value > other_value
            return False
        if operator is BinaryOperator.GREATER_EQUALS:
            # self: x >= 3 contradicts x < 3 and x <= 2 but not x <= 3
            if other_operator is BinaryOperator.SMALLER_EQUALS:
                return value > other_value
            if other_operator is BinaryOperator.SMALLER:
                return value >= other_value
            return False
        if operator is BinaryOperator.SMALLER:
            # self: x < 2 contradicts x > 2 and x >= 3 but not x >= 2
            if other_operator is BinaryOperator.GREATER:
                return value <= other_value
            if other_operator is BinaryOperator.GREATER_EQUALS:
                return value < other_value
            return False
        if operator is BinaryOperator.SMALLER_EQUALS:
            # self: x <= 2 contradicts x > 3 and x > 2 but not x >= 2
            if other_operator is BinaryOperator.GREATER_EQUALS:
                return value > other_value
            if other_operator is BinaryOperator.GREATER:
                return value <= other_value
            return False


class NAryExpression(Expression):

    def __init__(self, operator: NAryOperator, operands=None):
        super().__init__()
        if operands is None:
            operands = []
        self.operands = operands
        self.operator = operator

    def add_operand(self, operand):
        self.operands.append(operand)

    def print_sml(self):
        print_str = ""
        operator_str = self.operator.print_sml()
        operand: Expression
        n = len(self.operands)
        for idx in range(n):
            operand = self.operands[idx]
            operand_str = operand.print_sml()
            if operand_str.startswith("(") and operand_str.endswith(")"):
                operand_str = operand_str[1:-1]
            print_str += " (" + operand_str + ")"
            if idx < n - 1:
                print_str += " " + operator_str
        return print_str


class Conjunction(NAryExpression):

    def __init__(self, operands=None):
        super().__init__(NAryOperator.AND, operands)

    def print_sml(self):
        if len(self.operands) == 0:
            return "True"
        else:
            return super().print_sml()

    def add_operand(self, operand: Expression):
        # we often need to deal with binary comparisons (e.g. for continuous features)
        if isinstance(operand, BinaryExpression):
            operand: BinaryExpression
            operands = [operand] + self.operands[:]
            reduction = True
            while reduction:
                reduction, operands = self.reduce(operands)
            self.operands = operands
        else:
            self.operands.append(operand)

    def reduce(self, operands):
        first_operand = operands[0]
        reduction = False
        for idx in range(len(operands[1:])):
            other_operand = operands[1:][idx]
            if isinstance(other_operand, BinaryExpression):
                other_operand: BinaryExpression
                if first_operand.variable.value == other_operand.variable.value:
                    if first_operand.subsumes(other_operand):
                        reduction = True
                        reduced_operand = first_operand
                        operands = [reduced_operand] + operands[1:1+idx] + operands[2+idx:]
                        break
                    if other_operand.subsumes(first_operand):
                        reduction = True
                        reduced_operand = other_operand
                        operands = [reduced_operand] + operands[1:1+idx] + operands[2+idx:]
        return reduction, operands
class Disjunction(NAryExpression):

    def __init__(self, operands=None):
        super().__init__(NAryOperator.OR, operands)

    def print_sml(self):
        if len(self.operands) == 0:
            return "False"
        else:
            return super().print_sml()