from collections import Counter
from copy import copy

import numpy as np


class Operator(object):

    _props = [None, None, None]

    def __repr__(self):
        string = "{c:}({s:}, {u:})"
        string = string.format(
            c=self.__class__.__name__, s=str(self.symbol), u=str(self.ufunc))
        return string

    def __str__(self):
        return self.symbol

    @property
    def identifier(self):
        return self._props[0]

    @property
    def symbol(self):
        return self._props[0].strip()

    @property
    def ufunc(self):
        return self._props[1]

    @property
    def rank(self):
        return self._props[2]


class OperatorR(Operator):

    @property
    def nargs(self):
        return 1


class OperatorLR(Operator):

    @property
    def nargs(self):
        return 2


# List of supported operators. Each operator has an asociated symbol that
# represents it in standard python, a numpy ufunc that is applied to the
# number 'nargs' of operands, and a rank specifying the precedence of the
# operator compared to others.
operator_list = []
for identifier, ufunc, rank, nargs in [
        # (identifier, ufunc, rank, nargs)
        # arithmetic operators
        ("**",    np.power,         43, 2),
        ("-",     np.negative,      42, 1),
        ("*",     np.multiply,      41, 2),
        ("/",     np.divide,        41, 2),
        ("%",     np.mod,           41, 2),
        ("+",     np.add,           40, 2),
        ("-",     np.subtract,      40, 2),
        # bitwise operators
        ("~",     np.bitwise_not,   31, 1),
        ("&",     np.bitwise_and,   30, 2),
        ("|",     np.bitwise_or,    30, 2),
        ("^",     np.bitwise_xor,   30, 2),
        # comparators
        ("==",    np.equal,         20, 2),
        ("!=",    np.not_equal,     20, 2),
        ("<",     np.less,          20, 2),
        (">",     np.greater,       20, 2),
        ("<=",    np.less_equal,    20, 2),
        (">=",    np.greater_equal, 20, 2),
        # logial operators
        ("NOT ",  np.logical_not,   11, 1),
        (" AND ", np.logical_and,   10, 2),
        (" OR ",  np.logical_or,    10, 2),
        (" XOR ", np.logical_xor,   10, 2)]:
    operator = OperatorR() if nargs == 1 else OperatorLR()
    operator._props = (identifier, ufunc, rank)
    operator_list.append(operator)
# compile some useful listings of the operators
operator_symbols = set(operator.symbol for operator in operator_list)
operator_by_length = sorted(
    operator_list, key=lambda key: len(key.symbol), reverse=True)
operator_dict = {}  # dictionary of lists
for operator in operator_list:
    symbol = operator.symbol
    if symbol not in operator_dict:
        operator_dict[symbol] = {}
    key = "R" if type(operator) is OperatorR else "LR"
    operator_dict[symbol][key] = operator
# assemble an hierarchical list of operators
operator_max_rank = max(operator.rank for operator in operator_list)
operator_hierarchy = []
for rank in reversed(range(operator_max_rank + 1)):
    operators = [
        operator for operator in operator_list if operator.rank == rank]
    if len(operators) > 0:
        operator_hierarchy.append(operators)


def split_by_operator(math_string_list):
    math_string = math_string_list[0]
    for operator in operator_by_length:
        identifier = operator.identifier
        # check if the string contains the operator
        idx = math_string.find(identifier)
        if idx >= 0:
            # split the string left and right of the operator
            string_left = math_string[:idx]
            string_right = math_string[idx + len(identifier):]
            # recursively process the left and right sides
            math_string_list = []
            if len(string_left) > 0:
                math_string_list.extend(split_by_operator([string_left]))
            math_string_list.append(operator.symbol)
            if len(string_right) > 0:
                math_string_list.extend(split_by_operator([string_right]))
            return math_string_list
    # split each element on remaining whitespaces, indicating a syntax error
    expanded_list = []
    for entry in math_string_list:
        splitted = entry.split()
        if len(splitted) > 1:
            message = "operands '{:}' and '{:}' must be joined by an operator"
            raise SyntaxError(message.format(splitted[0], splitted[1]))
        expanded_list.extend(splitted)
    return expanded_list


def substitute_operators(math_string_list):
    for i, entry in enumerate(math_string_list):
        try:
            overloads = operator_dict[entry]
            if len(overloads) == 1:
                operator = tuple(overloads.values())[0]
            else:
                if i == 0:
                    operator = overloads["R"]
                elif isinstance(math_string_list[i - 1], Operator):
                    operator = overloads["R"]
                else:
                    operator = overloads["LR"]
            math_string_list[i] = operator
        except KeyError:
            pass
    return math_string_list


def insert_term(math_string_list, idx):
    operator = math_string_list[idx]
    # check if there is a valid right operand/operand
    if idx + 1 == len(math_string_list):
        message = "operator '{:}' is not followed by an operand"
        raise SyntaxError(message.format(operator.symbol))
    elif type(math_string_list[idx + 1]) is OperatorLR:
        message = "operator '{:}' cannot be followed by operator '{:}'"
        raise SyntaxError(message.format(
            operator.symbol, math_string_list[idx + 1].symbol))
    # if there is a single operand operator following process it first
    elif type(math_string_list[idx + 1]) is OperatorR:
        math_string_list = insert_term(math_string_list, idx + 1)
    # check if there is a valid left operand/operand
    if type(operator) is OperatorLR:
        message = "operator '{:}' requires a left operand"
        if idx == 0:
            raise SyntaxError(message.format(operator.symbol))
        elif isinstance(math_string_list[idx - 1], Operator):
            raise SyntaxError(message.format(operator.symbol))
        # insert a term instance by consuming the operands
        insert_idx = idx - 1
        left_operand = math_string_list.pop(insert_idx)
        math_string_list.pop(insert_idx)
        right_operand = math_string_list.pop(insert_idx)
        term = MathTerm(operator)
        term.operands = (left_operand, right_operand)
    else:
        # insert a term instance by consuming the operands
        insert_idx = idx
        math_string_list.pop(insert_idx)
        right_operand = math_string_list.pop(insert_idx)
        term = MathTerm(operator)
        term.operands = (right_operand,)
    math_string_list.insert(insert_idx, term)
    return math_string_list


def replace_entry(math_string_list):
    for operators in operator_hierarchy:
        for idx, entry in enumerate(math_string_list):
            if entry in operators:
                return insert_term(math_string_list, idx)


def scan_brackets(math_string):
    math_string = math_string.strip()
    counts = Counter(math_string)
    # check the occurences of ( and )
    try:
        n_open = counts["("]
    except KeyError:
        n_open = 0
    try:
        n_close = counts[")"]
    except KeyError:
        n_close = 0
    # check that the number of brackets match
    if n_open > n_close:
        raise SyntaxError("too many opening brackets")
    elif n_open < n_close:
        raise SyntaxError("too many closing brackets")
    # list which parts of the expression are within brackets
    while len(math_string) > 0:
        level = 0
        bracket_level = []
        # assing each character a priority level based on the bracketing
        for char in math_string:
            if char == ")":  # decrease priority level
                level -= 1
            if level < 0:
                raise SyntaxError("too many closing brackets")
            bracket_level.append(level)
            if char == "(":  # increase priority level
                level += 1
        # check for redundant brackets
        if Counter(bracket_level)[0] == 2:  # remove redundant outer levels
            math_string = math_string[1:-1]
        else:
            break
    # process the term
    print(split(math_string).code)
    return bracket_level, math_string


def parse_operand(string):
    normlised = string.upper()
    # check for boolean values
    if normlised == "TRUE":
        return True
    elif normlised == "FALSE":
        return False
    # convert to integer or floating point
    try:
        return int(normlised)
    except (ValueError, OverflowError):  # inf raises OverflowError
        try:
            return float(normlised)
        except ValueError:
            message = "cannot convert '{:}' to numerical or boolean type"
            raise ValueError(message.format(string))


class MathTerm:

    _operands = None

    def __init__(self, operator):
        self._operator = operator

    @staticmethod
    def from_string(expression_string):
        # split the input on operator occurences
        splitted = split_by_operator([expression_string])
        # substitute the operator symbols by matching Operator instances
        substituted = substitute_operators(splitted)
        # create a MathTerm instance
        message = "malformed expression {:}".format(expression_string)
        while len(substituted) > 1:
            substituted = replace_entry(substituted)
            if substituted is None:
                raise SyntaxError(message)
        if len(substituted) != 1:
            raise SyntaxError(message)
        else:
            term = substituted[0]
        return term

    @property
    def symbol(self):
        return self._operator.symbol

    @property
    def ufunc(self):
        return self._operator.ufunc

    @property
    def nargs(self):
        return self._operator.nargs

    @property
    def operands(self):
        return self._operands

    @operands.setter
    def operands(self, operands):
        if type(operands) is not tuple:
            raise TypeError("operands must be tuple")
        if len(operands) != self.nargs:
            message = "operator '{:}' expects {:d} arguments but got {:d}"
            raise ValueError(message.format(
                self.ufunc.__name__, self.nargs, len(operands)))
        self._operands = tuple(operands)

    @property
    def code(self):
        # ufunc([left operand, ] right operand)
        if self.operands is None:
            raise RuntimeError("operands not set")
        operands_list = []
        for operand in self.operands:
            if isinstance(operand, MathTerm):  # call recursively on terms
                operands_list.append(operand.code)
            else:
                operands_list.append(str(operand))
        operands_string = ", ".join(operands_list)
        code = "{:}({:})".format(self.ufunc.__name__,  operands_string)
        return code

    @property
    def expression(self):
        # [left operand] operator right operand 
        if self.operands is None:
            raise RuntimeError("operands not set")
        expression_list = []
        for operand in self.operands:
            if isinstance(operand, MathTerm):  # call recursively on terms
                # wrap the operand term in brackets
                expression_list.append("(" + operand.expression + ")")
            else:
                expression_list.append(str(operand))
        # insert the operator symbol in the front or middle, depending on nargs
        expression_list.insert(-1, self.symbol)
        expression = " ".join(expression_list)
        return expression

    def __call__(self, table=None):
        if self.operands is None:
            raise RuntimeError("operands not set")
        # get the numerical values of the operands
        operand_values = []
        for operand in self.operands:
            if isinstance(operand, MathTerm):  # call recursively on terms
                values = operand(table)
            elif type(operand) is str:  # convert to numerical type
                try:
                    values = parse_operand(operand)  # convert from string
                except ValueError as e:
                    if table is None:
                        raise e
                    values = table[operand]  # get values from the table column
            else:
                values = operand
            operand_values.append(values)
        # evaluate the operator ufunc
        result = self._operator.ufunc(*operand_values)  # call ufunc
        return result


if __name__ == "__main__":

    import sys

    math_string = " ".join(sys.argv[1:])
    print("input:     ", math_string)
    term = MathTerm.from_string(math_string)
    print("code:      ", term.code)
    print("expression:", term.expression)
