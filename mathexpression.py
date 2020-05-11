from collections import Counter
from copy import copy

import numpy as np


class MathTerm:

    _operator_map = {
        "**": np.power,
        "*": np.multiply,
        "/": np.divide,
        "%": np.mod,
        "+": np.add,
        "-": np.subtract,
        "~": np.bitwise_not,
        "&": np.bitwise_and,
        "|": np.bitwise_or,
        "^": np.bitwise_xor,
        "==": np.equal,
        "!=": np.not_equal,
        "<": np.less,
        ">": np.greater,
        "<=": np.less_equal,
        ">=": np.greater_equal,
        "NOT": np.logical_not,
        "AND": np.logical_and,
        "OR": np.logical_or,
        "XOR": np.logical_xor}
    _operator_nargs = {
        "**": 2,
        "*": 2,
        "/": 2,
        "%": 2,
        "+": 2,
        "-": 2,
        "~": 1,
        "&": 2,
        "|": 2,
        "^": 2,
        "==": 2,
        "!=": 2,
        "<": 2,
        ">": 2,
        "<=": 2,
        ">=": 2,
        "NOT": 1,
        "AND": 2,
        "OR": 2,
        "XOR": 2}

    def __init__(self, symbol):
        self.symbol = copy(symbol)
        try:
            # get the correct numpy ufunc
            self.func = self._operator_map[symbol]
            self.name = self.func.__name__
            # figure out the correct argument layout
            self.nargs = self._operator_nargs[symbol]
            if self.nargs == 1:
                self.args = ["%1"]
            else:
                self.args = ["%1", "%2"]
        except KeyError:
            raise SyntaxError("invalid operator '%s'" % symbol)

    def set_args(self, *args):
        if len(args) != self.nargs:
            raise SyntaxError(
                "invalid number of arguments for operator '%s'" % self.name)
        else:
            self.args = tuple(args)

    def __repr__(self):
        string = "<%s object of type %s() at %s>" % (
            self.__class__.__name__, self.name, hex(id(self)))
        return string

    def code(self):
        # function([left argument, ] right argument)
        arg_strings = []
        for arg in self.args:
            if type(arg) is type(self):
                arg_strings.append(arg.code())
            else:
                arg_strings.append(str(arg))
        return "%s(%s)" % (self.func.__name__,  ", ".join(arg_strings))

    def expression(self):
        # [left argument] operator right argument 
        strings = []
        for arg in self.args:
            if type(arg) is type(self):
                strings.append("(" + arg.expression() + ")")
            else:
                strings.append(str(arg))
        strings.insert(-1, self.symbol)
        return " ".join(strings)

    def eval(self, var_idx_stack):
        expression_lines = []
        var_idx = var_idx_stack[-1]
        arg_strings = []
        for arg in self.args:
            if type(arg) is type(self):
                next_idx = max(var_idx_stack) + 1
                var_idx_stack.append(next_idx)
                arg_strings.append("%%%c" % chr(next_idx))
                expression_lines.extend(
                    arg.eval(var_idx_stack))
            else:
                arg_strings.append(arg)
        eval_term = self.__class__(self.symbol)
        eval_term.set_args(*arg_strings)
        expression_lines.append(
            "  %%%c = %s" % (chr(var_idx), eval_term.expression()))
        return expression_lines

    @staticmethod
    def interpret_variable(string):
        if string.upper() == "TRUE":
            return True
        elif string.upper() == "FALSE":
            return False
        try:
            return int(string)
        except ValueError:
            try:
                return float(string)
            except ValueError:
                message = "cannot convert '{:}' to numerical type"
                raise ValueError(message.format(string))

    def __call__(self, table):
        if type(self.args[0]) is type(self):
            arg1 = self.args[0](table)
        else:
            try:
                arg1 = table[self.args[0]]
            except (KeyError, TypeError):
                arg1 = self.interpret_variable(self.args[0])
        if self.nargs == 1:
            return self.func(arg1)
        else:
            if type(self.args[1]) is type(self):
                arg2 = self.args[1](table)
            else:
                try:
                    arg2 = table[self.args[1]]
                except (KeyError, TypeError):
                    arg2 = self.interpret_variable(self.args[1])
            return self.func(arg1, arg2)


class MathExpression:

    _operator_hierarchy = (
        ("**",),
        ("*", "/", "%",),
        ("+", "-",),
        ("==", "!=", "<", ">", "<=", ">=",),
        ("~",),
        ("&", "|", "^",),
        ("NOT",),
        ("AND", "OR", "XOR",))

    def __init__(self, string):
        self.string = string
        # scan the expression to find terms that are within brackets
        is_bracketed = self.scan_brackets()
        # split the expression into elements: values/arguments, operators and
        # terms enclosed by brackets
        expression_list = []
        term_string = ""
        for i in range(len(string) - 1):
            term_string += string[i]
            # accumulate what is belonging to the term
            if is_bracketed[i] and not is_bracketed[i + 1]:
                    # process term recursively
                    expression = self.__class__(term_string)
                    expression_list.append(expression.term)
                    term_string = ""
            # accumulate what is belonging to the values/arguments
            elif not is_bracketed[i] and is_bracketed[i + 1]:
                    # split by white spaces for processing
                    expression_list.extend([
                        item for item in term_string.strip("() ").split()])
                    term_string = ""
        # add the last missing character which must be part of values/arguments
        term_string += string[-1]
        expression_list.extend([
            item for item in term_string.strip("() ").split()])
        # recursively construct MathTerm based on the operator precedence
        self.term = self.interpret_expression(expression_list)

    def scan_brackets(self):
        counts = Counter(self.string)
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
            raise SyntaxError(
                "missing %d closing brackets" % (n_open - n_close))
        elif n_open < n_close:
            raise SyntaxError(
                "missing %d opening brackets" % (n_close - n_open))
        # list which parts of the expression are within brackets
        bracket_level = 0
        is_bracketed = []
        for char in self.string:
            if char == ")":
                bracket_level -= 1
            # within bracket if bracket_level > 0
            is_bracketed.append(bool(bracket_level))
            if char == "(":
                bracket_level += 1
        return is_bracketed

    def subsitute_term(self, expression_list, operator_list):
        substituted_list = copy(expression_list)
        # match expression against list of operators
        for i, arg in enumerate(expression_list):
            if arg in operator_list:
                # remove the operator and arguments and reinsert a MathTerm
                # object instead
                term = MathTerm(substituted_list.pop(i))
                right_arg = substituted_list.pop(i)  # operator must have this
                if term.nargs == 1:
                    # insert in list where the operator symbol was
                    term.set_args(right_arg)
                    substituted_list.insert(i, term)
                else:
                    # insert in list where the right argument was
                    term.set_args(substituted_list.pop(i - 1), right_arg)
                    substituted_list.insert(i - 1, term)
                break  # we do only one substitution per call
        return substituted_list

    def interpret_expression(self, expression_list):
        # scan and replace expression by MathTerms based on operator precedence
        for operator_list in self._operator_hierarchy:
            # continue replace
            while any(  # subsitute symbols by MathTerms
                    operator in expression_list
                    for operator in operator_list):
                expression_list = self.subsitute_term(
                    expression_list, operator_list)
            if len(expression_list) == 1:
                break
        term_list = expression_list[0]
        return term_list

    def code(self):
        # call the code construction method from the internal MathTerm
        return self.term.code()

    def expression(self):
        # call the expression construction method from the internal MathTerm
        return self.term.expression()

    def eval(self):
        # write down the internal MathTerm operator by operator using variable
        # substitutions
        var_idx_stack = [ord("A") - 1]
        expression_lines = self.term.eval(var_idx_stack)
        expression_lines[-1] = "return" + expression_lines[-1].split("=", 1)[1]
        return "\n".join(expression_lines)

    def __call__(self, table=None):
        return self.term(table)


if __name__ == "__main__":

    import sys

    if len(sys.argv) > 2:
        strings = [" ".join(sys.argv[1:])]
    elif len(sys.argv) == 2:
        strings = [sys.argv[1]]
    else:
        strings = [
            "X + (Y + Z) + (P + Q) + R",
            "(X + Y) + Z + (P + Q) + R",
            "(X + Y) + ((Z + P) + Q)",
            "X + Y > Z / P ^ Q - R % S AND NOT T",
            "(X < Y) AND ((Z < P + 1) - (Q < 4 * R)) OR NOT S"]

    for string in strings:
        term = MathExpression(string)
        print("#" * 40)
        print("input:     ", string)
        for i, t in enumerate(term.eval().split("\n")):
            if i == 0:
                print("term list: ", t)
            else:
                print("           ", t)
        print("bracketed: ", term.expression())
        print("internal:  ", term.code())
