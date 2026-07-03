"""
Safe arithmetic evaluator for the model's <calc>...</calc> tool calls.

Deliberately does NOT use eval()/exec() -- the model's output is untrusted
text, and eval() on untrusted text is arbitrary code execution. Instead we
parse to a Python AST and only walk a small whitelist of numeric/operator
node types, rejecting anything else (names, calls, attribute access, etc.).
"""

import ast
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalcError(ValueError):
    pass


def safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression like '47+89' or '(3+4)*2'."""
    expr = expr.strip()
    if not expr:
        raise CalcError("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"could not parse '{expr}': {e}") from e
    return _eval_node(tree.body, expr)


def _eval_node(node, original_expr):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalcError(f"non-numeric constant in '{original_expr}'")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left, original_expr)
        right = _eval_node(node.right, original_expr)
        try:
            return _BINOPS[type(node.op)](left, right)
        except ZeroDivisionError as e:
            raise CalcError(f"division by zero in '{original_expr}'") from e
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, original_expr))
    raise CalcError(f"unsupported syntax in '{original_expr}'")


def format_result(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(round(value, 6))
