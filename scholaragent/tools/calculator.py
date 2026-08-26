"""内置工具 1:计算器。大模型心算容易出错,精确计算要靠工具。

为什么不用 eval():eval 会执行任意 Python 代码,模型一旦生成
恶意或错误的表达式,就等于直接在你的电脑上跑未知代码。
这里用 ast 把表达式解析成语法树,只放行数字和算术运算,其余一律拒绝。
这是"给 Agent 工具划安全边界"的最小示例 —— 以后做写文件、
执行代码这类更危险的工具时,同样的思路会更加重要。

安全边界有两层,缺一不可:
    1. 防执行代码 —— 只放行白名单里的语法节点(下面的 _eval)
    2. 防资源耗尽 —— 限制运算规模(下面的 _check_pow):
       像 9**9**9 这种表达式会陷入 C 层的不可中断大整数运算,
       算不完连 Ctrl+C 都救不了,只能杀掉整个终端
"""

import ast
import operator

from ..tool import Tool

# 允许的二元运算:加减乘除、整除、取余、乘方
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
# 允许的一元运算:正负号
_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _check_pow(base, exponent):
    """乘方规模检查:在真正计算之前拒绝会耗尽 CPU/内存的乘方。"""
    if abs(exponent) > 128:
        raise ValueError("乘方的指数过大(|指数| 上限 128)")
    if (isinstance(base, int) and isinstance(exponent, int) and exponent > 0
            and base.bit_length() * exponent > 10_000):
        raise ValueError("乘方结果过大,拒绝计算")


def safe_eval(expression: str):
    """只允许数字与算术运算的表达式求值器,其余成分一律抛 ValueError。"""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow):
                _check_pow(left, right)
            return _BINARY_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"不支持的表达式成分:{type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)


class CalculatorTool(Tool):
    name = "calculator"
    description = "精确计算数学表达式。支持 + - * / // % ** 和括号,例如 (3+5)*12"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式,例如 (3+5)*12",
            },
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> str:
        result = safe_eval(expression)
        # 8.0 这类整数结果显示成 8,输出更干净
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
