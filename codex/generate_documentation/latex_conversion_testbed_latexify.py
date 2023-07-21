
from sympy import parse_expr, latex

def python_to_latex(expression_str):
    try:
        expr = parse_expr(expression_str)
        latex_expr = latex(expr, mul_symbol='times')
        return latex_expr
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage:
python_expression = "x=2*y"
latex_expression = python_to_latex(python_expression)
print(latex_expression)
