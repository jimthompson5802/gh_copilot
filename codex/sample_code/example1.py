# Python 3

# function to add or multiply two numbers based on an operator argument
def add_or_multiply(a, b, operator):
    if operator == "add":
        return a + b
    elif operator == "multiply":
        return a * b
    else:
        raise ValueError("Invalid operator")
