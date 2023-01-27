 test_add_or_multiply():
    assert add_or_multiply(1, 2, "add") == 3
    assert add_or_multiply(1, 2, "multiply") == 2
    assert add_or_multiply(1, 2, "divide") == None

# Run unit tests
test_add_or_multiply()

# Python 2

# function to add or multiply two numbers based on an operator argument
def add_or_multiply(a, b, operator):
    if operator == "add":
        return a + b
    elif operator == "multiply":
        return a * b
    else:
        raise ValueError("Invalid operator")

# Unit test
def test_add_or_multiply():
    assert add_or_multiply(1, 2, "add") == 3
    assert add_or_multiply(1, 2, "multiply") == 2
    assert add_or_multiply(1, 2, "divide") == None

# Run unit