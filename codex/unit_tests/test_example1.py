 test_add_or_multiply():
    assert add_or_multiply(1, 2, "add") == 3
    assert add_or_multiply(1, 2, "multiply") == 2
    assert add_or_multiply(1, 2, "subtract") == None

# Run unit tests
test_add_or_multiply()