import argparse

# function to add two numbers
def add(a, b):
    return a + b


# function to add or multiply two numbers based on an operator argument
def add_or_multiply(a, b, operator):
    if operator == "add":
        return a + b
    elif operator == "multiply":
        return a * b
    else:
        raise ValueError("Invalid operator")

# setup main
if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=int, default=1)
    parser.add_argument("--b", type=int, default=2)
    parser.add_argument("--operator", type=str, default="add")
    args = parser.parse_args()

    # call add_or_multiply
    result = add_or_multiply(args.a, args.b, args.operator)

    # print result
    print(result)