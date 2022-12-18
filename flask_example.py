# import Flask class from flask module
from flask import Flask, render_template, request


# function to compute value of two numbers based on operator flag
def compute_value(num1, num2, operator):
    if operator == "add":
        return num1 + num2
    elif operator == "subtract":
        return num1 - num2
    elif operator == "multiply":
        return num1 * num2
    elif operator == "divide":
        return num1 / num2
    else:
        return "Invalid operator"

# setup flask app to serve compute_value function
app = Flask(__name__)
@app.route('/compute/<int:num1>/<int:num2>/<operator>')
def compute(num1, num2, operator):
    return str(compute_value(num1, num2, operator))

if __name__ == '__main__':
    app.run(debug=True)

