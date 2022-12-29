# import missing Flask module
from flask import Flask, request


# function to add two numbers
def add(num1, num2):
    return num1 + num2


# function to subtract two numbers
def subtract(num1, num2):
    return num1 - num2


# function add, subtract, multiply, or divide two numbers based on operator flag
def compute_value(num1, num2, operator):
    if operator == "add":
        return add(num1, num2)
    elif operator == "subtract":
        return subtract(num1, num2)
    elif operator == "multiply":
        return num1 * num2
    elif operator == "divide":
        return num1 / num2
    else:
        return "Invalid operator"


# define string for compute_value web page form
compute_value_form = """
<!DOCTYPE html>
<html>
<head>
<title>Compute Value</title>
</head>
<body>
<h1>Compute Value</h1>
<form action="/compute" method="post">
    <label for="num1">Number 1:</label>
    <input type="text" id="num1" name="num1"><br><br>
    <label for="num2">Number 2:</label>
    <input type="text" id="num2" name="num2"><br><br>
    <label for="operator">Operator:</label>
    <select id="operator" name="operator">
        <option value="add">Add</option>
        <option value="subtract">Subtract</option>
        <option value="multiply">Multiply</option>
        <option value="divide">Divide</option>            
    </select><br><br>
    <input type="submit" value="Submit">
</form>
</body>
</html>
"""

# define web page to display result of compute_value function
compute_value_result = """
<!DOCTYPE html>
<html>
<head>
<title>Compute Value Result</title>
</head>
<body>
<h1>Compute Value Result</h1>
<p>Result: {}</p>
</body>
</html>
"""

# setup flask app to accept connection from any source on port 8080
# to server compute_value web page form
app = Flask(__name__)


@app.route('/compute', methods=['GET', 'POST'])
def compute():
    if request.method == 'POST':
        num1 = int(request.form['num1'])
        num2 = int(request.form['num2'])
        operator = request.form['operator']
        return compute_value_result.format(compute_value(num1, num2, operator))
    else:
        return compute_value_form


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)





