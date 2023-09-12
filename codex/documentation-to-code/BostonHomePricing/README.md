# Boston Home Price Prediction Model 

Objective is to create a python based regression model to predict Boston Home Prices
https://www.kaggle.com/datasets/vikrishnan/boston-house-prices

## Setup
* Setup environment variable `OPENAI_API_KEY` with your OpenAI API key.
* Install `gpt-engineer` package.
* Copy the data from the above link to 
 ./codex/documentation-to-code/generate-synthetic-data/workspace/data/housing2.csv 
 * The data set has quite a few white spaces, so pelase make sure you delete them.


## Sample execution


```
$ cd documentation-to-code
$ gpt-engineer BostonHomePricing
```

## Observations

* Ridge Regression Algorithm seems to have an issue, so had to exclude it.
* By Placing the data under the below directory was getting deleted when running gpt-engineer, so as a work around it is better to have it in a different location
    documentation-to-code/BostonHomePricing/workspace/data/housing2.csv 

# Output from a run

8ecb5486d969:/workspaces/gh_copilot/codex/documentation-to-code# gpt-engineer BostonHomePricing
Summary of areas that need clarification:

1. Definition of "low correlation" in the context of feature selection.
2. The specific machine learning models to be used for multiple regression, given that Ridge regression is not to be used.
3. The metric to be used for calculating the loss value.
4. The format and location for saving the models.

Clarifying question:

Could you please specify what you consider as "low correlation" for the purpose of feature selection in this context?
(answer in text, or "c" to move on)


(letting gpt-engineer make its own assumptions)

Assumptions:

1. "Low correlation" is defined as an absolute correlation coefficient less than 0.5.
2. The multiple regression models to be used are Linear Regression, Lasso Regression, and ElasticNet Regression.
3. The loss value will be calculated using Mean Squared Error (MSE).
4. The models will be saved in the current working directory in pickle format (.pkl).

Clarifying question:

Are these assumptions acceptable, or would you like to modify any of them?
The core classes, functions, and methods that will be necessary are:

1. `preprocess_data`: This function will handle the preprocessing of the data, including filling missing values, handling outliers, and feature scaling.
2. `evaluate_correlations`: This function will evaluate the correlations between each feature and the target feature, and remove features with low correlation.
3. `split_data`: This function will split the data into a training set and a test set.
4. `train_models`: This function will train multiple regression models on the training data.
5. `evaluate_models`: This function will evaluate the models on the test data and print the loss value.
6. `save_models`: This function will save the trained models to disk.

Now, let's write the code for each of these functions.

main.py
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
import pickle
import os
from preprocess_data import preprocess_data
from evaluate_correlations import evaluate_correlations

# Load data
data = pd.read_csv('../../generate-synthetic-data/workspace/data/housing2.csv')

# Preprocess data
data = preprocess_data(data)

# Evaluate correlations and remove low-correlation features
data = evaluate_correlations(data, 'MEDV')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('MEDV', axis=1), data['MEDV'], test_size=0.2, random_state=42)

# Train models
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate models
for name, model in models.items():
    predictions = model.predict(X_test)
    loss = mean_squared_error(y_test, predictions)
    print(f'{name} Loss: {loss}')

# Save models
for name, model in models.items():
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
```

preprocess_data.py
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Fill missing values with median
    data.fillna(data.median(), inplace=True)

    # Handle outliers using IQR method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Feature scaling
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data
```

evaluate_correlations.py
```python
import pandas as pd

def evaluate_correlations(data, target):
    # Calculate correlations
    correlations = data.corr()[target]

    # Remove low-correlation features
    data = data.drop(correlations[correlations.abs() < 0.5].index, axis=1)

    return data
```

requirements.txt
```
pandas
scikit-learn
```

This code should be fully functional and compatible with each other. The `main.py` file is the entry point of the program, and it imports functions from the `preprocess_data.py` and `evaluate_correlations.py` files. The `requirements.txt` file lists the necessary dependencies.```bash
pip install -r requirements.txt
```

```bash
python main.py
```
Do you want to execute this code?

pip install -r requirements.txt

python main.py


If yes, press enter. Otherwise, type "no"


Executing the code...

Note: If it does not work as expected, consider running the code in another way than above.

You can press ctrl+c *once* to stop the execution.

Requirement already satisfied: pandas in /usr/local/lib/python3.9/site-packages (from -r requirements.txt (line 1)) (2.0.3)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.9/site-packages (from -r requirements.txt (line 2)) (1.3.0)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.9/site-packages (from pandas->-r requirements.txt (line 1)) (2.8.2)
Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.9/site-packages (from pandas->-r requirements.txt (line 1)) (2023.3)
Requirement already satisfied: numpy>=1.20.3 in /usr/local/lib/python3.9/site-packages (from pandas->-r requirements.txt (line 1)) (1.25.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.9/site-packages (from pandas->-r requirements.txt (line 1)) (2023.3)
Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.9/site-packages (from scikit-learn->-r requirements.txt (line 2)) (1.3.2)
Requirement already satisfied: scipy>=1.5.0 in /usr/local/lib/python3.9/site-packages (from scikit-learn->-r requirements.txt (line 2)) (1.11.2)
Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.9/site-packages (from scikit-learn->-r requirements.txt (line 2)) (3.2.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.9/site-packages (from python-dateutil>=2.8.2->pandas->-r requirements.txt (line 1)) (1.16.0)
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

[notice] A new release of pip is available: 23.0.1 -> 23.2.1
[notice] To update, run: pip install --upgrade pip
LinearRegression Loss: 0.44557038421755396
Lasso Loss: 0.8650844486326756
ElasticNet Loss: 0.7203932393479306