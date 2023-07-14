# Access OpenAI Codex API

import json
import openai

# Retrieve API from json file
with open('/openai/.openai/api_key.json') as f:
    api = json.load(f)

# set API key
openai.api_key = api['key']

# Function to translate SAS code to Python
def translate_sas_to_python(sas_code):
    prompt = f'convert this SAS program to Python using the sklearn package:\n\n{sas_code}\n\nPython code:'

    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=2048,
        temperature=0.6,
        n=1,
        stop=None,
    )

    python_code = response.choices[0].text.strip()
    return python_code

# Example SAS code to translate
sas_code = '''
/* Read in the CSV file */
proc import datafile='path_to_csv_file.csv'
     out=mydata
     dbms=csv
     replace;
     getnames=yes;
run;

/* Generate the linear regression model */
proc reg data=mydata;
     model y = x1 x2 x3; /* Replace x1, x2, x3 with your predictor variables */
run;

/* Print model diagnostics */
proc reg data=mydata;
     model y = x1 x2 x3;
     output out=model_diagnostics p pchi r rstudent;
run;

/* Print the model diagnostics */
proc print data=model_diagnostics;
run;
'''

# Translate SAS code to Python
python_code = translate_sas_to_python(sas_code)
print(python_code)


### generated output ###
# # Import the necessary packages
# import pandas as pd
# from sklearn.linear_model import LinearRegression
#
# # Read in the CSV file
# mydata = pd.read_csv('path_to_csv_file.csv')
#
# # Generate the linear regression model
# X = mydata[['x1', 'x2', 'x3']]
# y = mydata['y']
# model = LinearRegression()
# model.fit(X, y)
#
# # Print model diagnostics
# print('Model intercept:', model.intercept_)
# print('Model coefficients:', model.coef_)
# print('Model R-squared:', model.score(X, y))