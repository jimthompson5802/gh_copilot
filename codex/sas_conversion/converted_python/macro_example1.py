import pandas as pd
import statsmodels.api as sm

def generate_regression_models(input_files):
    for i in range(len(input_files)):
        input_file = input_files[i]

        mydata = pd.read_csv(input_file, delimiter=',', skiprows=1)
        x = mydata['x']
        y = mydata['y']

        model = sm.OLS(y, sm.add_constant(x))
        results = model.fit()

        results.save(f"outest{i+1}.pickle")

        print(f"Linear regression model for {input_file} has been generated")

    print("All regression models have been generated successfully")

input_files = ["path/to/file1.csv", "path/to/file2.csv", "path/to/file3.csv"]
generate_regression_models(input_files)