import pandas as pd

# Read the data from file
data = pd.read_csv('your_file_path')

# Convert selected variables to strings
data['var1'] = data['var1'].astype(str)
data['var2'] = data['var2'].astype(str)
data['var3'] = data['var3'].astype(str)

# Output the reformatted data
data.to_csv('reformatted_data.csv', index=False)