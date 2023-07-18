# convert the following sas code to Python
# data reformatted_data;
#    infile 'your_file_path';
#    input var1 var2 var3 var4 var5;
#
#    /* Convert selected variables to strings */
#    var1 = put(var1, $CHAR.);
#    var2 = put(var2, $CHAR.);
#    var3 = put(var3, $CHAR.);
#
#    /* Keep var4 and var5 as numeric */
#
#    /* Output the reformatted data */
#    output;
# run;

# import packages
import pandas as pd
import numpy as np

# read in data
df = pd.read_csv('your_file_path')

# convert selected variables to strings
df['var1'] = df['var1'].astype(str)
df['var2'] = df['var2'].astype(str)
df['var3'] = df['var3'].astype(str)

# keep var4 and var5 as numeric

# output the reformatted data
df.to_csv('your_file_path', index=False)
