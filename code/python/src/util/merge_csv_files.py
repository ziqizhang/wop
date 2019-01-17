# Takes 2 params with the names of the files to be merged

import sys 
import pandas as pd 

filename1 = sys.argv[1]
filename2 = sys.argv[2]

df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)

merged_df = df1.append(df2)

merged_df.to_csv("merged_output.csv", index = False)