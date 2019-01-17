# Used to duplicate users with multiple labels

import sys
import pandas as pd

# 

filename = sys.argv[1] 

start = int(sys.argv[2])

end = int(sys.argv[3]) 

df = pd.read_csv(filename)

col_names = ['twitter_id', 'label']

new_df = pd.DataFrame(columns = col_names)

label_column = df.label

for row in range(start-1, end):
 
    label = str(label_column[row])
    
    id = df.twitter_id[row]
   
    # split at ,
    label = label.split(",")
    
    for l in label:
    
        new_row = [id, l]
        new_df.loc[len(new_df)] = new_row
    
    
new_df.to_csv("output_file.csv", index = False)