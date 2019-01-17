import sys
import pandas as pd

# 

filename = sys.argv[1] 

start = int(sys.argv[2])

end = int(sys.argv[3]) 

df = pd.read_csv(filename)


label_column = df.label

for row in range(start-1, end):
    
    string = ""
    label1 = str(df.label[row])
    if label1 == "nan":
        label1 = ""
    else:
        string = label1
    label2 = str(df.label2[row])
    if label2 == "nan":
        label2 = ""
    else:
        string = string + "," + label2
    label3 = str(df.label3[row])
    if label3 == "nan":
        label3 = ""
    else:
        string = string + "," + label3
 
    df.label[row] = string 
    
df.to_csv("output_file_merged_labels.csv", index = False)