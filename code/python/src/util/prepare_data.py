import sys
import pandas as pd

# 

filename = sys.argv[1] 

start = int(sys.argv[2])

end = int(sys.argv[3]) 

df = pd.read_csv(filename)

label_column = df.label

for row in range(start-1, end):
    
    # change P with Patient 
    label = str(label_column[row])
    #print("row is: " + str(row) + " " + label + df.twitter_id[row])
    
    # split at ,
    label = label.split(",")
    
    # check if the list contains advocate
    if "Advocates" in label:
        # check if HPO or HPI or Research
        if "HPO" in label:
            # delete Advocates
            label.remove("Advocates")
        elif "HPI" in label:
            # delete Advocates
            label.remove("Advocates")
        elif "Research" in label:
            # delete Advocates
            label.remove("Advocates")
            
    # change P with Patient 
    label = ["Patient" if x == "P" else x for x in label]
         
    # join the elements again
    
    string = ",".join(label)
    
    # replace label 
    label_column[row] = string
    
df.to_csv("output_file.csv", index = False)