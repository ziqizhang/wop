# Used to browse through the users and label them 

import webbrowser
import sys
import pandas as pd
import time

# Filename of the csv file - first argument 
# ---- The second row of the CSV file must be named "label" !!----
filename = sys.argv[1]
# Start of the range (first user on the file is 1)
start = int(sys.argv[2])
# End of the range (inclusive)
end = int(sys.argv[3])
# get data frame 
df = pd.read_csv(filename)
# Get the column of the profile links
our_column = df.twitter_id

# Have a list of the possible labels
# Labels are ordered as in the email
labels = ['P', 'HPI', 'Advocates', 'Research', 'HPO', 'Other']

# Iterate through the list of users 
for row in range(start-1,end):

    print("User number is: " + str(row+1))
    url = our_column[row]
    # open the url
    webbrowser.open(url)
    # get label from keyboard as a digit from 1 to 6
    label = input("Enter label: ")
    # Multiple labels are separated by a comma (e.g. 1,3)
    label = label.split(",")
    string = ""
    # map the digits into labels
    for i in label:
        string = string + "," + labels[int(i)-1]
    
    if string[0] == ",":
        string = string[1:]
    #write label
    df.label[row] = string
    
    print("Setting label: " + string)
    print()

# write the new file
df.to_csv("output_file.csv", index=False)
