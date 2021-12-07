import json
import operator

''''
This method reads the json data file and save them as a matrix where each row is an instance with the following columns:
- 0: id
- 1: name
- 2: description
- 3: categorytext
- 4: url
- 5: lvl1
- 6: lvl2
- 7: lvl3
'''
def read_json(in_file):
    matrix=[]
    with open(in_file) as file:
        line = file.readline()

        while line is not None and len(line)>0:
            js=json.loads(line)

            row=[js['ID'],js['Name'],js['Description'],js['CategoryText'],js['URL'],js['lvl1'],js['lvl2'],js['lvl3']]
            matrix.append(row)
            line=file.readline()
    return matrix

'''
output a matrix in the above format to json
'''
def write_json(matrix, out_file):
    freq=dict()
    with open(out_file,'w') as file:
        for row in matrix:
            data=dict()
            data["ID"]=row[0]
            data["Name"] = row[1]
            data["Description"] = row[2]
            data["CategoryText"] = row[3]
            data["URL"] = row[4]
            data["lvl1"] = row[5]
            data["lvl2"] = row[6]
            data["lvl3"] = row[7]
            js=json.dumps(data)

            file.write(js+"\n")

            if row[7] in freq.keys():
                freq[row[7]]+=1
            else:
                freq[row[7]]=1

    sorted_x = sorted(freq.items(), key=operator.itemgetter(1))
    for t in sorted_x:
        print("lvl3 class {}={}".format(t[0],t[1]))
    print("\n")