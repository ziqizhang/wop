#This script uses the outputs from identify_GS1scheme.py to extend the dataset with the relevant
#GS1 ontology.

import pandas as pd
import numpy as np
import os

input = "H:/SeedProject/Dataset/ProductCategorisation/goldstandard_eng_v1_cleanedCategories.csv"
output = "H:/SeedProject/Dataset/ProductCategorisation/goldstandard_eng_v1_cleanedCategories_GS1_name.csv"
f = open(output, "w+", encoding="utf-8")
f.write("GeneratedID;EntityNodeID;URL;HOST;s:name;s:description;s:brand;Properties;s:category;s:breadcrumb;")
f.write("GS1_Level1_Category;GS1_Level2_Category;GS1_Level3_Category;s:cleanedCategory;")

inputFields = ["cat", "catClean", "name", "cat_name", "catClean_name"]
#inputFields = ["cat_name", "catClean_name"]
startLevel = "1"
endLevels = ["4", "5"] #, "6"]
weightLevels = ["TF", "TF-IDF"]

df = pd.read_csv(input, header=0, delimiter=";", quoting=0, encoding="utf-8")
df = df.replace(np.nan, '', regex=True)
df = df.as_matrix()
name = df[:, 4]
cleanedCategories = df[:, 13]
originalCategories = df[:, 8]

start = 0
end = len(originalCategories)

gsCategoryDict = {}

for endLevel in endLevels:
    for inputField in inputFields:
        for weight in weightLevels:
            if inputField == "cat" or inputField == "cat_name":
                inputCategories = originalCategories
                mapping = "H:/SeedProject/GS1/cat_L" + startLevel + "L" + endLevel + "_" + weight + ".txt"
            elif inputField == "catClean" or inputField == "catClean_name":
                inputCategories = cleanedCategories
                mapping = "H:/SeedProject/GS1/catClean_L" + startLevel + "L" + endLevel + "_" + weight + ".txt"
            elif inputField == "name":
                inputCategories = name
                mapping = "H:/SeedProject/GS1/name_L" + startLevel + "L" + endLevel + "_" + weight + ".txt"

            #mapping = "H:/SeedProject/GS1/" + inputField + "_L" + startLevel + "L" + endLevel + "_" + weight + ".txt"
            print(mapping)

            exists = os.path.isfile(mapping)

            if exists:

                f.write(inputField + "_L" + startLevel + "L" + endLevel + "_" + weight + ";")

                #load mapping to a dictionary
                mappingdf = pd.read_csv(mapping, header=1, delimiter="\t", quoting=0, encoding="utf-8")
                mappingdf = mappingdf.as_matrix()

                mappingDict = {}
                nameMappingDict = {}

                for i in range(0, len(mappingdf[:, 0])):
                    inputCategory = mappingdf[i, 0]
                    gsCategory = mappingdf[i, 1]

                    mappingDict[inputCategory] = gsCategory

                # Load the name -> category mapping
                if "_name" in inputField:
                    nameMapping = "H:/SeedProject/GS1/name_L1L4_" + weight + ".txt"
                    #nameMapping = "H:/SeedProject/GS1/name_L1L4_TF.txt"
                    nameMappingdf = pd.read_csv(nameMapping, header=1, delimiter="\t", quoting=0, encoding="utf-8")
                    nameMappingdf = nameMappingdf.as_matrix()

                    for i in range(0, len(nameMappingdf[:, 0])):
                        inputName = nameMappingdf[i, 0]
                        gsCategory = nameMappingdf[i, 1]

                        nameMappingDict[inputName] = gsCategory

                #For each line in the input file, look at the input category, get the mapped GS1 equivalence
                #If the mapping is not available, include the value of the input category instead.
                for i in range(start, end):
                    inputCat = inputCategories[i]
                    if inputCat in mappingDict:
                        gsCategory = mappingDict[inputCat]
                    else:
                        #If the input category does not exist in the mapping,
                        #check if the name can be mapped to a category (as a back-up)
                        if "_name" in inputField:
                            #Use the category of the name if available
                            productName = name[i].replace("\t", " ")
                            if productName in nameMappingDict:
                                gsCategory = nameMappingDict[productName]
                            else:
                                gsCategory = inputCat
                        #If not, include the original input category as the category
                        else:
                            gsCategory = inputCat

                    if i in gsCategoryDict:
                        updatedCat = gsCategoryDict[i] + ";" + gsCategory
                        gsCategoryDict[i] = updatedCat
                    else:
                        gsCategoryDict[i] = gsCategory

f.write("\n")
indexStart = 0
indexEnd = 14

for i in range(start,end):
    for j in range(indexStart, indexEnd):
        f.write("%s;" % df[i, j])
    f.write("%s;\n" % (gsCategoryDict[i]))

f.close()