#This script performs a look up to identify the most similar GS1 scheme
#based on an input category (either the original category or the cleaned
#category)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

input = "H:/SeedProject/Dataset/ProductCategorisation/goldstandard_eng_v1_cleanedCategories.csv"
schema = "H:/SeedProject/GS1/EN 2018-12/EN/GPC Schema 2018-12 EN.txt"
          #/GoogleDrive/wop/GPC Schema 2018-12 EN.txt

#The following variables need to be set:
#1. the input category (original category (cat) or cleaned category (catClean)
#2. the ontology levels that the look-up should be performed:
#   e.g., startLevel = 1 and endLevel = 5 means the look up will be performed against
#   a concatenated category values between GS1 level 1 to level 5.
#3. the weight of the cosine similarity ("TF" or "TF-IDF")
inputField = "catClean"  # options: cat, catClean
startLevel = "1"         # 1
endLevel = "5"           # options: 4, 5 or 6
weight = "TF-IDF"        # options: "TF" or "TF-IDF"

output = "H:/SeedProject/GS1/" + inputField + "_L" + startLevel + "L" + endLevel + "_" + weight + ".txt"

f = open(output, "w+", encoding="utf-8")
f.write("originalCategory\tcleanedCategory\trelatedGSCategory\n")

df = pd.read_csv(schema, header=0, delimiter="\t", quoting=0, encoding="utf-8")
df = df.replace(np.nan, '', regex=True)
df = df.as_matrix()

segmentDesc = df[:, 1]
familyDesc =  df[:, 3]
classDesc =  df[:, 5]
brickDesc =  df[:, 7]
coreDesc =  df[:, 9]
coreValue =  df[:, 11]
lists = []

for i in range(len(df)):
    #4 levels:
    if (startLevel == "1" and endLevel == "4"):
        category = segmentDesc[i] + " > " + familyDesc[i] + " > " + classDesc[i] + " > " + brickDesc[i]

    elif (startLevel == "1" and endLevel == "5"):
        category = segmentDesc[i] + " > " + familyDesc[i] + " > " + classDesc[i] + " > " + brickDesc[i] + " > " \
                   + coreDesc[i]

    elif (startLevel == "1" and endLevel == "6"):
        category = segmentDesc[i] + " > " + familyDesc[i] + " > " + classDesc[i] + " > " + brickDesc[i] + " > " \
                   + coreDesc[i] + " > " + coreValue[i]
    else:
        print("ERROR: Please define a suitable startLevel and endLevel of the ontology.")

    if category not in lists:
        lists.append(category)

print(len(lists))

df = pd.read_csv(input, header=0, delimiter=";", quoting=0, encoding="utf-8")
df = df.replace(np.nan, '', regex=True)
df = df.as_matrix()
cleanedCategories = df[:, 13]
originalCategories = df[:, 8]

if (inputField == "cat"):
    inputCategories = originalCategories
elif (inputField == "catClean"):
    inputCategories = cleanedCategories

stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words = stopWords)
transformer = TfidfTransformer()

trainVectorizer = vectorizer.fit_transform(lists)

if weight == "TF-IDF":
    trainVectorizerArray = vectorizer.fit_transform(lists).toarray()
    transformer.fit(trainVectorizerArray)
    tfidfTrain = transformer.transform(trainVectorizerArray)

count = 1;
matchesFound = 1;

for inputCat in np.unique(inputCategories):
    print("%s input category: %s" % (count, inputCat))
    count += 1

    test_set = [inputCat]

    if weight == "TF":
        testVectorizer = vectorizer.transform(test_set)
        results = cosine_similarity(testVectorizer, trainVectorizer)[0]
    elif weight == "TF-IDF":
        testVectorizerArray = vectorizer.transform(test_set).toarray()

        tfidfTest = transformer.transform(testVectorizerArray)
        results = cosine_similarity(tfidfTest, tfidfTrain)[0]

    score = max(results)

    if score > 0:
        maxIndex = [i for i, j in enumerate(results) if j == max(results)]

        for index in maxIndex:
            print("--- %d. %s" % (matchesFound, lists[index]))
            f.write("%s\t%s\n" % (inputCat, lists[index]))
            matchesFound += 1
            break

f.close()
