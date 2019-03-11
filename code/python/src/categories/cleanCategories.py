import pandas as pd
import numpy as np
import re
import nltk

from urllib.parse import unquote
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words("english")

csv_training_text_data = "H:/SeedProject/Dataset/ProductCategorisation/goldstandard_eng_v1.csv"
output = "H:/SeedProject/Dataset/ProductCategorisation/goldstandard_eng_v1_cleanedCategories.csv"
f = open(output, "w+", encoding="utf-8")
f.write("GeneratedID;EntityNodeID;URL;HOST;s:name;s:description;s:brand;Properties;s:category;s:breadcrumb;")
f.write("GS1_Level1_Category;GS1_Level2_Category;GS1_Level3_Category;s:cleanedCategory\n")

df = pd.read_csv(csv_training_text_data, header=0, delimiter=";", quoting=0, encoding="utf-8")
df = df.replace(np.nan, '', regex=True)
df = df.as_matrix()

print(len(df))
hosts = df[:, 3]
categories = df[:, 8]
breadcrumbs = df[:, 9]
#newCategories = categories

#id = 1
start = 0#3343#6400
id = start+1
#end = 3345
end = len(categories)
tfidfList = {}

def createTFIDF(host):
    corpus = []
    for i in range(len(df)):
        if (host in hosts[i]):
            categories[i] = normaliseCategories(categories[i])
            corpus.append(categories[i])
    print(len(corpus))

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_

    print("IDF: ", idf)
    dictionary = dict(zip(vectorizer.get_feature_names(), idf))
    print(dictionary)

    return dictionary

def normaliseCategories(originalCategory):
    originalCategory = unquote(originalCategory)
    originalCategory = originalCategory.replace("<br>", "")
    originalCategory = originalCategory.replace("\t", " > ")
    originalCategory = originalCategory.replace(" › ", " > ")
    originalCategory = originalCategory.replace("\\|", " > ")
    originalCategory = re.sub(r" [>/|]+[ >]*", " > ", originalCategory)
    originalCategory = originalCategory.strip()
    return originalCategory

def getCategories(originalCategory):
    categoryValues = []

    regexp1 = re.compile(r' [>/|]+[ >]*')
    regexp2 = re.compile(r'[^\s],[^\s]')
    if re.search(regexp1, originalCategory):
        categoryValues = re.split(" [>/|]+[ >]*", originalCategory)
    elif re.search(regexp2, originalCategory):
        categoryValues = re.split(",", originalCategory)
    else:
        categoryValues.append(originalCategory)

    return categoryValues

def getTfIdfScore(text, dictionary):
    print("Text:", text)
    words = re.split("[\W]+", text.lower())
    print(words)

    score=0
    totalWords=len(words)

    for word in words:
        if word in dictionary.keys():
            score += dictionary[word]
            print("IDF for the text", word, "is", dictionary[word])
        else:
            print("Word %s does not exist." %word)
            totalWords-=1

    #need to normalise the scores based on the total words?
    if totalWords >= 1:
        score = score/totalWords
    return score

def cleanCategory(categoryValues, tfidf):
    i=1
    found=0
    total=len(categoryValues)-1

    #replace College, league names
    #originalCategory = originalCategory.replace("College > ", "")
    #regexp = re.compile(r'^[A-Z]+ > ')
    #originalCategory = re.sub(r'^[A-Z]+ > ', "", originalCategory)

    # H1: Replace the text if it's written in all upper cases
    # This is only checked in the first category

    categoryValues[0] = categoryValues[0].strip()
    parentCategory = categoryValues[0]

    value = "College"
    if parentCategory == value:
        categoryValues.remove(value)

    regexp = re.compile(r'^[A-Z]{3,6}$')
    if regexp.match(parentCategory):
        categoryValues.remove(parentCategory)

    if len(categoryValues) > 0:
        parentCategory = categoryValues[0]

        # Some categories are based on a team's name, identified as the first category (parent category)
        # e.g., "North Carolina Tar Heels > North Carolina Tar Heels Ladies"
        while i<=(len(categoryValues)-1):
            categoryValues[i] = categoryValues[i].strip()
            if categoryValues[i] == parentCategory:
                #value is exactly the same
                #don't do anything but to report found, so the parent category gets deleted
                found=1
            elif parentCategory in categoryValues[i]:
                #print("Before replacement: %s" % categoryValues[i])
                categoryValues[i] = categoryValues[i].replace(parentCategory, "").strip()
                #print("After replacement: %s" % categoryValues[i])
                found=1
            i+=1

        if(found == 1):
            del categoryValues[0]

        print("53: ", categoryValues)

        maxScore = 0
        maxScoreCategory = ""

        # If categories are related to price rather than topics (e.g., "Deals", "Sale", or "$") remove them
        for category in list(categoryValues):
            if "Sale" in category or "Deals" in category or "$" in category:
                print("62: ", categoryValues, "before removing", category)
                categoryValues.remove(category)

        #If more than one category are left, compute the TF-IDF score and chose the highest scoring category
        if len(categoryValues) > 1:
            for category in list(categoryValues):
                #check if the category represent "Sale" or contain price.
                #If that was the case, delete the values.
                score = getTfIdfScore(category, tfidf)
                print("----category: ", category, ":", score)
                if(score >= maxScore):
                    #remove the prev max score
                    if (maxScoreCategory != ""):
                        categoryValues.remove(maxScoreCategory)
                    maxScore = score
                    maxScoreCategory = category
                else:
                    categoryValues.remove(category)
            print("Category with the max score: ", maxScoreCategory, "; score=", maxScore)

        #categoryValues = [maxScoreCategory]
        return categoryValues
        #return categoryValues
    else:
        return []

for i in range(start, end):
    category = categories[i]
    host = hosts[i]

    if type(category) == str:
        category = normaliseCategories(category)
        #check if category is empty or not
        regex = re.compile(r'[\w]+')
        if re.search(regex, category):
            if host not in tfidfList:
                tfidf = createTFIDF(host)
                tfidfList[host] = tfidf
            else:
                tfidf = tfidfList[host]

            category = unquote(category)
            print(id, ". Original category: ", category)

            listOfCategories = getCategories(category)
            finalCategories = cleanCategory(listOfCategories, tfidf)

            finalCategories = ' > '.join(map(str, finalCategories))
        else:
            finalCategories = category
    else:
        finalCategories = ""

    for j in range(0, 13):
        f.write("%s;" % df[i,j])

    print("--- Processed category: ", finalCategories)
    f.write("%s\n" %finalCategories)

    id+=1

print(len(categories))

f.close()