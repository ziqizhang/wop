import csv
from feature import nlp


def create_disease_dictionary(csv_hashtag_file, outfolder):
    word_to_disease={}
    hashtag_to_disease={}
    with open(csv_hashtag_file, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in csvreader:
            disease=row[0]
            hashtag_to_disease[disease]=disease
            #create word to disease
            hashtag_text = nlp.normalize_tweet(disease)
            update_word_to_disease(word_to_disease,hashtag_text, disease)

            for i in range(1, len(row)):
                values = row[i]
                for v in values.split("|"):
                    if len(v)<2:
                        continue
                    hashtag_to_disease[v] = disease
                    #create word to disease
                    hashtag_text = nlp.normalize_tweet(disease)
                    update_word_to_disease(word_to_disease,hashtag_text, disease)
    with open(outfolder+"/dictionary_hashtag_disease.csv", 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for k,v in hashtag_to_disease.items():
            csvwriter.writerow([k, v])
    with open(outfolder+"/dictionary_word_disease.csv", 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for k,vs in word_to_disease.items():
            values=[k]
            for v in vs:
                values.append(v)
            csvwriter.writerow(list(values))

def update_word_to_disease(word_to_disease:dict, hashtag_text, disease):
    disease_strip_hashtag=disease[1:].lower()
    for w in hashtag_text.split(" "):
        w=w.lower()
        if len(w)>3:
            if w in word_to_disease.keys():
                diseases=word_to_disease[w]
                diseases.add(disease)
            else:
                diseases=set()
                diseases.add(disease)

            word_to_disease[w] = diseases

    diseases=set()
    diseases.add(disease)
    word_to_disease[disease_strip_hashtag]=diseases

def load_disease_hashtag_dictionary(csvfile):
    out_dict={}
    with open(csvfile, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            word = row[0].lower()
            diseases=row[1:]
            diseases = list(filter(None, diseases))
            out_dict[word]=diseases
    return out_dict

if __name__=="__main__":
    create_disease_dictionary(
        "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/2_PART2_processed_hashtags.tsv",
        "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/stakeholder_classification/dictionary_feature/hashtag_dict")