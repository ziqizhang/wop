'''
This class analyses stats of dataset (both classification and wop_matching, including
- word frequency in the corpus used to train embeddings
- the above frequency for every word found in a training dataset
- # of words (excl. digits) per instance in the training set, and as % of that instance
- # of words found in the embedding training corpus per instance, as % of that instance
'''
import re, pandas, numpy,csv
import pandas as pd
from exp import exp_util


def read_wop_matching_data(in_dir):
    # train_df = pd.read_csv(in_dir + "/small.csv")
    # valid_df = pd.read_csv(in_dir + "/small.csv")
    # test_df = pd.read_csv(in_dir + "/small.csv")
    # return train_df, valid_df, test_df
    dm_train = pd.read_csv(in_dir + "/train.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                           )
    header = list(dm_train.columns.values)
    #dm_train = dm_train.fillna('').to_numpy()

    label_col = -1
    left_start = -1
    right_start = -1
    for i in range(0, len(header)):
        h = header[i]
        if h == "label":
            label_col = i
        if h.startswith("left_") and left_start == -1:
            left_start = i
        if h.startswith("right_") and right_start == -1:
            right_start = i

    dm_validation = pd.read_csv(in_dir + "/validation.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                                )

    dm_test = pd.read_csv(in_dir + "/test.csv", header=0, delimiter=',', quoting=0, encoding="utf-8",
                          ).fillna('')

    return dm_data_to_bert_nli(dm_train, left_start, right_start, label_col), \
           dm_data_to_bert_nli(dm_validation, left_start, right_start, label_col), \
           dm_data_to_bert_nli(dm_test, left_start, right_start, label_col)


def dm_data_to_bert_nli(dataset, leftstart, rightstart, labelcol):
    rows = []
    header = ["similarity", "sentence1", "sentence2"]

    total_words=0
    max_words=0
    min_words=99999999

    dataset=dataset.values
    for r in dataset:
        label = r[labelcol]
        sent1 = ""
        for i in (range(leftstart, rightstart)):
            sent1 += str(r[i]) + " "
        sent1.strip()

        words=count_words(sent1)
        total_words+=words
        if words>max_words:
            max_words=words
        if words<min_words:
            min_words=words

        sent2 = ""
        for i in (range(rightstart, len(r))):
            sent2 += str(r[i]) + " "
        sent2.strip()

        words = count_words(sent1)
        total_words += words
        if words > max_words:
            max_words = words
        if words < min_words:
            min_words = words

        rows.append([label, sent1, sent2])

    df = pd.DataFrame(rows, columns=header)

    print("\t\t maxwords={}, minwords={}, average={}".format(max_words, min_words, total_words/(len(df)*2)))
    return df

def count_words(sent):
    return len(sent.split(" "))

'''
find words in the prod desc corpus and count freq
'''
def count_reference_freq(in_file):
    freq={}

    file = open(in_file, 'r')

    count=0
    while True:
        count += 1

        line = file.readline()
        # if line is empty
        # end of file is reached
        if not line:
            break

        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', line)
        text = re.sub(r'\W+', ' ', text).strip().lower()

        for w in text.split(" "):
            if w not in freq.keys():
                freq[w]=1
            else:
                freq[w]=freq[w]+1

        if count%10000==0:
            print(count)
        # if count==10000:
        #     break
    file.close()
    return freq

def freq_to_bins(freq:dict, bins=100):
    freq_values=list(freq.values())
    freq_values=sorted(freq_values, reverse=True)
    bin_brackets=pandas.cut(freq_values, bins, retbins=True)
    return bin_brackets[1]

'''
FOR WOP_CLASSIFICATION

returns: a list of freq for each word found in the target corpus. freq is based on looking up dictionary from
the reference freq
'''
def count_target_freq(dataset_type, reference_freq:dict,train_data_file, test_data_file, text_fields:list):
    if dataset_type=="mwpd":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonMPWD(train_data_file, test_data_file)
    elif dataset_type=="rakuten":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_csvRakuten(train_data_file, test_data_file, delimiter="\t")
    elif dataset_type=="icecat":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonIceCAT(train_data_file, test_data_file)
    else:#wdc
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonWDC(train_data_file, test_data_file)

    freq = []
    not_found = 0
    total = 0
    for row in df:
        text=""
        for c in text_fields:
            text+=row[c]+" "
        text=text.strip()
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        text = re.sub(r'\W+', ' ', text).strip().lower()

        for w in text.split(" "):
            # if w in nlp.stopwords:
            #     continue
            total+=1
            if w in reference_freq.keys():
                freq.append(reference_freq[w])
            else:
                not_found+=1

    return freq, not_found/total



'''
returns: a dictionary with #entries=bin size, then for each key (bin), the average (of all instances') %ratio between: 
the #of toks in an instance found in the reference_freq (product desc corpus), and the #of total toks in that instance
'''
def count_target_freq_per_bin(dataset_type, reference_freq:dict,train_data_file, test_data_file, text_fields:list):
    if dataset_type=="mwpd":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonMPWD(train_data_file, test_data_file)
    elif dataset_type=="rakuten":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_csvRakuten(train_data_file, test_data_file, delimiter="\t")
    elif dataset_type=="icecat":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonIceCAT(train_data_file, test_data_file)
    else:#wdc
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonWDC(train_data_file, test_data_file)

    freq = []
    not_found = 0
    total = 0
    for row in df:
        text=""
        for c in text_fields:
            text+=row[c]+" "
        text=text.strip()
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        text = re.sub(r'\W+', ' ', text).strip().lower()

        for w in text.split(" "):
            # if w in nlp.stopwords:
            #     continue
            total+=1
            if w in reference_freq.keys():
                freq.append(reference_freq[w])
            else:
                not_found+=1

    return freq, not_found/total

#note: bins will be ranked low-high
def apply_bins(data, bins):
    arr = numpy.array(data)
    dist=numpy.digitize(arr, bins, right=True)
    bcount=numpy.bincount(dist)
    return bcount


def write_to_file(bincount, outfile, nofound=-1):
    with open(outfile, 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["nofound",nofound])
        writer.writerow(["bin","freq"])
        for i in range(0, len(bincount)):
            writer.writerow(["bin"+str(i), bincount.item(i)])

# for wop_classification
#count words in each training data instance, keeping: #of words (excl. digits), #of total words (incl. digits),
# and #words found in the reference_freq, and then the percentages
def count_trainingdata_wordstats_classify(dataset_type, reference_freq:dict, train_data_file, test_data_file, text_fields:list,
                                          outfile):
    if dataset_type=="mwpd":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonMPWD(train_data_file, test_data_file)
    elif dataset_type=="rakuten":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_csvRakuten(train_data_file, test_data_file, delimiter="\t")
    elif dataset_type=="icecat":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonIceCAT(train_data_file, test_data_file)
    else:#wdc
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonWDC(train_data_file, test_data_file)

    outf= open(outfile, 'w', newline='\n')
    writer = csv.writer(outf, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["totaltoks","totalnondigitwords","totalnondigitwords_in_emc"])

    row_count=0
    ttotal_w_inproddesccorpus=0
    ttotal_w_nodigit=0
    ttotal_toks=0
    ttotal_percentage=0.0

    for row in df:
        row_count+=1
        text=""
        #merge all the text fields
        for c in text_fields:
            text+=row[c]+" "
        text=text.strip()

        #total words of the instance
        total_toks = len(text.split(" "))

        #total words with no digits
        # and total words found in the embedding training corpus
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        total_w_nodigit=0
        total_w_inproddesccorpus=0
        for w in text.split(" "):
            w = re.sub(r'\W+', ' ', w).strip().lower()
            if len(w)>0:
                total_w_nodigit+=1
                if w in reference_freq.keys():
                    total_w_inproddesccorpus+=1

        if total_toks>0:
            percent = total_w_inproddesccorpus/float(total_toks)
        writer.writerow([total_toks, total_w_nodigit, total_w_inproddesccorpus])

        ttotal_toks+=total_toks
        ttotal_w_nodigit+=total_w_nodigit
        ttotal_w_inproddesccorpus+=total_w_inproddesccorpus
        ttotal_percentage+=percent

        if row_count%1000 ==0:
            print("\trow="+str(row_count))

    print("avg toks={}, avg tok_nondigit= {}, avg tok_nondigitinemc={}, avg percentof_tok_nondigitinemc={}".
          format(ttotal_toks/row_count, ttotal_w_nodigit/row_count,
                 ttotal_w_inproddesccorpus/row_count, ttotal_percentage/row_count))
    outf.close()


# for wop_matching
#count words in each training data instance, keeping: #of words (excl. digits), #of total words (incl. digits),
# and #words found in the reference_freq, and then the percentages
def count_trainingdata_wordstats_matching(in_dir, reference_freq:dict,
                                outfile):
    train,val, test=read_wop_matching_data(in_dir)

    outf= open(outfile, 'w', newline='\n')
    writer = csv.writer(outf, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["totaltoks","totalnondigitwords","totalnondigitwords_in_emc"])

    row_count=0
    ttotal_w_inproddesccorpus=0
    ttotal_w_nodigit=0
    ttotal_toks=0
    ttotal_percentage=0.0

    df=train.values
    for row in df:
        row_count+=2
        text=(row[1]+" "+row[2]).strip()

        #total words of the instance
        total_toks = len(text.split(" "))

        #total words with no digits
        # and total words found in the embedding training corpus
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        total_w_nodigit=0
        total_w_inproddesccorpus=0
        for w in text.split(" "):
            w = re.sub(r'[^a-zA-Z]',' ',w).strip().lower()
            if len(w)>0:
                total_w_nodigit+=1
                if w in reference_freq.keys():
                    total_w_inproddesccorpus+=1

        if total_toks>0:
            percent = total_w_inproddesccorpus/float(total_toks)
        writer.writerow([total_toks, total_w_nodigit, total_w_inproddesccorpus])

        ttotal_toks+=total_toks
        ttotal_w_nodigit+=total_w_nodigit
        ttotal_w_inproddesccorpus+=total_w_inproddesccorpus
        ttotal_percentage+=percent

        if row_count%1000 ==0:
            print("\trow="+str(row_count))

    print("avg toks={}, avg tok_nondigit= {}, avg tok_nondigitinemc={}, avg percentof_tok_nondigitinemc={}".
          format(ttotal_toks/row_count, ttotal_w_nodigit/row_count,
                 ttotal_w_inproddesccorpus/row_count, ttotal_percentage/(row_count/2)))
    outf.close()

if __name__ == "__main__":
    in_file="/home/zz/Work/data/wdc/prod_desc_corpus/desc_20-250.txt"
    out_folder="/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/"

    ref_freq = count_reference_freq(in_file)
    ###########################################################
    ## the following code are used for counting word stats in training sets for wop_matching
    ###########################################################
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Amazon-Google"
    # print("s_amazon_google")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=s_amazon_google.csv")
    #
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Beer"
    # print("s_beer")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=s_beer.csv")
    #
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Fodors-Zagats"
    # print("s_fodors")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=s_fodors.csv")
    #
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/iTunes-Amazon"
    # print("s_itunes")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=s_itunes.csv")
    #
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Walmart-Amazon"
    # print("s_walmart")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=s_walmart.csv")
    #
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Textual/abt_buy_exp_data"
    # print("t_abtbuy")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=t_abtbuy.csv")
    #
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Dirty/iTunes-Amazon"
    # print("d_amazon_google")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=d_itunes.csv")
    #
    # in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Dirty/Walmart-Amazon"
    # print("s_amazon_google")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=d_walmart.csv")
    #
    # in_dir = "/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_original/all_small"
    # print("wdc small")
    # count_trainingdata_wordstats_matching(in_dir, ref_freq,
    #                                       out_folder + "wf=wdc_small.csv")
    #
    # exit(0)



    ref_freq = count_reference_freq(in_file)
    ###########################################################
    ## the following code are used for counting word stats in training sets for wop_classification
    ###########################################################
    # mwpd
    train_file = "/home/zz/Work/data/wop/swc/swc_dataset/train.json"
    test_file = None
    dataset = "mwpd"
    text_fields = [1, 2, 3]  # 1=title, 2=desc, 3=cat
    print("mwpd")
    count_trainingdata_wordstats_classify(dataset, ref_freq, train_file, test_file, text_fields, out_folder + "wf=mwpd-pc.csv")

    # wdc
    train_file = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train.json"
    test_file = None
    dataset = "wdc"
    text_fields = [1, 2, 3, 4]  # 1=title, 2=desc, 3=cat
    print("wdc")
    count_trainingdata_wordstats_classify(dataset, ref_freq, train_file, test_file, text_fields, out_folder + "wf=wdc-25.csv")

    # rakuten
    train_file = "/home/zz/Work/data/Rakuten/original/rdc-catalog-train.tsv"
    test_file = None
    dataset = "rakuten"
    text_fields = [0]  # 1=title
    print("rak")
    count_trainingdata_wordstats_classify(dataset, ref_freq, train_file, test_file, text_fields, out_folder + "wf=rakuten.csv")

    # icecat
    train_file = "/home/zz/Work/data/IceCAT/icecat_data_train.json"
    test_file = None
    dataset = "icecat"
    text_fields = [2, 3, 4]  # 1=title, 2=desc, 3=cat
    print("icecat")
    count_trainingdata_wordstats_classify(dataset, ref_freq, train_file, test_file, text_fields, out_folder + "wf=icecat.csv")

    exit(0)


    ###########################################################
    ## the following code are used for creating binning stats for wop_classification
    ###########################################################
    #mwpd
    bin_numbers = 100
    bins = freq_to_bins(ref_freq, bin_numbers)
    train_file="/home/zz/Work/data/wop/swc/swc_dataset/train.json"
    test_file=None
    dataset="mwpd"
    text_fields=[1,2,3] #1=title, 2=desc, 3=cat

    target_freq, nofound=count_target_freq(dataset, ref_freq,train_file, test_file, text_fields)
    print("not found:"+str(nofound))

    bincount=apply_bins(target_freq, bins)
    write_to_file(bincount, out_folder+"mwpd_word_freq.csv", nofound)

    # wdc
    train_file = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train.json"
    test_file = None
    dataset = "wdc"
    text_fields = [1, 2, 3,4]  # 1=title, 2=desc, 3=cat

    target_freq, nofound = count_target_freq(dataset, ref_freq, train_file, test_file, text_fields)
    print("not found:" + str(nofound))

    bincount = apply_bins(target_freq, bins)
    write_to_file(bincount, out_folder + "wdc_word_freq.csv", nofound)

    # rakuten
    train_file = "/home/zz/Work/data/Rakuten/original/rdc-catalog-train.tsv"
    test_file = None
    dataset = "rakuten"
    text_fields = [0]  # 1=title

    target_freq, nofound = count_target_freq(dataset, ref_freq, train_file, test_file, text_fields)
    print("not found:" + str(nofound))

    bincount = apply_bins(target_freq, bins)
    write_to_file(bincount, out_folder + "rakuten_word_freq.csv", nofound)

    # icecat
    train_file = "/home/zz/Work/data/IceCAT/icecat_data_train.json"
    test_file = None
    dataset = "icecat"
    text_fields = [2, 3,4]  # 1=title, 2=desc, 3=cat

    target_freq, nofound = count_target_freq(dataset, ref_freq, train_file, test_file, text_fields)
    print("not found:" + str(nofound))

    bincount = apply_bins(target_freq, bins)
    write_to_file(bincount, out_folder + "icecat_word_freq.csv", nofound)



