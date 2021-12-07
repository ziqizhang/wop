'''
This uses the data_stats_analyser.py file and creates the data needed to create the heatmap in the paper
'''
from exp import data_stats_analyser as dsa
from exp import exp_util
import re, csv,numpy

'''
this is for wop classification dataset

returns: a dictionary with #entries=bin size, then for each key (bin), the average (of all instances') %ratio between: 
the #of toks in an instance found in the reference_freq (product desc corpus), and the #of total toks in that instance
'''
def count_data_freq_per_bin(df, reference_word2bin:dict, outfile, dataset, totalbins):

    count_dataset_bin_percents={}
    total_instances=0
    for row in df:
        total_instances+=1
        text=row.strip()
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        text = re.sub(r'\W+', ' ', text).strip().lower()

        instance_wtotal = 0
        instance_bin_count={}

        #checking for each word in the instance, which bin it belongs to
        for w in text.split(" "):
            # if w in nlp.stopwords:
            #     continue
            instance_wtotal+=1
            if w in reference_word2bin.keys():
                bin = reference_word2bin[w]
                if bin in instance_bin_count.keys():
                    instance_bin_count[bin]+=1
                else:
                    instance_bin_count[bin]=1

        #calculating %
        for k in instance_bin_count.keys():
            v = instance_bin_count[k]
            if instance_wtotal>0:
                percent = v/instance_wtotal
            else:
                percent=0

            instance_bin_count[k]=percent

        #update for the entire dataset counter
        for k, v in instance_bin_count.items():
            if k in count_dataset_bin_percents.keys():
                count_dataset_bin_percents[k]+=v
            else:
                count_dataset_bin_percents[k]=v

    #gone through all instances now, let's calculate average for each bin
    outf = open(outfile, 'a', newline='\n')
    writer = csv.writer(outf, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_ALL)


    for b in range(0, totalbins):
        b=b+1
        if b not in count_dataset_bin_percents.keys():
            avg=0
        else:
            percent = count_dataset_bin_percents[b]
            avg = percent/total_instances
        writer.writerow([dataset,b,avg])
    outf.close()


def load_data_classification(dataset_type, train_data_file, test_data_file, text_fields:list
                                          ):
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

    data=[]

    for row in df:
        text=""
        #merge all the text fields
        for c in text_fields:
            text+=row[c]+" "
        text=text.strip()

        data.append(text)

    return data

def load_data_matching(in_dir):
    train,val, test=dsa.read_wop_matching_data(in_dir)

    df=train.values
    data=[]
    for row in df:
        text=(row[1]+" "+row[2]).strip()
        data.append(text)
    return data


if __name__ == "__main__":
    in_file = "/home/zz/Work/data/wdc/prod_desc_corpus/desc_20-250.txt"
    out_file = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/wordfreq/heatmap.csv"
    totalbins=100
    ref_freq_lookup = dsa.count_reference_freq(in_file)
    bins = dsa.freq_to_bins(ref_freq_lookup, totalbins)

    ref_freq_freqonly=list(ref_freq_lookup.values())
    arr = numpy.array(ref_freq_freqonly)
    ref_bin_number = numpy.digitize(arr, bins, right=True)
    ref_bin_lookup = {}
    index=0
    for k in ref_freq_lookup.keys():
        ref_bin_lookup[k] = ref_bin_number[index]
        index+=1

    #classification
    print("mwpd")
    train_file = "/home/zz/Work/data/wop/swc/swc_dataset/train.json"
    test_file = None
    dataset = "MWPD-PC"
    text_fields = [1, 2, 3]  # 1=title, 2=desc, 3=cat
    mwpd_data = load_data_classification("mwpd", train_file, test_file, text_fields)
    count_data_freq_per_bin(mwpd_data, ref_bin_lookup, out_file, dataset, totalbins)

    #wdc
    print("wdc")
    train_file = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train.json"
    test_file = None
    dataset = "WDC-25"
    text_fields = [1, 2, 3, 4]
    wdc_data = load_data_classification("wdc", train_file, test_file, text_fields)
    count_data_freq_per_bin(wdc_data, ref_bin_lookup, out_file, dataset, totalbins)

    # rakuten
    print("rakuten")
    train_file = "/home/zz/Work/data/Rakuten/original/rdc-catalog-train.tsv"
    test_file = None
    dataset = "Rakuten"
    text_fields = [0]  # 1=title
    rak_data = load_data_classification("rakuten", train_file, test_file, text_fields)
    count_data_freq_per_bin(rak_data, ref_bin_lookup, out_file, dataset, totalbins)

    # icecat
    print("icecat")
    train_file = "/home/zz/Work/data/IceCAT/icecat_data_train.json"
    test_file = None
    dataset = "IceCat"
    text_fields = [2, 3, 4]  # 1=title, 2=desc, 3=cat
    icecat_data = load_data_classification("icecat", train_file, test_file, text_fields)
    count_data_freq_per_bin(icecat_data, ref_bin_lookup, out_file, dataset, totalbins)


    #matching
    in_dir = "/home/zz/Work/data/wdc-lspc/dm_wdclspc_small_original/all_small"
    print("wdc small")
    dataset = "WDC-small)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset, totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Beer"
    print("BeerAdvo-RateBeer (S)")
    dataset = "BeerAdvo-RateBeer (S)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/iTunes-Amazon"
    print("iTunes-Amazon1 (S)")
    dataset = "iTunes-Amazon1 (S)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Fodors-Zagats"
    print("Fodors-Zagats")
    dataset = "Fodors-Zagats (S)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Amazon-Google"
    print("Amazon-Google (S)")
    dataset = "Amazon-Google (S)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Structured/Walmart-Amazon"
    print("Walmart-Amazon1 (S)")
    dataset = "Walmart-Amazon1 (S)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Textual/abt_buy_exp_data"
    print("Abt-Buy (T)")
    dataset = "Abt-Buy (T)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Dirty/iTunes-Amazon"
    dataset = "iTunes-Amazon2 (D)"
    dataset = "iTunes-Amazon2 (D)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    in_dir = "/home/zz/Work/data/entity_linking/deepmatcher/processed/Dirty/Walmart-Amazon"
    print("Walmart-Amazon2 (S)")
    dataset = "Walmart-Amazon2 (S)"
    data = load_data_matching(in_dir)
    count_data_freq_per_bin(data, ref_bin_lookup, out_file, dataset,totalbins)

    print("done")