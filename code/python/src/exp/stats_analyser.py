'''
This class analyses stats of dataset, including
- word frequency in the corpus used to train embeddings
- the above frequency for every word found in a training dataset
'''
import re, pandas, numpy,csv
from exp import exp_util

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



if __name__ == "__main__":
    in_file="/home/zz/Work/data/embeddings/wop/training_corpus/desc_20-250.txt"
    out_folder="/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/"
    bin_numbers=100
    ref_freq = count_reference_freq(in_file)
    bins = freq_to_bins(ref_freq, bin_numbers)


    #mwpd
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

