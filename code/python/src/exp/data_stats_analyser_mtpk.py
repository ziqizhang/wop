'''
This file analyses mt-based product keywords

The following is calculated

- boxplot of word count per instance, in rakuten 'name' and the cat-label training data 'name'
- box plot of word count per instance, in rakuten 'label' and the cat-label training data 'label'
- unique words in rakuten name, label, and that in cat-label training data
'''

from exp import exp_util
import re,os,csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def normalise(string):
    text = string.strip()
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub(r'\W+', ' ', text).strip().lower()
    return text.strip()

def load_rakuten_data(train_data_file, test_data_file):
    df, train_size, test_size = exp_util. \
        load_and_merge_train_test_csvRakuten(train_data_file, test_data_file, delimiter="\t")

    unique_name_words=set()
    unique_label_words=set()

    distribution_name_words=[]
    distribution_label_words=[]

    count=0
    for row in df:
        count+=1
        n = normalise(row[0])
        l = normalise(row[1])

        nwords = n.split(" ")
        unique_name_words.update(nwords)
        lwords = l.split(" ")
        unique_label_words.update(lwords)

        distribution_name_words.append(len(nwords))
        distribution_label_words.append(len(lwords))
        if count%10000==0:
            print("\t {}".format(count))
            #break

    return len(unique_name_words), len(unique_label_words), distribution_name_words, distribution_label_words

def load_catlabel_mt_traindata(in_dir):
    unique_name_words = set()
    unique_label_words = set()

    distribution_name_words = []
    distribution_label_words = []

    count = 0
    for f in os.listdir(in_dir):
        print(f)

        df = pd.read_csv(in_dir+f, header=-1, delimiter=",", quoting=0, encoding="utf-8",
                            ).fillna('').as_matrix()

        for row in df:
            count+=1
            n = normalise(row[0])
            l = normalise(row[1])

            nwords = n.split(" ")
            unique_name_words.update(nwords)
            lwords = l.split(" ")
            unique_label_words.update(lwords)

            distribution_name_words.append(len(nwords))
            distribution_label_words.append(len(lwords))
            if count % 10000 == 0:
                print("\t {}".format(count))
                #break

    print(count)
    return len(unique_name_words), len(unique_label_words), distribution_name_words, distribution_label_words

#calc unique words in the mwpd dataset's 'category' metadata and total words
def load_mwpd_catwordfreq(train_data_file, test_data_file, idx_cat=3):
    df, train_size, test_size = exp_util. \
        load_and_merge_train_test_data_jsonMPWD(train_data_file, test_data_file)

    label_word_freq={}

    total_words=0
    count=0
    for row in df:
        count+=1
        l = normalise(row[idx_cat])

        lwords = l.split(" ")
        total_words+=len(lwords)
        for w in lwords:
            if w in label_word_freq.keys():
                label_word_freq[w] = label_word_freq[w] + 1
            else:
                label_word_freq[w] = 1
        if count%10000==0:
            print("\t {}".format(count))
            #break

    print("mwpd unique={}, total={}".format(len(label_word_freq), total_words))
    return list(label_word_freq.values())

#calc unique words in the mwpd dataset's mt translated 'category' metadata and total words
def load_mttrain_catwordfreq(in_dir):
    label_word_freq={}

    count = 0
    total_words=0

    for f in os.listdir(in_dir):
        print(f)

        df = pd.read_csv(in_dir+f, header=-1, delimiter=",", quoting=0, encoding="utf-8",
                            ).fillna('').as_matrix()

        for row in df:
            count+=1
            l = normalise(row[1])
            lwords = l.split(" ")
            total_words+=len(lwords)
            for w in lwords:
                if w in label_word_freq.keys():
                    label_word_freq[w]=label_word_freq[w]+1
                else:
                    label_word_freq[w]=1
            if count % 10000 == 0:
                print("\t {}".format(count))
                #break
    print("mttrain unique={}, total={}".format(len(label_word_freq), total_words))
    print(count)
    return list(label_word_freq.values())


def analyse_mwpd_namecat_words(train_data_file, test_data_file, out_file, idx_cat=3, idx_name=1):
    df, train_size, test_size = exp_util. \
        load_and_merge_train_test_data_jsonMPWD(train_data_file, test_data_file)
    outf = open(out_file, 'w', newline='\n')
    writer = csv.writer(outf, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["Class","Level","NameWords","CatWords","Ratio"])


    class_lvl={}
    class_unique_namewords={}
    class_unique_catwords={}

    count=0
    for row in df:

        cls1 = row[5]
        cls2=row[6]
        cls3=row[7]
        class_lvl[cls1]=1
        class_lvl[cls2]=2
        class_lvl[cls3]=3
        name = normalise(row[idx_name])
        namewords=name.split(" ")

        count+=1
        cat = normalise(row[idx_cat])
        catwords=cat.split(" ")

        for cls in [cls1,cls2, cls3]:
            if cls in class_unique_namewords.keys():
                class_unique_namewords[cls].update(namewords)
            else:
                class_unique_namewords[cls]=set(namewords)

            if cls in class_unique_catwords.keys():
                class_unique_catwords[cls].update(catwords)
            else:
                class_unique_catwords[cls]=set(catwords)

        if count%10000==0:
            print("\t {}".format(count))
            #break

    for k, v in class_lvl.items():
        nwords=class_unique_namewords[k]
        cwords=class_unique_catwords[k]
        row=[k, v, len(nwords),len(cwords), len(nwords)/len(cwords)]
        writer.writerow(row)
    outf.close()


def merge_to_R(mwpd_name_cat_uniquewordsratio_original, mwpd_name_cat_uniquewordsratio_mtcat,
               mwpd_name_cat_uniquewordsratio_R):
    original=pd.read_csv(mwpd_name_cat_uniquewordsratio_original, header=0, delimiter=",", quoting=0, encoding="utf-8",
                ).fillna('').as_matrix()
    mtcat=pd.read_csv(mwpd_name_cat_uniquewordsratio_mtcat, header=0, delimiter=",", quoting=0, encoding="utf-8",
                ).fillna('').as_matrix()
    outf = open(mwpd_name_cat_uniquewordsratio_R, 'w', newline='\n')
    writer = csv.writer(outf, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["Class", "Ratio"])

    lvl_sum={}
    lvl_count={}

    for row in original:
        lvl=row[1]
        cls = str(lvl)+"_original"
        ratio=row[4]
        writer.writerow([cls, ratio])

        if lvl in lvl_sum.keys():
            lvl_sum[lvl]+=ratio
        else:
            lvl_sum[lvl]=ratio

        if lvl in lvl_count.keys():
            lvl_count[lvl]+=1
        else:
            lvl_count[lvl]=1

    for k, v in lvl_sum.items():
        lvl=k
        sum=v
        count=lvl_count[k]
        print("original, lvl {} avg={}".format(lvl, sum/count))

    lvl_sum.clear()
    lvl_count.clear()
    for row in mtcat:
        lvl=row[1]
        cls = str(lvl)+"_mtcat"
        ratio=row[4]
        writer.writerow([cls, ratio])

        if lvl in lvl_sum.keys():
            lvl_sum[lvl] += ratio
        else:
            lvl_sum[lvl] = ratio

        if lvl in lvl_count.keys():
            lvl_count[lvl] += 1
        else:
            lvl_count[lvl] = 1

    for k, v in lvl_sum.items():
        lvl=k
        sum=v
        count=lvl_count[k]
        print("mtcat, lvl {} avg={}".format(lvl, sum/count))

    outf.close()


def print_example_mtwords(infile, outfile):
    df, train_size, test_size = exp_util. \
        load_and_merge_train_test_data_jsonMPWD(infile, None)
    outf = open(outfile, 'w', newline='\n')
    writer = csv.writer(outf, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["name", "site-specific cat","mtpk"])
    for row in df:
        name=row[1]
        sitecat=row[3]
        pk=row[4]
        writer.writerow([name, sitecat, pk])

    outf.close()


if __name__ == "__main__":

    ###########################################################
    # the following code creates box plot for rakuten and mt training data product names and labels
    ###########################################################
    # mt_nwords, mt_nlabel, mt_dist_namewords, mt_dist_labelwords = \
    #     load_catlabel_mt_traindata("/home/zz/Work/data/wop_data/mt/product/name_cat_v4/cat_label_words/")
    #
    # rak_nwords, rak_nlabel, rak_dist_namewords, rak_dist_labelwords = \
    #     load_rakuten_data("/home/zz/Work/data/Rakuten/original/rdc-catalog-train.tsv", None)
    #
    # data = [rak_dist_namewords, mt_dist_namewords,
    #         rak_dist_labelwords, mt_dist_labelwords]
    #
    # fig7, ax7 = plt.subplots()
    # ax7.set_title('Multiple Samples with Different sizes')
    # ax7.boxplot(data)
    # # show plot
    # plt.xticks([1, 2, 3,4], ['Rakuten - product name', 'MT training data - product name', 'Rakuten - product class','MT training data - product class'])
    # plt.show()
    #
    # print("unique words: rak name={}, mt name={}, rak label={}, mt label={}".format(rak_nwords,
    #                                                                                 mt_nwords,
    #                                                                                 rak_nlabel,
    #                                                                                 mt_nlabel))
    # print("done")

    ###########################################################
    # the following code creates box plot for mwpd-pc and mt training data product category labels
    ###########################################################

    # mwpd_word_freq_dist=\
    #     load_mwpd_catwordfreq("/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train.json", None)
    #
    # # mttrain_word_freq_dist = \
    # #     load_mttrain_catwordfreq("/home/zz/Work/data/wop_data/mt/product/name_cat_v4/cat_label_words/")
    #
    # mwpd_mt_word_freq_dist = \
    #     load_mwpd_catwordfreq("/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train_mtcat.json", None, idx_cat=4)
    #
    # data = [mwpd_word_freq_dist, mwpd_mt_word_freq_dist
    #         ]
    #
    # fig7, ax7 = plt.subplots()
    # ax7.set_title('Multiple Samples with Different sizes')
    # ax7.boxplot(data)
    # # show plot
    # plt.xticks([1, 2], ['MWPD-PC', 'MT training data'])
    # plt.show()

    ##########################################################
    #  the following code collects stats about mwpd dataset's category and name words stats
    ###########################################################

    # analyse_mwpd_namecat_words("/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train.json", None,
    #                                "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/mt_keywords/mwpd_original_name-cat.csv")
    # analyse_mwpd_namecat_words("/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train_mtcat.json", None,
    #                            "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/mt_keywords/mwpd_mtcat_name-cat.csv",idx_cat=4)

    # merge_to_R("/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/mt_keywords/mwpd_original_name-cat.csv",
    #            "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/mt_keywords/mwpd_mtcat_name-cat.csv",
    #            "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/mt_keywords/mwpd_R.csv")

    print_example_mtwords("/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train_mtcat.json",
                          "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/stats/mt_keywords/name_sitecat_mt(mpwd_train).csv")


    print("done")