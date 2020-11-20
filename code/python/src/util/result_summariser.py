import glob
import csv
import os

def parse_acc(value):
    if len(value)==0:
        return value
    acc=value.split("=")[1].strip()
    if ',' in acc:
        acc=acc[0:acc.index(',')]
    return acc.strip()

#results: expected multi-level dict
#lvl1 keys: gs level
#lvl2 keys: algorithm
def write_header(results:dict, csvwr:csv.writer, values:int):
    row=[""]
    for lvl, algs in results.items():
        for alg in algs.keys():
            header=[lvl+"/"+alg]
            for x in range(0, values-1):
                header.append("")
            row.extend(header)

    csvwr.writerow(row)

#use this for the DNN (cnn,lstm,han) results and fasttext results and ...
#NOT to be used by CML (svm etc because the format is different)
def summarise(infolder, outfile):
    print("Summarusing features and levels...")
    files = sorted(os.listdir(infolder))
    levels_=set()
    features_=set()
    for f in files:
        if not f.endswith(".csv"):
            continue
        if f.startswith("prediction"):
            continue
        if "=" in f:
            setting = f[f.index("=") + 1:]
        else:
            setting = f[f.index("-") + 1:f.index(".txt.csv")]

        lvl = setting.split("_")[0]
        feature=setting.split("_",1)[1]
        levels_.add(lvl)
        features_.add(feature)
    levels_=sorted(list(levels_))
    features_=sorted(list(features_))


    print("Creating empty result maps...")
    acc_ = {}
    micro_ = {}
    macro_ = {}
    wmacro_ = {}
    for f in features_:
        f_levels={}
        for l in levels_:
            f_levels[l]={}
        acc_[f]=f_levels
    for f in features_:
        f_levels={}
        for l in levels_:
            f_levels[l]={}
        micro_[f] = f_levels
    for f in features_:
        f_levels = {}
        for l in levels_:
            f_levels[l] = {}
        macro_[f] = f_levels
    for f in features_:
        f_levels = {}
        for l in levels_:
            f_levels[l] = {}
        wmacro_[f] = f_levels

    print("Collecting results...")
    files = sorted(os.listdir(infolder))
    for f in files:
        if not f.endswith(".csv"):
            continue
        if "=" in f:
            setting = f[f.index("=") + 1:]
        else:
            setting = f[f.index("-") + 1:f.index(".txt.csv")]
        lvl = setting.split("_")[0]
        feature = setting.split("_",1)[1]

        df = open(infolder + "/" + f).readlines()

        prev_alg= None
        curr_alg= None
        alg_counter = 0
        for row in df:
            row=row.strip()
            if 'results' in row: #start of a new algorithm, init containers
                curr_alg=row.split(",")[0]
                if prev_alg is None:
                    prev_alg=curr_alg
                elif prev_alg==curr_alg:
                    curr_alg=curr_alg+"_"+str(alg_counter)
                    alg_counter+=1

                acc_[feature][lvl][curr_alg]=[]
                micro_[feature][lvl][curr_alg] = []
                macro_[feature][lvl][curr_alg] = []
                wmacro_[feature][lvl][curr_alg] = []

            elif 'mac avg' in row: #macro
                stats=row.split(",")[1:-1] #-1 because the last number is the support
                macro_[feature][lvl][curr_alg].extend(stats)
            elif 'macro avg w' in row: #weighted macro
                stats = row.split(",")[1:-1]
                wmacro_[feature][lvl][curr_alg].extend(stats)
            elif 'micro avg' in row:
                stats = row.split(",")[1:-1]
                micro_[feature][lvl][curr_alg].extend(stats)
            elif 'accuracy' in row: #end of a new algorithm
                acc=parse_acc(row)
                acc_[feature][lvl][curr_alg].append(acc)
            else:
                continue

    print("Output results...")
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerow(["Acc"])

        has_header=False
        for fea, results in acc_.items():
            if not has_header:
                write_header(results, writer,1)
                has_header=True
            #navigate into the multi-level dict structure to output result
            row=[fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["Macro"])
        has_header = False
        for fea, results in macro_.items():
            if not has_header:
                write_header(results, writer,3)
                has_header=True
            #navigate into the multi-level dict structure to output result
            row=[fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["W Macro"])
        has_header = False
        for fea, results in wmacro_.items():
            if not has_header:
                write_header(results, writer,3)
                has_header=True
            #navigate into the multi-level dict structure to output result
            row=[fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)


        writer.writerow([""])
        writer.writerow(["Micro"])
        has_header = False
        for fea, results in micro_.items():
            if not has_header:
                write_header(results, writer,3)
                has_header=True
            #navigate into the multi-level dict structure to output result
            row=[fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)


#Use this for the CML (svm etc because the format is different) results
def summarise_cml(infolder, outfile):
    print("Summarusing features and levels...")
    files = sorted(os.listdir(infolder))
    levels_ = set()
    features_ = set()

    #cml results are written in different way, the embedding indicators are written into the result file. So we
    #need to read in at least one file to discover the embedding settings used
    embeddings=set()
    for f in files:
        if not f.endswith(".csv"):
            continue
        df = open(infolder + "/" + f).readlines()

        for row in df:
            row = row.strip()
            #svm_l|glove.840B.300d.bin.gensimN-fold results:
            if 'results' in row:  # start of a new algorithm, init containers
                emb=row.split("|")[1]
                emb=emb[0:emb.index("results")].strip()
                embeddings.add(emb)

        break
    embeddings=sorted(list(embeddings))

    for f in files:
        if not f.endswith(".csv"):
            continue
        if "=" in f:
            setting = f[f.index("=") + 1:]
        else:
            setting = f[f.index("-") + 1:f.index(".txt.csv")]

        lvl = setting.split("_")[0]
        feature = setting.split("_", 1)[1]
        levels_.add(lvl)
        for emb in embeddings:
            features_.add(feature+"|"+emb)
    levels_ = sorted(list(levels_))
    features_ = sorted(list(features_))

    print("Creating empty result maps...")
    acc_ = {}
    micro_ = {}
    macro_ = {}
    wmacro_ = {}
    for f in features_:
        f_levels = {}
        for l in levels_:
            f_levels[l] = {}
        acc_[f] = f_levels
    for f in features_:
        f_levels = {}
        for l in levels_:
            f_levels[l] = {}
        micro_[f] = f_levels
    for f in features_:
        f_levels = {}
        for l in levels_:
            f_levels[l] = {}
        macro_[f] = f_levels
    for f in features_:
        f_levels = {}
        for l in levels_:
            f_levels[l] = {}
        wmacro_[f] = f_levels

    print("Collecting results...")
    files = sorted(os.listdir(infolder))
    for f in files:
        if not f.endswith(".csv"):
            continue
        if "=" in f:
            setting = f[f.index("=") + 1:]
        else:
            setting = f[f.index("-") + 1:f.index(".txt.csv")]
        lvl = setting.split("_")[0]
        feature_field = setting.split("_", 1)[1]

        df = open(infolder + "/" + f).readlines()

        curr_alg = None
        for row in df:
            row = row.strip()
            if 'results' in row:  # start of a new algorithm, init containers
                line=row.split(",")[0]
                curr_alg = line.split("|")[0]

                fea= line[line.index("|")+1:line.index("results")].strip()
                feature=feature_field+"|"+fea

                acc_[feature][lvl][curr_alg] = []
                micro_[feature][lvl][curr_alg] = []
                macro_[feature][lvl][curr_alg] = []
                wmacro_[feature][lvl][curr_alg] = []

            elif 'mac avg' in row:  # macro
                stats = row.split(",")[1:-1]  # -1 because the last number is the support
                macro_[feature][lvl][curr_alg].extend(stats)
            elif 'macro avg w' in row:  # weighted macro
                stats = row.split(",")[1:-1]
                wmacro_[feature][lvl][curr_alg].extend(stats)
            elif 'micro avg' in row:
                stats = row.split(",")[1:-1]
                micro_[feature][lvl][curr_alg].extend(stats)
            elif 'accuracy' in row:  # end of a new algorithm
                acc = parse_acc(row)
                acc_[feature][lvl][curr_alg].append(acc)
            else:
                continue

    print("Output results...")
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        writer.writerow(["Acc"])

        has_header = False
        for fea, results in acc_.items():
            if not has_header:
                write_header(results, writer, 1)
                has_header = True
            # navigate into the multi-level dict structure to output result
            row = [fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["Macro"])
        has_header = False
        for fea, results in macro_.items():
            if not has_header:
                write_header(results, writer, 3)
                has_header = True
            # navigate into the multi-level dict structure to output result
            row = [fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["W Macro"])
        has_header = False
        for fea, results in wmacro_.items():
            if not has_header:
                write_header(results, writer, 3)
                has_header = True
            # navigate into the multi-level dict structure to output result
            row = [fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)

        writer.writerow([""])
        writer.writerow(["Micro"])
        has_header = False
        for fea, results in micro_.items():
            if not has_header:
                write_header(results, writer, 3)
                has_header = True
            # navigate into the multi-level dict structure to output result
            row = [fea]
            for lvl, alg in results.items():
                for a, scores in alg.items():
                    row.extend(scores)
            writer.writerow(row)


if __name__ == "__main__":

    # transform_score_format_lodataset("/home/zz/Work/wop/tmp/classifier_with_desc",
    #                                   "/home/zz/Work/wop/tmp/desc.csv")
    # summarise("/home/zz/Work/wop/output/cml+dnn_mwpd_val/classifier/scores",
    #                                  "/home/zz/Work/wop/output/cml+dnn_mwpd_val/cml_mwpd_val.csv")

    # transform_score_format_lodataset("/home/zz/Work/wop/tmp/classifier_with_desc",
    #                                  "/home/zz/Work/wop/output/classifier/dnn_d_X_result.csv")

    # summarise("/home/zz/Work/wop/output/classifier/dnn_icecat-test-missing/output/classifier",
    #               "/home/zz/Work/wop/output/classifier/scores.csv")

    # summarise_cml("/home/zz/Work/wop/output/classifier/scores",
    #               "/home/zz/Work/wop/output/classifier/cml_wdc-missed.csv")

    input="/home/zz/Work/wop/output/classifier"
    settings_folder=os.listdir(input)

    for s in settings_folder:
        if os.path.isfile(input+"/"+s):
            continue
        files = glob.glob(input+"/"+s + '/**/*.csv', recursive=True)
        f =files[0]
        result_folder=os.path.dirname(f)
        out_file=input+"/"+s+".csv"
        summarise(result_folder, out_file)


