import collections
import csv
import os

def parse_acc(value):
    if len(value)==0:
        return value
    return value.split("=")[1].strip()

def transform_score_format_lodataset(infolder, outfile):
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')

        files=sorted(os.listdir(infolder))

        acc_outlines=collections.defaultdict(dict)
        prf_mac_outlines=collections.defaultdict(dict)
        prf_macw_outlines = collections.defaultdict(dict)
        prf_mic_outlines = collections.defaultdict(dict)

        for f in files:
            if not f.endswith(".csv"):
                continue
            print(f)

            setting=f[f.index("=")+1: f.index("|training")]
            lvl = setting.split("_")[0]
            field = setting.split("_")[1]

            acc={}
            prf_mac={}
            prf_macw={}
            prf_mic={}

            if 'predictions' in f:
                continue

            df = open(infolder+"/"+f).readlines()

            cnn_acc=[""]
            han_acc=[""]

            if 'lvl1' in f:
                bilstm_mac_row=df[39].split(",")
                bilstm_macw_row = df[40].split(",")
                bilstm_mic_row=df[41].split(",")
                bilstm_acc=df[42].split(",")

                if len(df)>83:
                    cnn_mac_row = df[83].split(",")
                    cnn_macw_row = df[84].split(",")
                    cnn_mic_row = df[85].split(",")
                    cnn_acc = df[86].split(",")

                if len(df)>127:
                    han_mac_row = df[127].split(",")
                    han_macw_row = df[128].split(",")
                    han_mic_row = df[129].split(",")
                    han_acc = df[130].split(",")

            elif 'lvl2' in f:
                bilstm_mac_row=df[78].split(",")
                bilstm_macw_row = df[79].split(",")
                bilstm_mic_row=df[80].split(",")
                bilstm_acc=df[81].split(",")

                if len(df)>161:
                    cnn_mac_row = df[161].split(",")
                    cnn_macw_row = df[162].split(",")
                    cnn_mic_row = df[163].split(",")
                    cnn_acc = df[164].split(",")

                if len(df)>244:
                    han_mac_row = df[244].split(",")
                    han_macw_row = df[245].split(",")
                    han_mic_row = df[246].split(",")
                    han_acc = df[247].split(",")
            else:
                bilstm_mac_row = df[291].split(",")
                bilstm_macw_row = df[292].split(",")
                bilstm_mic_row = df[293].split(",")
                bilstm_acc = df[294].split(",")

                if len(df)>587:
                    cnn_mac_row = df[587].split(",")
                    cnn_macw_row = df[588].split(",")
                    cnn_mic_row = df[589].split(",")
                    cnn_acc = df[590].split(",")

                if len(df)>883:
                    han_mac_row = df[883].split(",")
                    han_macw_row = df[884].split(",")
                    han_mic_row = df[885].split(",")
                    han_acc = df[886].split(",")

            # acc_ = [parse_acc(bilstm_acc[0]), parse_acc(cnn_acc[0]),
            #        parse_acc(han_acc[0])]
            # mac = [bilstm_mac_row[1], bilstm_mac_row[2], bilstm_mac_row[3],
            #        cnn_mac_row[1], cnn_mac_row[2], cnn_mac_row[3],
            #        han_mac_row[1], han_mac_row[2], han_mac_row[3]]
            # macw = [bilstm_macw_row[1], bilstm_macw_row[2], bilstm_macw_row[3],
            #         cnn_macw_row[1], cnn_macw_row[2], cnn_macw_row[3],
            #         han_macw_row[1], han_macw_row[2], han_macw_row[3]]
            # mic = [bilstm_mic_row[1], bilstm_mic_row[2], bilstm_mic_row[3],
            #        cnn_mic_row[1], cnn_mic_row[2], cnn_mic_row[3],
            #        han_mic_row[1], han_mic_row[2], han_mic_row[3]]

            # acc_ = [parse_acc(bilstm_acc[0]),
            #         parse_acc(cnn_acc[0])]
            # mac = [bilstm_mac_row[1], bilstm_mac_row[2], bilstm_mac_row[3],
            #        cnn_mac_row[1], cnn_mac_row[2], cnn_mac_row[3]]
            # macw = [bilstm_macw_row[1], bilstm_macw_row[2], bilstm_macw_row[3],
            #         cnn_macw_row[1], cnn_macw_row[2], cnn_macw_row[3]]
            # mic = [bilstm_mic_row[1], bilstm_mic_row[2], bilstm_mic_row[3],
            #        cnn_mic_row[1], cnn_mic_row[2], cnn_mic_row[3]]

            acc_ = [parse_acc(bilstm_acc[0])]
            mac = [bilstm_mac_row[1], bilstm_mac_row[2], bilstm_mac_row[3]]
            macw = [bilstm_macw_row[1], bilstm_macw_row[2], bilstm_macw_row[3]]
            mic = [bilstm_mic_row[1], bilstm_mic_row[2], bilstm_mic_row[3]]

            acc[lvl]=acc_
            prf_mac[lvl]=mac
            prf_macw[lvl]=macw
            prf_mic[lvl] = mic

            acc_outlines[field][lvl]=acc_
            prf_mac_outlines[field][lvl]=mac
            prf_macw_outlines[field][lvl]=macw
            prf_mic_outlines[field][lvl]=mic

        #write accuracy
        writer.writerow(['ACC', 'lvl1','','','lvl2','','','lvl3','',''])
        writer.writerow(['ACC', 'bilstm', 'cnn', 'han','bilstm', 'cnn', 'han', 'bilstm', 'cnn', 'han'])
        for k,v in acc_outlines.items():
            values=[k]
            for k_lvl, v_lvl in v.items():
                if 'lvl1' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                if 'lvl2' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                if 'lvl3' in k_lvl:
                    values.extend(v_lvl)
            writer.writerow(values)

        #writer prf mac
        writer.writerow(['PRF_MAC', 'lvl1','','','','','','','','',
                         'lvl2','','','','','','','','',
                         'lvl3','','','','','','','',''])
        writer.writerow(['PRF_MAC', 'bilstm', '', '', 'cnn', '', '', 'han', '', '',
                         'bilstm', '', '', 'cnn', '', '', 'han', '', '',
                         'bilstm', '', '', 'cnn', '', '', 'han', '', ''])
        for k, v in prf_mac_outlines.items():
            values = [k]
            for k_lvl, v_lvl in v.items():
                if 'lvl1' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                if 'lvl2' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                if 'lvl3' in k_lvl:
                    values.extend(v_lvl)
            writer.writerow(values)

        # writer prf mac
        writer.writerow(['PRF_MAC_W', 'lvl1', '', '', '', '', '', '', '', '',
                         'lvl2', '', '', '', '', '', '', '', '',
                         'lvl3', '', '', '', '', '', '', '', ''])
        writer.writerow(['PRF_MAC_W', 'bilstm', '', '', 'cnn', '', '', 'han', '', '',
                         'bilstm', '', '', 'cnn', '', '', 'han', '', '',
                         'bilstm', '', '', 'cnn', '', '', 'han', '', ''])
        for k, v in prf_macw_outlines.items():
            values = [k]
            for k_lvl, v_lvl in v.items():
                if 'lvl1' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                if 'lvl2' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                 if 'lvl3' in k_lvl:
                      values.extend(v_lvl)
            writer.writerow(values)

        # writer prf mac
        writer.writerow(['PRF_MIC', 'lvl1', '', '', '', '', '', '', '', '',
                         'lvl2', '', '', '', '', '', '', '', '',
                         'lvl3', '', '', '', '', '', '', '', ''])
        writer.writerow(['PRF_MIC', 'bilstm', '', '', 'cnn', '', '', 'han', '', '',
                         'bilstm', '', '', 'cnn', '', '', 'han', '', '',
                         'bilstm', '', '', 'cnn', '', '', 'han', '', ''])
        for k, v in prf_mic_outlines.items():
            values = [k]
            for k_lvl, v_lvl in v.items():
                if 'lvl1' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                if 'lvl2' in k_lvl:
                    values.extend(v_lvl)
            for k_lvl, v_lvl in v.items():
                if 'lvl3' in k_lvl:
                    values.extend(v_lvl)
            writer.writerow(values)


if __name__ == "__main__":
    # transform_score_format_lodataset("/home/zz/Work/wop/tmp/classifier_with_desc",
    #                                   "/home/zz/Work/wop/tmp/desc.csv")
    transform_score_format_lodataset("/home/zz/Work/wop/output/classifier/tmp",
                                     "/home/zz/Work/wop/output/classifier/run_ft_nfold_none.csv")

    # transform_score_format_lodataset("/home/zz/Work/wop/tmp/classifier_with_desc",
    #                                  "/home/zz/Work/wop/output/classifier/dnn_d_X_result.csv")
