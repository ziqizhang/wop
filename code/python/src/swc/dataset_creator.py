'''
This takes the original csv-convereted-json (8361 instances) and the newly created extended-json,
and create train, val, test sets

The test set will only contain those from extended-json
the train and validation contains both csv-convereted-json and extended-json
'''
from sklearn.model_selection import train_test_split
import numpy as np
import textwrap

from swc import data_io as di

def split_data(original_as_json, extended_json, class_col, out_folder):
    original_data=di.read_json(original_as_json)
    extended_data=di.read_json(extended_json)

    #set product ids across the two datasets
    pid=0
    for r in original_data:
        r[0]=str(pid)
        desc=r[2]
        if len(desc)>500:
            desc=textwrap.shorten(desc,500)
            r[2]=desc
        pid+=1

    Xe_id=0
    Xe_index = []
    extended_data = np.array([np.array(xi) for xi in extended_data])
    for r in extended_data:
        r[0]=str(pid)
        desc = r[2]
        if len(desc) > 5000:
            desc = textwrap.shorten(desc, 5000)
            r[2] = desc
        Xe_index.append(Xe_id)
        pid+=1
        Xe_id+=1

    #split the extended set to test set and the remainder
    ye = extended_data[:, class_col]
    Xe_A, Xe_test, ye_A, ye_test = train_test_split(Xe_index, ye, test_size = 0.40, random_state = 42)
    test=repack_data(Xe_test, extended_data)

    #add the remainder to the original set
    for i in Xe_A:
        data = extended_data[i]
        original_data.append(data)

    Xo_id = 0
    Xo_index = []
    yo=[]
    for r in original_data:
        yo.append(r[7])
        Xo_index.append(Xo_id)
        Xo_id += 1

    #split the merged orginal set
    X_train, X_val, y_train, y_val = train_test_split(Xo_index, yo, test_size=0.231, random_state=42)

    train = repack_data(X_train, original_data)
    val=repack_data(X_val, original_data)
    di.write_json(train,out_folder+"/train.json")
    di.write_json(val, out_folder + "/validation.json")
    di.write_json(test, out_folder + "/test.json")


def repack_data(X_index, raw_data):
    res = []
    for i in X_index:
        r = raw_data[i]
        res.append(r)
    return res


if __name__ == "__main__":
    original_json = "/home/zz/Work/data/wop/swc/swc_dataset/original.json"
    extended_json = "/home/zz/Work/data/wop/swc/swc_dataset/extended.json"
    class_col=7
    out_folder = "/home/zz/Work/data/wop/swc/swc_dataset"
    split_data(original_json, extended_json,class_col, out_folder)