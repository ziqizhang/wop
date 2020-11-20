#fakes iaa using cohen kappa
from sklearn.metrics import cohen_kappa_score
import pandas as pd

def read_meuselgs(lvl3col, in_csv):
    df = pd.read_csv(in_csv, header=0, delimiter=";", quoting=0, encoding="utf-8").as_matrix()
    list=[]
    count=0
    for r in df:
        count+=1
        if count<3900 and count>2000:
            continue
        if count<6800 and count>5400:
            continue
        list.append(r[lvl3col])
    return list


list1=read_meuselgs(12, "/home/zz/Work/data/wop/swc/fake_IAA/goldstandard_eng_v1_utf8.csv")
list2=read_meuselgs(12, "/home/zz/Work/data/wop/swc/fake_IAA/goldstandard_eng_v1_utf8_correction.csv")
count=0
for x, y in zip(list1, list2):
    if x!=y:
        print(count)
    count+=1

print(cohen_kappa_score(list1,list2))

