import sklearn
from sklearn.metrics import cohen_kappa_score

lookup={}
lookup["Advocates"]=0
lookup["Patient"]=1
lookup["P"]=1
lookup["HPO"]=2
lookup["HPI"]=3
lookup["Other"]=4
lookup["Research"]=5


def read_annotations(in_csv, num_lines:int, ignore_header=True):
    converted_labels=[]
    with open(in_csv, 'r') as f:
        lines = f.readlines()
        for i in range(0, num_lines+1):
            if ignore_header and i==0:
                continue
            l = lines[i].replace('"','').strip()

            part=l.split(",")
            labels=[]
            for x in range(1, len(part)):
                p = part[x]
                if len(p)==0:
                    continue
                else:
                    labels.append(lookup[p.strip()])
            labels=sorted(labels, reverse=True)
            try:
                converted_labels.append(labels)
            except KeyError:
                print("error")
    return converted_labels

def maximize_agreement(annotator1:list, annotator2:list):
    for i in range(0, len(annotator1)):
        ann1 = annotator1[i]
        ann2 = annotator2[i]
        if len(ann1)>1 or len(ann2)>1:
            inter = set(ann1) & set(ann2)
            if len(inter)>0:
                annotator1[i]=list(inter)[0]
                annotator2[i] = list(inter)[0]
            else:
                annotator1[i] = ann1[0]
                annotator2[i] = ann2[0]
        else:
            annotator1[i]=ann1[0]
            annotator2[i]=ann2[0]

if __name__=="__main__":
    annotator1 = \
        read_annotations("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/annotation/GB_annotation.csv",100)
    annotator2 = \
        read_annotations("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/annotation/ZZ_annotation.csv",100)

    maximize_agreement(annotator1,annotator2)
    print(cohen_kappa_score(annotator1, annotator2,
                                      labels=None, weights=None))