import pandas as pd
import sys
import csv

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], header=0, delimiter=";", quoting=0, encoding="utf-8").as_matrix()

    lvl1_col=10
    lvl2_col=11
    lvl3_col=12
    cluster_col=13

    lvl1_entropy=dict()
    lvl2_entropy=dict()
    lvl3_entropy=dict()

    for row in df:
        lvl1 = row[lvl1_col]
        lvl2 = row[lvl2_col]
        lvl3 = row[lvl3_col]
        cluster=row[cluster_col]

        if cluster in lvl1_entropy.keys():
            lvl1_entropy[cluster].add(lvl1)
        else:
            labels=set()
            labels.add(lvl1)
            lvl1_entropy[cluster]=labels

        if cluster in lvl2_entropy.keys():
            lvl2_entropy[cluster].add(lvl2)
        else:
            labels=set()
            labels.add(lvl2)
            lvl2_entropy[cluster]=labels

        if cluster in lvl3_entropy.keys():
            lvl3_entropy[cluster].add(lvl3)
        else:
            labels=set()
            labels.add(lvl3)
            lvl3_entropy[cluster]=labels


    with open(sys.argv[2], mode='w') as employee_file:
        csv_writer = csv.writer(employee_file, delimiter=';', quotechar='"', quoting=0)
        header=["cluster","lvl1","lvl2","lvl3"]
        csv_writer.writerow(header)
        for i in range(int(sys.argv[3])):
            values=[]
            values.append(i)
            lvl1=lvl1_entropy[i]
            lvl2=lvl2_entropy[i]
            lvl3=lvl3_entropy[i]
            values.append(len(lvl1))
            values.append(len(lvl2))
            values.append(len(lvl3))
            csv_writer.writerow(values)