import csv


def clean_sort_hashtags(infile, outfile):
    with open(infile) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    unique_hashtags=[]
    for tag in content:
        if tag.startswith("#") and tag not in unique_hashtags:
            unique_hashtags.append(tag)
    unique_hashtags=sorted(unique_hashtags)

    with open(outfile, 'w') as file:
        for tag in unique_hashtags:
            file.write(tag+"\n")


# diseases, tags, deleted, non-disease but cb tags
# save as tsv. col1 = key; col2= | separated disease tags;
# col3  = | separated non-disease but community building tags with (cb) suffix
#
# this will output two files in 'outfolder', one tsv file using the above format.


def organise_hashtags(infile, outfolder):
    with open(infile) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    max_track_keywords=400
    discarded_tags={} #key-type of tag; value-set containing those tags
    disease_tags={} #key-a name assigned to that disease; key - set of tags
    disease_cb_tags={} #key-a name assigned to that disease; value: set of cb tags
    unique_hashtags=[] #unique hashtags found, to be used by the Twitter streaming api to track tweets

    with open(outfolder+'tags_organised.tsv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in content:
            if line.startswith("#"):
                if("," in line):
                    print("error")

                tag_and_type=parse_brackets(line)
                disease_name=assign_disease_name(tag_and_type)
                add_unique_tag(unique_hashtags, tag_and_type[0])
                if not tag_and_type[1]:
                    add_values(disease_tags, tag_and_type,disease_name)
                else:
                    add_values(disease_cb_tags,tag_and_type,disease_name)
                if tag_and_type[1]:
                    csvwriter.writerow([disease_name,'',tag_and_type[0]])
                else:
                    csvwriter.writerow([disease_name, tag_and_type[0],''])

            elif line.startswith("[#"):
                e = line.find("]")
                assert e > -1
                line=line[1:e]
                tags=sorted(line.split(","))
                tag_and_type=parse_brackets(tags[0])
                disease_name = assign_disease_name(tag_and_type)
                disease_tag_string=""
                disease_cb_tag_string=""
                for t in tags:
                    tag_and_type = parse_brackets(t)
                    add_unique_tag(unique_hashtags,tag_and_type[0])
                    if not tag_and_type[1]:
                        add_values(disease_tags,tag_and_type, disease_name)
                        disease_tag_string+=tag_and_type[0]+'|'
                    else:
                        add_values(disease_cb_tags,tag_and_type, disease_name)
                        disease_cb_tag_string += tag_and_type[0] + '|'
                csvwriter.writerow([disease_name, disease_tag_string, disease_cb_tag_string])

            elif line.startswith("x"):
                parts=line.split("\t")
                type=parts[2]
                tag=parts[1]
                if type in discarded_tags.keys():
                    discarded_tags[type].append(tag)
                else:
                    values=[]
                    values.append(tag)
                    discarded_tags[type]=values

    #output stats
    output_stats(discarded_tags,disease_tags,disease_cb_tags,unique_hashtags)
    #output unique hashtags
    output_unique_hashtags(unique_hashtags,outfolder,max_track_keywords)


def output_stats(discarded_tags, disease_tags, disease_cb_tags, unique_hashtags):
    print("discarded tags:")
    total=0
    for k, v in discarded_tags.items():
        print("\t{}={}".format(k,len(v)))
        total+=len(v)
    print("total={}".format(total))

    print("\ntotal tags={}".format(len(unique_hashtags)))

    unique_diseases=set()
    unique_diseases.update(disease_tags.keys())
    unique_diseases.update(disease_cb_tags.keys())
    print("\ntotal diseases={}".format(len(unique_diseases)))
    print("\nout of which {} has disease tags, {} has cb tags".format(len(disease_tags), len(disease_cb_tags)))


def output_unique_hashtags(unique_hashtags, outfolder, max_track_keywords):
    unique_hashtags=sorted(unique_hashtags)
    files=int(len(unique_hashtags)/max_track_keywords)+1
    elements = len(unique_hashtags)/files

    filecounter=0
    line=""
    for i in range(len(unique_hashtags)):
        line+=unique_hashtags[i]+","
        if i>elements*(filecounter+1):
            with open(outfolder+"tag_list_"+str(filecounter), 'w') as file:
                file.write(line)
            line=""
            filecounter+=1
    if line!="":
        with open(outfolder + "tag_list_" + str(filecounter), 'w') as file:
            file.write(line)


def assign_disease_name(tag_and_type:list):
    if tag_and_type[1]:
        dt="unknown-"+tag_and_type[0]
    else:
        dt=tag_and_type[0]
    return dt

def add_values(dictionary:dict, tag_and_type:list, tag_name):
    if tag_name in dictionary.keys():
        values=dictionary[tag_name]
    else:
        values=[]

    if tag_and_type[0] not in values:
        values.append(tag_and_type[0])
        values=sorted(values)
        dictionary[tag_name]=values


def add_unique_tag(tags:list, to_add):
    if to_add not in tags:
        tags.append(to_add)

#first element, the parsed hashtag; second element, whether this is a CB tag
def parse_brackets(hashtag):
    if not hashtag.startswith("#"):
        print(hashtag)

    s = hashtag.find("(")
    if s==-1:
        return [hashtag,False]
    else:
        e = hashtag.find(")")
        assert e>-1
        tag = hashtag[0:s].strip()
        return [tag, True]


if __name__ == '__main__':
    # clean_sort_hashtags("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/symplur_hashtags_disease.txt",
    #                     "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/unique_symplur_hashtags_disease.txt")

    organise_hashtags("/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/unique_symplur_hashtags_disease.txt",
                      "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/data/")