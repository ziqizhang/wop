from feature import dictionary_feature_extractor_auto as dfea
from feature import dictionary_extractor as de

if __name__=="__main__":
    #folder containing the dictionaries
    dictionary_folder="/home/zz/Work/msm4phi/resources/dictionary"
    #original feature csv file containing at least the text fields to be matched, and user id
    csv_input_feature_file= "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/training_data/basic_features_filled_profiles.csv"


    #dict1-dictionary created based on lemmatisation; dict2-based on stemming
    #therefore also change the value'text_normalization_option = 0' in dictionary_extractor to use corresponding normalisation on text
    dict_lemstem_option="dict2"
    # output folder to save dictionary features
    outfolder = "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/features/empty_profile_filled/manual_dict_feature_1"

    # column id of the target text field
    target_text_cols = 22  # 22=profile text; 15=name field
    target_text_name_suffix="_profile"
    col_id=0
    #how many entries from each dictionary should be selected for matching(top n). Changing this param will generate
    #different features, so perhaps influencing classification results
    topN_of_dict=500000

    #load auto extracted dictionaries, match to 'profile'
    postype_dictionaries = \
        de.load_extracted_dictionary(dictionary_folder+"/manually_created/typed",
                                     topN_of_dict, "dict")
    extracted_dictionaries = dfea.flatten_dictionary(postype_dictionaries)
    dfea.match_extracted_dictionary(extracted_dictionaries, csv_input_feature_file,
                               col_id, outfolder+"/feature_manualcreated_dict_match"+target_text_name_suffix+".csv",
                               target_text_cols)

