

# GLOBAL VARIABLES
#DATA_ORG = "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_o.csv"
#TASK_NAME = "scholarlydata_org"
#DATA_COLS_START = 3  # inclusive
#DATA_COLS_END = 20  # exclusive 16
#DATA_COLS_FT_END = 16  # exclusive 12
#DATA_COLS_TRUTH = 16  # inclusive 12


def create_dataset_props(task, identifier, feature_file, feature_col_start, feature_col_end, feature_size, truth_col, appended_list):
    props=(task, identifier, feature_file, feature_col_start, feature_col_end, feature_size, truth_col)
    appended_list.append(props)

def load_exp_datasets():
    l=list()
    load(l)
    #load_per_datasets(l)
    return l

#methods to create different experiment runs where each run uses a different range of features as
#identified by the start and end feature columns
def load(list):
    # param 0 and 1, are identifiers to be used to name the output
    # param 2: file pointing to the feature csv
    # param 3: column (0 indexed) index of feature start
    # param 4: column index of feature end
    # param 5: # of features
    # param 6: column index minus index of feature start, for the annotated labels. so if int he original csv has 20 columns
    # and the last column (index 19) is the label column, and first feature starts from column 3, this value is 19-3 = 16

    # create_dataset_props("scholarlydata_org", "original ",
    #                      "data/train/training_org(expanded)_features_o.csv",
    #                      3,20,16,16, list) #3,20,16,16
    create_dataset_props("georgica", "original ",
                         "data/train/output_features.csv",
                         1,15,13,13, list,
                         "/home/zz/Work/msm4phi/output")
