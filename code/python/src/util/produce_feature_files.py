# Used to produce a new file with selected features
# feature_columns needs to be changed accordingly 
# first argument is the name of the file with features
import sys
import pandas as pd 

feature_file = sys.argv[1]

# read annotated data file 

# Read the file with annotated data 
annotated_df = pd.read_csv("merged_output.csv")

# feature columns 
feature_columns = ['user_statuses_count', 'user_friends_count',
    'user_favorites_count','user_retweeted_count',
    'user_retweet_count', 'user_followers_count',
    'user_listed_count', 'user_newtweet_count', 'user_favorited_count',
    'user_entities_hashtag', 'user_entities_url',
    'user_entities_user_mention', 'user_entities_media_url']

col_names = ['twitter_id'] + feature_columns

# read features file 

features_df = pd.read_csv(feature_file)
# select required features 
features_df = features_df[col_names]

col_names = col_names + ['label']
output_df = pd.DataFrame(columns = col_names)


# iterate through the annotate users

for row in range(len(annotated_df)):
    label_value  = annotated_df.label[row]
    
    if label_value == "" :
        label_value = "Other"
        print("empty")
    current_id = annotated_df.twitter_id[row]
    
    features = pd.DataFrame(columns = col_names)
    #print("columns are: " + str(list(features.columns.values)))
    #features['twitter_id'] = current_id
    #features['label'] = label_value
    # get row from features with this twitter id
    feature_row = features_df.loc[features_df['twitter_id'] == current_id]
    
    features[feature_columns] = feature_row[feature_columns]
    
    # add the feature vector into the new file
    output_df.loc[row] = features.iloc[0]
    output_df.twitter_id[row] = current_id
    output_df.label[row] = label_value
    # add the label 
    #output_df.label[row] = label
#output_df = pd.to_numeric(output_df[feature_columns], errors='coerce')
output_df[feature_columns] = output_df[feature_columns].apply(pd.to_numeric)
output_df = output_df.fillna("Other")
output_df.to_csv("output_features.csv", index = False)