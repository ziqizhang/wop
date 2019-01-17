import pandas as pd
from feature import dictionary_feature_extractor_auto as dfe

def match_extracted_healthconditions(dictionary: dict, csv_input_tweets_file, col_id, outfile,
                                     *col_target_texts):
    df = pd.read_csv(csv_input_tweets_file, header=0, delimiter=",", quoting=0).as_matrix()

    output_matrix = []

    output_header = ["user_id"]
    output_header.append("relevant_tweets")
    output_header.append("count_hc")
    output_matrix.append(output_header)

    for row in df:
        row_data = [row[col_id]]
        target_text = ""
        for tt_col in col_target_texts:
            text = row[tt_col]
            if type(text) is float:
                row_data.append("0")
                row_data.append("0")
                output_matrix.append(row_data)
                continue
            target_text += row[tt_col] + " "
        target_text = target_text.strip().lower()

        if len(target_text) < 2:
            output_matrix.append(row_data)
            continue

        count_hc = dfe.find_hc_matches(dictionary, target_text)

        has_hc = 0
        if count_hc > 0:
            has_hc = 1
        row_data.append(has_hc)
        row_data.append(count_hc)
        output_matrix.append(row_data)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)
