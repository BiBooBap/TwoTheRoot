import pandas as pd
import correlation_matrix as cor_mat
import remove_duplicated_rows as rem_dup
import null_value_analisys as null_val
import remove_rows_bytrigger as rem_bytr
import knnimputation_medicalnone as knn_med

try:
    df = pd.read_csv("../0_data/dataset.csv")
    cor_mat.main(df)
    rem_dup.main(df)
    null_val.main(df)
    rem_bytr.main(df)
    try:
        df = pd.read_csv("../0_data/dataset_no_unknown.csv")
        null_val.main(df)
        knn_med.main(df)
        null_val.main(df)
    except FileNotFoundError:
        print(f"Error: File '../0_data/dataset_no_unknown.csv' not found.")
except FileNotFoundError:
    print(f"Error: File '../0_data/dataset.csv' not found.")