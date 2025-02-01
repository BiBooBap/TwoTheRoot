import pandas as pd
import remove_duplicated_rows as rem_dup
import null_value_analisys as null_val
import remove_rows_bytrigger as rem_bytr
import knnimputation_medicalempty as knn_med
import normalization as norm
import oversampling_smote as over
import controllo_none as cont
try:
    df = pd.read_csv("./0_data/dataset.csv")
    df = rem_dup.main(df)
    null_val.main(df)
    df = rem_bytr.main(df)
    null_val.main(df)
    df = knn_med.main(df)
    null_val.main(df)
    df = norm.main(df)
    null_val.main(df)
    df = over.oversampling(df)
    df= cont.controllo_none(df)
    df.to_csv("./0_data/dataset_cleaned.csv", index=False)
except FileNotFoundError:
    print(f"Error: File './0_data/dataset.csv' not found.")