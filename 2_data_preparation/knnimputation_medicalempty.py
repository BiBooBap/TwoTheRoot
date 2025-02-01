import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Imputazione dei valori mancanti della colonna "Medical_History" con KNN Imputation
def main(df):
    df_temp = df.copy()

    # Conserviamo i tipi originali delle colonne numeriche per ripristinarli dopo l'imputazione, che li trasforma in float
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    original_dtypes = df[numeric_cols].dtypes

    # Eseguiamo l'encoding della feature "Medical_History" per poterla usare con l'imputazione KNN
    label_enc = LabelEncoder()
    df_temp["Medical_History"] = label_enc.fit_transform(df_temp["Medical_History"].astype(str))

    # Individuiamo le colonne categoriche (escluse quelle già codificate) e applichiamo un encoding one-hot
    cat_cols = df_temp.select_dtypes(include=['object']).columns.tolist()
    if "Medical_History" in cat_cols:
        cat_cols.remove("Medical_History")

    df_temp = pd.get_dummies(df_temp, columns=cat_cols, dummy_na=True)

    # Applichiamo l'imputazione KNN
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df_temp)

    # Ricreiamo un DataFrame con le stesse colonne di df_temp
    imputed_df = pd.DataFrame(df_imputed, columns=df_temp.columns)

    # Ripristiniamo i tipi originali (per gli interi arrotondiamo)
    for col in numeric_cols:
        df[col] = imputed_df[col]
        if original_dtypes[col] in ['int64', 'int32']:
            df[col] = df[col].round().astype(int)

    # Decode dei valori di "Medical_History" dopo l'imputazione
    med_col_idx = df_temp.columns.get_loc("Medical_History")
    df["Medical_History"] = label_enc.inverse_transform(
        imputed_df.iloc[:, med_col_idx].round().astype(int)
    )

    return df