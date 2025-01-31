import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Imputazione dei valori mancanti della colonna "Medical_History" con KNN Imputation
def main(df):
    # Conserviamo i tipi originali delle colonne numeriche per ripristinarli dopo l'imputazione, che li trasforma in float
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    original_dtypes = df[numeric_cols].dtypes

    # Eseguiamo l'encoding della feature "Medical_History" per poterla usare con l'imputazione KNN
    label_enc = LabelEncoder()
    df["Medical_History"] = label_enc.fit_transform(df["Medical_History"].astype(str))

    # Applichiamo l'imputazione KNN
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = df_imputed

    # Ripristiniamo i tipi originali (per gli interi arrotondiamo)
    for col in numeric_cols:
        if original_dtypes[col] == 'int64' or original_dtypes[col] == 'int32':
            df[col] = df[col].round().astype(int)

    # Decode dei valori di "Medical_History" dopo l'imputazione
    df["Medical_History"] = label_enc.inverse_transform(df["Medical_History"].round().astype(int))

    return df