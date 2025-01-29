import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

def main(df):
    # Codifichiamo "Medical_History" per poter usare l'imputazione KNN
    label_enc = LabelEncoder()
    df["Medical_History"] = label_enc.fit_transform(df["Medical_History"].astype(str))

    # Selezioniamo le colonne numeriche da utilizzare per l'imputazione
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Applichiamo l'imputazione KNN
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = df_imputed

    # Decodifichiamo i valori di "Medical_History" dopo l'imputazione
    df["Medical_History"] = label_enc.inverse_transform(df["Medical_History"].round().astype(int))

    # Salviamo un nuovo dataset con i valori imputati
    df.to_csv("../0_data/dataset_imputed.csv", index=False)