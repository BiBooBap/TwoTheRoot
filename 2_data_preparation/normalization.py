import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main(df):
    # Selezioniamo le colonne numeriche, ignorando l'ID
    numeric_cols = df.select_dtypes(include=['float64', 'int32', 'int64']).drop(columns=["ID"], errors='ignore').columns

    print("Numeric columns to be normalized:")
    print(numeric_cols)

    # Trasformazione feature che hanno "Yes"/"No" come valori in booleane
    colonne_da_convertire = ['Sweating', 'Shortness_of_Breath', 'Chest_Pain', 'Dizziness', 'Trembling', 'Medication', 'Smoking', 'Therapy']
    for col in colonne_da_convertire:
        df[col] = df[col].replace({'Yes': True, 'No': False})
    
    # Normalizzazione min-max con min=0 e max=1, escludendo la colonna ID
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
    df[numeric_cols] = df[numeric_cols].round(3)

    print("DataFrame dopo normalizazione:")
    print(df.head())

    return df