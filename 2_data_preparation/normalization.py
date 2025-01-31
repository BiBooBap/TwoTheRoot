import matplotlib.pyplot as plt

def main(df):
    # Selezioniamo le colonne numeriche, ignorando l'ID
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).drop(columns=["ID"], errors='ignore').columns

    # Normalizzazione min-max
    df[numeric_cols] = ((df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())).round(3)

    # Trasformazione feature che hanno "Yes"/"No" come valori in booleane
    for col in df.columns:
        if df[col].dropna().isin(["Yes", "No"]).all():
            df[col] = df[col].map({"Yes": True, "No": False})

    return df