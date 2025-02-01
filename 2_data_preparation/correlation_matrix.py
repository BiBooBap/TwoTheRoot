# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main(df):
    # Memorizziamo le colonne numeriche e categoriche
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Le colonne categoriche necessitano di encoding prima di calcolare la correlazione
    print("Categorical columns that need encoding:", list(categorical_cols))

    # Copiamo il DataFrame per non modificare l'originale
    df_numeric = df.copy()

    # Numero di occorrenze di ciascun sintomo
    symptom_cols = ['Sweating', 'Shortness_of_Breath', 'Dizziness', 'Chest_Pain', 'Trembling']
    symptom_counts = df[symptom_cols].apply(lambda x: (x == "Yes").sum())

    # Yes e No vengono convertiti in 1 e 0
    yes_no_cols = symptom_cols + ['Smoking', 'Therapy', 'Medication']
    df_numeric[yes_no_cols] = df_numeric[yes_no_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

    # Gestiamo le colonne categoriche con più di due categorie, tipo "Gender"
    for col in categorical_cols:
        if col not in yes_no_cols:  # Escludiamo le colonne categoriche già convertite in numeriche, in questo caso erano solo quelle con Yes/No
            # Usiamo get_dummies per convertire le colonne categoriche in numeriche, e separiamo in più colonne, per ogni categoria dei valori che erano nella colonna originale (es. Maschio/Femmina/Non-binario diventano tre colonne separate)
            dummies = pd.get_dummies(df_numeric[col], prefix=col)
            # Uniamo le nuove colonne al DataFrame principale
            df_numeric = pd.concat([df_numeric, dummies], axis=1)
            # Rimuoviamo la colonna originale
            df_numeric.drop(col, axis=1, inplace=True)

    # Ora possiamo calcolare la correlazione tra le colonne numeriche
    correlation_matrix = df_numeric.corr()

    # Qui viene visualizzata la matrice di correlazione con un grafico a matrice di calore Heatmap
    plt.figure(figsize=(18,15))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5,
                square=True,
                annot_kws={"fontsize": 9})
    plt.title("Correlation Matrix of Panic Attack Dataset")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()