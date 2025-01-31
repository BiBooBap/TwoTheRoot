import pandas as pd
import matplotlib.pyplot as plt

def main(df):
    # Trasforma le istanze object in boolean/string prima di effettuare l'analisi
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if all(val in ["Yes", "No"] for val in unique_vals):
                df[col] = df[col].map({"Yes": True, "No": False})

    analysis_data = []

    for column in df.columns:
        if column == 'Trigger':
            null_count = df[column][df[column] == 'Unknown'].count()
        else:
            null_count = df[column].isnull().sum()

        not_null_count = len(df) - null_count

        if df[column].dtype == 'object':
            non_nulls = df[column].dropna()
            if len(non_nulls) > 0:
                python_type = type(non_nulls.iloc[0]).__name__
            else:
                python_type = 'object'
        else:
            python_type = df[column].dtype.name

        analysis_data.append([null_count, not_null_count, python_type])

    analysis_df = pd.DataFrame(
        analysis_data,
        index=df.columns,
        columns=["Null/Empty Values", "Non-Null Values", "Data Type"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    table = ax.table(
        cellText=analysis_df.values,
        colLabels=analysis_df.columns,
        rowLabels=analysis_df.index,
        loc='center'
    )
    table.scale(0.5, 1.3)
    ax.axis('off')
    plt.show()