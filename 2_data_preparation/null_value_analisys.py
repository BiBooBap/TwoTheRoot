import pandas as pd
import matplotlib.pyplot as plt

def main(df):
    analysis_data = []

    for column in df.columns:
        if column == 'Trigger':
            null_count = df[column][df[column] == 'Unknown'].count()
        else:
            null_count = df[column].isnull().sum()

        not_null_count = len(df) - null_count

        if df[column].dtype == 'object':
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