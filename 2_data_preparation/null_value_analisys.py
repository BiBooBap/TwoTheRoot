import pandas as pd
import matplotlib.pyplot as plt

def main(df):
    analysis_data = []

    for column in df.columns:
        if column == 'Trigger':
            null_count = df[column][df[column] == 'Unknown'].count()
        elif column == 'Medical_History':
            null_count = df[column].isnull().sum() + df[column][df[column] == 'None'].count()
        else:
            null_count = df[column].isnull().sum()

        not_null_count = len(df) - null_count
        data_type = df[column].dtype

        analysis_data.append([null_count, not_null_count, data_type])

    analysis_df = pd.DataFrame(
        analysis_data,
        index=df.columns,
        columns=["Null Values", "Non-Null Values", "Data Type"]
    )

    fig, ax = plt.subplots()
    ax.table(
        cellText=analysis_df.values,
        colLabels=analysis_df.columns,
        rowLabels=analysis_df.index,
        loc='center'
    )
    ax.axis('off')
    plt.show()