import pandas as pd

def main(df):
    df = df[df["Trigger"] != "Unknown"].copy()
    df = df.reset_index(drop=True)
    df["ID"] = df.index + 1
    df.to_csv("../0_data/dataset_no_unknown.csv", index=False)