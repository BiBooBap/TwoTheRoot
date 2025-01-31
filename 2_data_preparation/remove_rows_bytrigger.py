import pandas as pd

def main(df):
    df = df[df["Trigger"] != "Unknown"].copy()
    df = df.reset_index(drop=True)
    df["ID"] = df.index + 1

    return df