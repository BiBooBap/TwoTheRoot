import pandas as pd

def main(df):
    df = df[df["Trigger"] != "Unknown"].copy()

    return df