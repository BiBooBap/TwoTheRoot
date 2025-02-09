import pandas as pd
import numpy as np

def controllo_none(df):
   # Debug: Print unique values in "Medical_History" before replacement
   print("Unique values in 'Medical_History' before replacement:", df["Medical_History"].unique())
   
   # Convert "None" strings to actual None values
   df["Medical_History"].replace("None", np.nan, inplace=True)
   
   # Debug: Print unique values in "Medical_History" after replacement
   print("Unique values in 'Medical_History' after replacement:", df["Medical_History"].unique())
   
   # Debug: Print rows where "Medical_History" is null
   print("Rows with null 'Medical_History':")
   print(df[df["Medical_History"].isnull()])
   
   # Drop rows where "Medical_History" is null, NaN, or None and "Medication" is "Yes" or True
   df = df.drop(df[(df["Medical_History"]=='nan') & (df["Medication"].isin(["Yes", True]))].index)
   
   return df


