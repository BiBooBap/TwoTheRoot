import pandas as pd 


def controllo_none(df):
   
   df = df.drop(df[((df["Medical_History"] == "nan") & (df['Medication']==True))].index)

   return df


