from sklearn.preprocessing import MinMaxScaler

def main(df):
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Normalizzazione min-max con min=0 e max=1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(0))

    print("DataFrame dopo normalizzazione:")
    print(df.head())

    return df