from sklearn.preprocessing import LabelEncoder

def main(df):
    # Encoding delle features categoriche
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Trigger':
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
    
    print("DataFrame dopo encoding:")
    print(df.head())

    return df