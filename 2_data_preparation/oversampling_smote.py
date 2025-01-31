
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def oversampling(df):
    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    if('Trigger' in object_cols):
        object_cols.remove('Trigger')
        trigger_is_object = True
    
    encoder_trigger = LabelEncoder()
    encoders = {}
    for col in object_cols:
        encoders[col] = LabelEncoder()
        df[col]= encoders[col].fit_transform(df[col])

    #prendo prima tutte le colonne tranne la colonna target
    #poi prendo solo la colonna target
    values_drop = ['Trigger', 'ID']
    X = df.drop(values_drop, axis=1)
    Y= df['Trigger']

    if trigger_is_object:
        Y = encoder_trigger.fit_transform(Y)
        #Trasformo solo qua Trigger
        #il valore randomico permette la riproducibilità, assicurandosi che si otterranno gli stessi risultati ogni volta che si esegue il codice
    
    sm = SMOTE(random_state=42)
    X_res, Y_res = sm.fit_resample(X, Y)

    df = pd.DataFrame(X_res, columns=X.columns)
    df['Trigger'] = Y_res
    for col in object_cols:
        df[col] = encoders[col].inverse_transform(df[col])
    df['Trigger'] = encoder_trigger.inverse_transform(df['Trigger'])
    

    df.insert(0, 'ID', range(1, 1 + len(df)))

    #stampa un grafico a torta per mostrare la nuova distribuzione dei valori di trigger
    print("Nuova distribuzione dei valori di Trigger:")
    print(df['Trigger'].value_counts())
    print("\n")
    # ... (Previous code for oversampling - same as before) ...

    # Conta il numero di sample per ogni classe dopo l'oversampling
    class_counts = df['Trigger'].value_counts()

    # Crea il grafico a torta
    plt.figure(figsize=(8, 8))  # Imposta la dimensione della figura
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribuzione dei valori Trigger dopo l\'oversampling') 
    plt.show()

    return df

