import pandas as pd
from sklearn.preprocessing import LabelEncoder
import knnimputation_medicalnone as knn_med

def controllo_none_migliorato(df):
    # Seleziona solo le righe che soddisfano la condizione (Medical_History NaN e Medication True)
    rows_for_imputation = df[(pd.isna(df["Medical_History"])) & (df['Medication'] == True)]
    
    # Verifica se ci sono righe da imputare
    if not rows_for_imputation.empty:
        # Codifica la colonna 'Medical_History' solo sui dati da imputare
        encoder_rows = LabelEncoder()
        
        # Codifica solo le righe selezionate per imputazione
        rows_for_imputation['Medical_History'] = encoder_rows.fit_transform(rows_for_imputation['Medical_History'].fillna('Unknown'))
        
        # Esegui l'imputazione tramite KNN (assicurati che la funzione knn_med.main() lavori con le righe giuste)
        df_imputed = knn_med.main(rows_for_imputation)
        
        # Sostituisci i valori imputati nelle righe originali del DataFrame
        df.loc[rows_for_imputation.index, 'Medical_History'] = df_imputed['Medical_History']

        # Decodifica i valori imputati
        df['Medical_History'] = encoder_rows.inverse_transform(df['Medical_History'])
    
    return df
