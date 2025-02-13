from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# Percorsi dei file
data_path = r"../0_data/dataset_pre.csv"
model_path = r"../3_data_modeling/random_forest_model.pkl"
with open(model_path, "rb") as f:
    rf_model = pickle.load(f)
print("Tipo di modello caricato:", type(rf_model))

# Carica il dataset originale
df_dataset = pd.read_csv(data_path)

# Definisci le colonne usate in input (escludo ID e Trigger)
categorical_columns = [col for col in df_dataset.select_dtypes(include=['object']).columns if col != 'Trigger']
numerical_columns = [col for col in df_dataset.select_dtypes(include=['number']).columns if col != 'ID']

# Prepara gli encoder: ad ogni feature categorica viene associato un LabelEncoder addestrato sui valori originali
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_dataset[col] = le.fit_transform(df_dataset[col])
    encoders[col] = le

# Prepara lo scaler per le feature numeriche, escludendo ID
scaler = MinMaxScaler(feature_range=(0, 1))
df_dataset[numerical_columns] = scaler.fit_transform(df_dataset[numerical_columns].fillna(0))

# Carica il modello random forest
with open(model_path, "rb") as f:
    rf_model = pickle.load(f)

# Prepara i valori per i dropdown: leggi i valori originali (non trasformati) dal CSV
df_original = pd.read_csv(data_path)
dropdown_options = {}
for col in categorical_columns:
    dropdown_options[col] = sorted(df_original[col].dropna().astype(str).unique())

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_data = {}
        # Processa gli input numerici
        for col in numerical_columns:
            value = request.form.get(col)
            if not value:
                return f"Il campo '{col}' non può essere vuoto."
            try:
                input_data[col] = float(value)
            except ValueError:
                return f"Valore non valido per il campo '{col}'."
        # Processa gli input categorici
        for col in categorical_columns:
            value = request.form.get(col)
            if not value:
                return f"Il campo '{col}' non può essere vuoto."
            input_data[col] = value

        # Crea un DataFrame con i dati di input
        input_df = pd.DataFrame([input_data])
        
        # Applica la codifica con gli encoder pre-addestrati
        for col in categorical_columns:
            input_df[col] = encoders[col].transform(input_df[col])
            
        # Applica la normalizzazione alle colonne numeriche
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns].fillna(0))
        
        # Riordina le colonne per farle corrispondere all'ordine usato in training
        # L'ordine delle feature è quello presente nel dataset originale esclusa la colonna 'Trigger'
        model_features = list(df_dataset.drop(columns=["Trigger"]).columns)
        input_df = input_df[model_features]
        
        # Predici il Trigger con il modello Random Forest
        prediction = rf_model.predict(input_df)
        pred_probs = rf_model.predict_proba(input_df)
        max_prob = max(pred_probs[0])
        if max_prob < 0.3:
            predicted_trigger = prediction[0] + " (low probability, uncertain prediction)"
        else:
            predicted_trigger = prediction[0]
        
        return render_template("index.html", dropdown_options=dropdown_options,  prediction=predicted_trigger, form_data=input_data)
    return render_template("index.html", dropdown_options=dropdown_options)

if __name__ == "__main__":
    app.run(debug=True)