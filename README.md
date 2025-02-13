<p align="center">
  <img src="./logo.png" width=200>
  <h1 align="center">TwoTheRoot</h1>
</p>

Questo repository contiene un modello di Machine Learning progettato per supportare i medici nell'analisi delle cause degli attacchi di panico. Il sistema è stato sviluppato per fornire uno strumento di supporto decisionale che, attraverso l'analisi e la classificazione dei dati, aiuta a identificare pattern e fattori critici associati a queste condizioni.

Il progetto include:
- **Pre-elaborazione dei dati:** Pulizia e preparazione dei dataset per l'addestramento del modello.
- **Costruzione e addestramento del modello:** Utilizzo di algoritmi di classificazione per sviluppare un modello predittivo.
- **Validazione del modello:** Valutazione tramite metriche di performance e analisi delle curve ROC per ogni classe.
- **Visualizzazione dei risultati:** Generazione di grafici e tabelle che aiutano nell'interpretazione dei risultati.
- **Deploy per l'utilizzo:** Applicazione web per l'utilizzo del modello pre-trained usando Flask.

L'obiettivo è fornire uno strumento affidabile e utile nel contesto medico, integrando tecnologie avanzate per migliorare la comprensione e la gestione degli attacchi di panico.

Dataset usato: [Panic Attack Dataset](https://www.kaggle.com/datasets/ashaychoudhary/panic-attack-dataset/data)
<br><br>

# Panoramica delle Fasi del Progetto

Questo repository è strutturato in diverse fasi, ciascuna rappresentata da un file principale, per riprodurre l'intero flusso che va dalla preparazione dei dati fino al deployment del modello.

<br>

## 1. DataPreparation

Il file `DataPreparation.py` si occupa del pre-processing dei dati. In questa fase vengono eseguiti diversi passaggi fondamentali:

- **Importazione dei dati:** Caricamento del dataset originale.
- **Pulizia e rimozione delle righe duplicate:** Utilizzo di funzioni dedicate per eliminare dati ridondanti.
- **Analisi dei valori nulli:** Verifica e gestione dei valori mancanti.
- **Rimozione di righe non rilevanti:** Filtraggio dei dati in base a determinati trigger.
- **Imputazione dei valori mancanti:** Applicazione di un'algoritmo KNN specifico per il contesto medico.
- **Oversampling:** Bilanciamento del dataset tramite tecniche come SMOTE per affrontare le classi sbilanciate.
- **Controllo dei valori None:** Assicurarsi che non vi siano valori nulli prima della codifica.
- **Codifica e normalizzazione:** Applicazione dell'encoding per le feature categoriali e scaling per quelle numeriche.

Per riprodurre questa fase, eseguire il codice tramite terminale, dalla cartella di progetto:
```bash
python 2_data_preparation/DataPreparation.py
```

Il risultato di questa fase sono due file CSV:

`dataset_pre.csv`: Il dataset dopo le prime operazioni di pulizia e oversampling (utilizzato nella fase di Deploy).
`dataset_cleaned.csv`: Il dataset completamento preprocessato (con encoding e normalizzazione) per l'addestramento.

<br>

## 2. Random Forest

Il file `random_forest.py` si occupa della costruzione, validazione e analisi del modello. I passaggi principali sono:

- **Lettura del dataset pulito:** Caricamento del file dataset_cleaned.csv.
- **Addestramento del modello:** Utilizzo di una RandomForestClassifier con specifici parametri, addestrato in una procedura di cross-validation (con ripetizioni e fold multipli).
- **Valutazione delle performance:** Calcolo delle metriche come Accuracy, Precision, Recall, F1-score e la creazione di matrici di confusione.
- **Calcolo delle curve ROC:** Binarizzazione delle classi e valutazione delle curve ROC per ogni classe, formando una media delle performance attraverso i fold.
- **Visualizzazione:** Creazione di grafici e tabelle che permettono di interpretare i risultati aggregati del modello.

Per riprodurre questa fase, eseguire il codice tramite terminale, dalla cartella di progetto:
```bash
python 3_data_modeling/random_forest.py
```
Al termine, il modello addestrato può essere serializzato per l’uso nel deployment.

<br>

## 3. Deployment

Il file `deployment.py` gestisce la fase di deploy del modello tramite un'applicazione web basata su Flask. In particolare, questa fase comprende:

- **Caricamento del modello e dei dati:** Il modello serializzato (Random Forest) e il dataset `dataset_pre.csv` vengono caricati in memoria.
- **Preparazione dell'interfaccia utente:** Creazione di dropdown e campi di input per raccogliere i dati inseriti dall'utente.
- **Pre-processamento in fase di inferenza:** Applicazione degli stessi encoder e scaler usati durante l’addestramento per trasformare i nuovi dati.
- **Predizione:** Utilizzo del modello per effettuare predizioni sul trigger, includendo una gestione della soglia per bassa probabilità.
- **Visualizzazione del risultato:** Rendering del risultato tramite un template HTML.

Per eseguire l'applicazione web, eseguire nella cartella del progetto:
```bash
python 5_deploy/deployment.py
```
Questa operazione lancerà il server Flask in modalità debug, rendendo disponibile l'interfaccia per inserire nuovi dati e ottenere predizioni in tempo reale.

<br>

---

<br>
Con queste tre fasi è possibile riprodurre l'intero workflow: dalla preparazione e pulizia dei dati, alla formazione e valutazione del modello, fino all'implementazione di un'interfaccia web per l'uso in ambiente clinico.