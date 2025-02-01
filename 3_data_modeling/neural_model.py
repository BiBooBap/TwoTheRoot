import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Carichiamo il dataset
    df = pd.read_csv("../0_data/dataset_cleaned.csv")
    
    # Separiamo le feature (X) dal target (y)
    X = df.drop(columns=['Trigger'])
    y = df['Trigger']
    
    # Pre-elaborazione delle feature:
    # - One-hot encoding per le feature categoriche
    X = pd.get_dummies(X, drop_first=True)
    
    # - Standardizzazione delle feature numeriche
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Pre-elaborazione del target: Label encoding e one-hot encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_ohe = tf.keras.utils.to_categorical(y_enc)
    num_classes = y_ohe.shape[1]
    input_dim = X_scaled.shape[1]
    
    # Validazione incrociata K-Fold (n=10)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = []
    best_val_acc = 0
    best_model = None

    for train_index, val_index in kfold.split(X_scaled, y_ohe):
        model = create_model(input_dim, num_classes)
        history = model.fit(X_scaled[train_index], y_ohe[train_index],
                            validation_data=(X_scaled[val_index], y_ohe[val_index]),
                            epochs=100,
                            batch_size=16,
                            callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)],
                            verbose=0)
        val_loss, val_acc = model.evaluate(X_scaled[val_index], y_ohe[val_index], verbose=0)
        cv_scores.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    print("CV Accuracy: {:.2f}% (+/- {:.2f}%)".format(np.mean(cv_scores)*100, np.std(cv_scores)*100))
    
    # Salviamo il modello migliore per poterlo utilizzare da allenato
    best_model.save("../3_data_modeling/best_neural_network_model.keras")

if __name__ == '__main__':
    main()