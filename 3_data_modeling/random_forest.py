import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)  # Probabilità per tutte le classi

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, cm, y_prob

# Lettura del dataset e preparazione dei dati
df = pd.read_csv("../0_data/dataset_cleaned.csv")
X = df.drop(columns=["Trigger"])
y = df["Trigger"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestramento e valutazione del modello Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
accuracy, precision, recall, f1, cm, y_prob = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

if cm.shape[0] == cm.shape[1]:
    num_classes = cm.shape[0]
    class_names = sorted(list(df["Trigger"].unique()))  # Get class names from the original dataframe
    for i in range(num_classes):
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        TN = sum(sum(cm)) - TP - FP - FN
        print(f"Class {class_names[i]}:")  # Print class name instead of index
        print("  True Positives (TP):", TP)
        print("  False Positives (FP):", FP)
        print("  False Negatives (FN):", FN)
        print("  True Negatives (TN):", TN)
else:
    print("Confusion matrix is not square, cannot calculate TP, FP, FN, TN per class")
    print("Shape of confusion matrix:", cm.shape)

# Binarizza l'output
y_bin = label_binarize(y_test, classes=sorted(list(df["Trigger"].unique())))
n_classes = y_bin.shape[1]

# Calcolo della curva ROC e dell'AUC per ogni classe
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot delle curve ROC per ogni classe
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange'])
class_names = sorted(list(df["Trigger"].unique()))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Salvataggio del modello per utilizzi futuri
joblib.dump(rf_model, "random_forest_model.pkl")