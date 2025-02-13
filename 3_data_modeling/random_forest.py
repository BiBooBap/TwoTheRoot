import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np

csfont = {'fontname':'Times New Roman'}

# Lettura del dataset e preparazione dei dati
df = pd.read_csv("../0_data/dataset_cleaned.csv")
X = df.drop(columns=["Trigger"])
y = df["Trigger"]

# Parametri per la cross-validation
n_splits = 6  # Numero di folds
n_repeats = 3  # Numero di ripetizioni della cross-validation

# Inizializza RepeatedKFold
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=37)

# Inizializza liste per salvare le metriche
all_acc = []
all_precision = []
all_recall = []
all_f1 = []
all_cm = []
all_fpr = []
all_tpr = []
all_roc_auc = []

fold_data = []  # Lista per salvare i dati di ogni fold
repetition_tables = []  # Lista per salvare le tabelle di ogni fold nella ripetizione corrente

# Loop per le ripetizioni della cross-validation
for repeat, (train_index, test_index) in enumerate(rkf.split(X, y)):
    # All'inizio di ogni ripetizione, resettare repetition_tables
    if repeat % n_splits == 0:
        repetition_tables = []

    fold_number = (repeat % n_splits) + 1  # Resetta il numero del fold ad ogni ripetizione
    print(f"Ripetizione {repeat // n_splits + 1}/{n_repeats}, Fold: {fold_number}/{n_splits}")

    # Divide i dati in training e test set
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Addestramento del modello Random Forest
    rf_model = RandomForestClassifier(n_estimators=500, random_state=37, max_depth=8, min_samples_split=6, min_samples_leaf=3)
    rf_model.fit(X_train, y_train)

    # Predizioni
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)

    # Calcolo delle metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Calcolo di TP, FP, FN, TN per ogni classe
    if cm.shape[0] == cm.shape[1]:
        num_classes = cm.shape[0]
        class_names = sorted(list(df["Trigger"].unique()))
        fold_metrics = {}
        for i in range(num_classes):
            TP = cm[i, i]
            FP = sum(cm[:, i]) - TP
            FN = sum(cm[i, :]) - TP
            TN = sum(sum(cm)) - TP - FP - FN

            fold_metrics[class_names[i]] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
    else:
        print("Confusion matrix is not square, cannot calculate TP, FP, FN, TN per class")
        print("Shape of confusion matrix:", cm.shape)
        fold_metrics = None

    # Salva le metriche per questo fold
    fold_data.append(fold_metrics)

    # Creazione e salvataggio della tabella per questo fold
    if fold_metrics:
        cell_text = []
        row_labels = list(fold_metrics.keys())
        for class_name in row_labels:
            metrics = fold_metrics[class_name]
            cell_text.append([metrics['TP'], metrics['FP'], metrics['FN'], metrics['TN']])
        repetition_tables.append({
            'fold': fold_number,
            'row_labels': row_labels,
            'cell_text': cell_text
        })

    # Binarizza l'output per ROC
    y_bin = label_binarize(y_test, classes=sorted(list(df["Trigger"].unique())))
    n_classes = y_bin.shape[1]

    # Calcola le curve ROC e l'area sotto la curva (AUC) per ogni classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Salva le metriche per questo fold
    all_acc.append(accuracy)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_cm.append(cm)
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

    # Alla fine di una ripetizione (dopo l'ultimo fold) mostra le tabelle aggregate
    if fold_number == n_splits and repetition_tables:
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        axs = axs.flatten()
        for idx, table_data in enumerate(repetition_tables):
            ax = axs[idx]
            ax.axis('off')
            table = ax.table(
                cellText=table_data['cell_text'],
                rowLabels=table_data['row_labels'],
                colLabels=['TP', 'FP', 'FN', 'TN'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)

            ax.set_title(f'Fold {table_data["fold"]}', style='italic', **csfont, fontsize=15)
        plt.tight_layout()
        plt.show()

# Calcola le metriche medie
mean_acc = np.mean(all_acc)
mean_precision = np.mean(all_precision)
mean_recall = np.mean(all_recall)
mean_f1 = np.mean(all_f1)

print("\nRisultati medi:")
print("Accuracy:", mean_acc)
print("Precision:", mean_precision)
print("Recall:", mean_recall)
print("F1-Score:", mean_f1)

# Creazione del plot delle metriche medie
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_values = [mean_acc, mean_precision, mean_recall, mean_f1]

# Ordina le metriche in ordine decrescente
metric_data = sorted(zip(metric_names, metric_values), key=lambda x: x[1], reverse=True)
metric_names = [x[0] for x in metric_data]
metric_values = [x[1] for x in metric_data]

plt.figure(figsize=(12, 6))
plt.ylim(0, 1)  # Imposta i limiti dell'asse y tra 0 e 1

# Plot delle metriche
bar_width = 0.4
x = np.arange(len(metric_names))
colors = ['lightgreen', 'lightsalmon', 'lightseagreen', 'plum']  # Definisci una lista di colori
for i in range(len(metric_names)):
    plt.bar(x[i], metric_values[i], bar_width, color=colors[i], zorder=3)
    plt.text(x[i], metric_values[i] + 0.01, f"{metric_values[i]:.5f}", ha='center', va='bottom', zorder=4)

plt.xticks(x, metric_names)
plt.xlabel('Metrica', style='italic', **csfont, fontsize=20, labelpad=20)
plt.ylabel('Valore Medio', style='italic', **csfont, fontsize=20, labelpad=20)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

# Calcola e plotta le curve ROC aggregate
n_classes = label_binarize(y, classes=sorted(list(df["Trigger"].unique()))).shape[1]
mean_fpr = np.linspace(0, 1, 100)  # Define a common set of FPR values

mean_tpr = dict()
aucs = dict()

for i in range(n_classes):
    tprs = []
    for fold_fpr, fold_tpr, fold_auc in zip(all_fpr, all_tpr, all_roc_auc):
        # Interpolate the TPR values at the mean_fpr values
        tpr = np.interp(mean_fpr, fold_fpr[i], fold_tpr[i])
        tpr[0] = 0.0  # Ensure proper start point
        tprs.append(tpr)
    
    mean_tpr[i] = np.mean(tprs, axis=0)
    mean_tpr[i][-1] = 1.0  # Ensure proper end point
    aucs[i] = auc(mean_fpr, mean_tpr[i])

plt.figure(figsize=(8, 8))
colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown'])
class_names = sorted(list(df["Trigger"].unique()))
for i, color in zip(range(n_classes), colors):
    plt.plot(mean_fpr, mean_tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(class_names[i], aucs[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', style='italic', **csfont, fontsize=20, labelpad=20)
plt.ylabel('True Positive Rate', style='italic', **csfont, fontsize=20, labelpad=20)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# Calcola le metriche medie per ogni classe aggregando le confusion matrix raccolte in all_cm
# Prima, ricava le classi usate (ordinando per coerenza)
class_names = sorted(list(df["Trigger"].unique()))

# Inizializza un dizionario per salvare le metriche di ogni classe da ogni fold
per_class_metrics = {cls: {"accuracy": [], "precision": [], "recall": [], "f1": []} for cls in class_names}

# Per ogni confusion matrix (una per fold) calcola le metriche per ciascuna classe
for cm in all_cm:
    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = np.sum(cm) - TP - FP - FN
        total = TP + FP + FN + TN

        # Calcola le metriche per la classe corrente (con gestione dei casi a denominatore zero)
        acc_cls = (TP + TN) / total if total > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_cls = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        # Salva il valore calcolato per la classe corrente
        per_class_metrics[cls]["accuracy"].append(acc_cls)
        per_class_metrics[cls]["precision"].append(prec)
        per_class_metrics[cls]["recall"].append(rec)
        per_class_metrics[cls]["f1"].append(f1_cls)

# Calcola la media delle metriche per ogni classe su tutti i fold
avg_per_class = {cls: {metric: np.mean(values) for metric, values in metrics.items()} for cls, metrics in per_class_metrics.items()}

# Crea dei plot separati per ciascuna metrica, mostrando i valori medi per ogni classe
metrics_list = ["accuracy", "precision", "recall", "f1"]

# Mappatura dei colori per ciascuna metrica
colors_metric = {
    "accuracy": "lightsalmon",
    "precision": "lightgreen",
    "recall": "lightseagreen",
    "f1": "plum",
}

for metric in metrics_list:
    plt.figure(figsize=(8, 6))
    # Recupera i valori medi per ogni classe per la metrica corrente
    y_values = [avg_per_class[cls][metric] for cls in class_names]
    # Usa il colore associato alla metrica
    plt.bar(class_names, y_values, color=colors_metric[metric], zorder=4)
    plt.ylim(0, 1)
    plt.xlabel('Classe', style='italic', **csfont, fontsize=20, labelpad=20)
    plt.ylabel(metric.capitalize(), style='italic', **csfont, fontsize=20,labelpad=20)
    for i, val in enumerate(y_values):
        plt.text(i, val + 0.01, f"{val:.5f}", ha='center', va='bottom')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Salva il modello
joblib.dump(rf_model, "random_forest_model.pkl")