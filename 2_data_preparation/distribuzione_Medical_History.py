import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv("./0_data/dataset_imputed.csv")
# Crea il grafico a barre
sns.countplot(x='Medical_History', hue='Trigger', data=df)

# Aggiungi etichette e titolo
plt.xlabel('Medical_History')
plt.ylabel('Count')
plt.title('Distribuzione dei valori Trigger rispetto ad Age')

# Mostra il grafico
plt.show()