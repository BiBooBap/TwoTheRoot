import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv("./0_data/dataset_imputed.csv")
# Crea il grafico a barre
sns.countplot(x='Gender', hue='Trigger', data=df)

# Aggiungi etichette e titolo
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribuzione dei valori Trigger rispetto a Gender')

# Mostra il grafico
plt.show()