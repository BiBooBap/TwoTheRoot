import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Leggi il DataFrame
df = pd.read_csv('dataset_imputed.csv')

# Crea il grafico KDE
sns.kdeplot(x='Age', hue='Trigger', data=df, common_norm=False)

# Aggiungi etichette e titolo
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribuzione di Age rispetto a Trigger')

# Aggiungi il piano cartesiano
plt.axhline(0, color='black', linewidth=0.5)  # Asse x
plt.axvline(0, color='black', linewidth=0.5)  # Asse y

# Mostra il grafico
plt.show()