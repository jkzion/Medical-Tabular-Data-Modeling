import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Shortcut Learning In Medical Tabular Data\heart_disease_uci.csv')

# Convert boolean text columns to numeric
# Convert boolean columns to numeric (True->1, False->0, NaN->NaN)
df['fbs'] = df['fbs'].astype('bool').astype(int)
df['exang'] = df['exang'].astype('bool').astype(int)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal'], drop_first=False)
numeric_df = df_encoded.select_dtypes(include=[np.number])

# Then create heatmap with all variables
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1, fmt='.2f', 
            cbar_kws={'label': 'Correlation'}, square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Heart Disease Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()



