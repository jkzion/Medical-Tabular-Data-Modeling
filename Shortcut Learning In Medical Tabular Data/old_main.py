import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score



df = pd.read_csv('Shortcut Learning In Medical Tabular Data\heart_disease_uci.csv')


df['fbs'] = df['fbs'].map({True:1, False:0})
df['exang'] = df['exang'].map({True:1, False:0})


X_full = pd.get_dummies(df.drop(columns=['id', 'num']), drop_first=False)


y = (df['num'] > 0).astype(int)


# Generate all combinations of dropping 1, 2, or 3 features
all_features = X_full.columns.tolist()
ablation_combinations = []

for r in range(1, 4):
    ablation_combinations.extend(combinations(all_features, r))

print(f"Total combinations to test: {len(ablation_combinations)}")

# Train and evaluate each combination
results = []

for i, combo in enumerate(ablation_combinations, 1):
    cols_to_drop = list(combo)
    X = X_full.drop(columns=cols_to_drop, errors='ignore')
    
    if (i - 1) % 50 == 0:
        print(f"\nProcessing combination {i}/{len(ablation_combinations)}...")
    
    # split (X may contain NaNs; XGBoost accepts them)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, verbose=0)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    
    dropped_str = ', '.join(cols_to_drop)
    results.append({'dropped_features': dropped_str, 'n_features': X.shape[1], 'accuracy': acc, 'auc': auc})

# Display summary sorted by AUC (descending)
print("\n\n=== SUMMARY (Top 20 by AUC) ===")
summary_df = pd.DataFrame(results)
summary_df = summary_df.sort_values('auc', ascending=False)
print(summary_df.head(20).to_string(index=False))

print("\n=== Bottom 10 by AUC ===")
print(summary_df.tail(10).to_string(index=False))

print(f"\nBest AUC: {summary_df['auc'].max():.4f}")
print(f"Worst AUC: {summary_df['auc'].min():.4f}")
print(f"Mean AUC: {summary_df['auc'].mean():.4f}")


