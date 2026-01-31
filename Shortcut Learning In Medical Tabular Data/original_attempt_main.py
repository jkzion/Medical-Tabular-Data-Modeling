import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score



df = pd.read_csv('Shortcut Learning In Medical Tabular Data\heart_disease_uci.csv')


df['fbs'] = df['fbs'].map({True:1, False:0})
df['exang'] = df['exang'].map({True:1, False:0})


X_full = pd.get_dummies(df.drop(columns=['id', 'num']), drop_first=False)


y = (df['num'] > 0).astype(int)


# Define feature ablation variants
variants = {
    'full': X_full,
    'no_age_sex': X_full.drop(columns=['age', 'sex_Female', 'sex_Male'], errors='ignore'),
    'no_sex': X_full.drop(columns=['sex_Female', 'sex_Male'], errors='ignore'),
    'no_age': X_full.drop(columns=['age'], errors='ignore'),
    'no_origin': X_full.drop(columns=['origin'], errors='ignore'),
    'no_cp': X_full.drop(columns=['cp'], errors='ignore'),
    'no_trestbps': X_full.drop(columns=['trestbps'], errors='ignore'),
    'no_oldpeak': X_full.drop(columns=['oldpeak'], errors='ignore'),
    'no_chol': X_full.drop(columns=['chol'], errors='ignore'),
    'no_fbs': X_full.drop(columns=['fbs'], errors='ignore'),
    'no_exang': X_full.drop(columns=['exang'], errors='ignore'),
}

# Train and evaluate each variant
results = []

for name, X in variants.items():
    print(f"\n--- {name} (features: {X.shape[1]}) ---")
    
    # split (X may contain NaNs; XGBoost accepts them)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    
    print(f'acc: {acc:.4f}, auc: {auc:.4f}')
    results.append({'variant': name, 'n_features': X.shape[1], 'accuracy': acc, 'auc': auc})

# Display summary
print("\n\n=== SUMMARY ===")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

