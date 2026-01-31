from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier
import numpy as np

class Models:
    @staticmethod
    def get_model(model_type, X_train, random_state=42):
        if model_type == 'xgboost':
            return XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state = random_state)
        
        elif model_type == "rf":
            return RandomForestClassifier(random_state=random_state, n_jobs=-1)
        
        elif model_type == "tabpfn":
            if len(X_train) > 10000:
                raise ValueError("TabPFN only supports up to 10,000 training samples")
            if X_train.shape[1] > 100:
                raise ValueError("TabPFN only supports up to 100 features")
            
            return TabPFNClassifier(device='cpu')
        
        else:
            raise ValueError(f"Algorithm '{model_type}' not supported yet....")
        
        
class StackingEnsemble:
    def __init__(self, base_model1, base_model2, folds=5, random_state=42):
        # Keep short attribute names consistent with usage in fit
        self.base_model1 = base_model1
        self.base_model2 = base_model2
        self.folds = folds
        self.random_state = random_state

        self.meta_learner = LogisticRegression(random_state=random_state, max_iter=1000)

        self.base_model1_fit = None
        self.base_model2_fit = None

    def fit(self, X_full, y):
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((len(X_full), 2))

        print(f"   Training stacking ensemble with {self.folds}-fold cross validation....")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y), 1):
            X_train_fold = X_full.iloc[train_idx] if hasattr(X_full, 'iloc') else X_full[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X_full.iloc[val_idx] if hasattr(X_full, 'iloc') else X_full[val_idx]

            model1 = Models.get_model(self.base_model1, X_train_fold, self.random_state)
            model1.fit(X_train_fold, y_train_fold)

            model2 = Models.get_model(self.base_model2, X_train_fold, self.random_state)
            model2.fit(X_train_fold, y_train_fold)

            meta_features[val_idx, 0] = model1.predict_proba(X_val_fold)[:, 1]
            meta_features[val_idx, 1] = model2.predict_proba(X_val_fold)[:, 1]

        # Fit meta learner on out-of-fold predictions
        self.meta_learner.fit(meta_features, y)

        # Refit base models on full data for later predictions
        self.base_model1_fit = Models.get_model(self.base_model1, X_full, self.random_state)
        self.base_model1_fit.fit(X_full, y)

        self.base_model2_fit = Models.get_model(self.base_model2, X_full, self.random_state)
        self.base_model2_fit.fit(X_full, y)

        return self

    def predict(self, X_full):
        pred1 = self.base_model1_fit.predict_proba(X_full)[:, 1]
        pred2 = self.base_model2_fit.predict_proba(X_full)[:, 1]
        meta_features = np.column_stack([pred1, pred2])
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X_full):
        pred1 = self.base_model1_fit.predict_proba(X_full)[:, 1]
        pred2 = self.base_model2_fit.predict_proba(X_full)[:, 1]
        meta_features = np.column_stack([pred1, pred2])
        return self.meta_learner.predict_proba(meta_features) 