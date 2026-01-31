import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import itertools
import csv
import os

from models import Models

class Evaluator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def evaluate(self, X_full, y, model_type, output_type):
        results = {}
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        if isinstance(model_type, tuple) and model_type[0] == 'stacking':
            from models import StackingEnsemble
            _, base_model1, base_model2 = model_type
            model = StackingEnsemble(base_model1, base_model2, folds=5, random_state=self.random_state)
        
        else:
            model = Models.get_model(model_type, X_train, self.random_state)
        
        
        model.fit(X_train, y_train)
        
        if 'acc' in output_type:
            pred_acc = model.predict(X_test)
            results['acc'] = accuracy_score(y_test, pred_acc)
            
        if 'auc' in output_type:
            pred_auc = model.predict_proba(X_test)[:, 1]
            results['auc'] = roc_auc_score(y_test, pred_auc)
            
        return results
    
    def run_ablation(self, X_full, y, model_type, output_type, column_ablate_no):
        ablation_results = {}
        
        for n_vars in range(1, column_ablate_no + 1):
            print(f"\nTesting ablations with {n_vars} variable/s dropped...")
            
            column_combos = itertools.combinations(X_full.columns, n_vars)
            
            for cols_to_drop in column_combos:
                X_ablated = X_full.drop(columns=list(cols_to_drop))
                
                results = self.evaluate(X_ablated, y, model_type, output_type)
                
                dropped_cols_str = ", ".join(cols_to_drop)
                label = f"Without {dropped_cols_str}"
                
                ablation_results[label] = results
                
        return ablation_results
        
    def save_results(self, results_dict, baseline_results, filename="ablation_results.csv"):
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        rows = []
        
        baseline_row = {
            'ablation_type': 'Baseline',
            'variables_dropped': 'None',
            'num_vars_dropped': 0,
        }
        baseline_row.update(baseline_results)
        rows.append(baseline_row)
        
        for label, scores in results_dict.items():
            vars_dropped = label.replace("Without ", "")
            num_dropped = len(vars_dropped.split(", "))
            
            row = {
                'ablation_type': label,
                'variables_dropped': vars_dropped,
                'num_vars_dropped': num_dropped,
            }
            row.update(scores)
            rows.append(row)
        
            
        if rows:
            fieldnames = ['ablation_type', 'variables_dropped', 'num_vars_dropped']
            metric_cols = [k for k in rows[0].keys() if k not in fieldnames]
            fieldnames.extend(metric_cols)
            
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
            print(f"\nResults saved to: {filepath}")
            return filepath
        
        return None
    
    def analyse_results(self, results_dict, baseline_results, output_type):
        if not results_dict:
            return None
        
        summary = {}
        
        for metric in output_type:
            metric_values = [scores[metric] for scores in results_dict.values()]
            baseline_value = baseline_results[metric]
            
            
            best_score = max(metric_values)
            worst_score = min(metric_values)
            
            best_ablations = [label for label, scores in results_dict.items() if scores[metric] == best_score]
            worst_ablations = [label for label, scores in results_dict.items() if scores[metric] == worst_score]
            
            
            mean_score = sum(metric_values) / len(metric_values)
            std_score = (sum((x - mean_score) ** 2 for x in metric_values) / len(metric_values)) ** 0.5
            
            
            beat_baseline = sum(1 for score in metric_values if score > baseline_value)
            match_baseline = sum(1 for score in metric_values if score == baseline_value)
            
            
            sorted_results = sorted(results_dict.items(), key=lambda x: x[1][metric], reverse=True)
            top_5 = sorted_results[:5]
            bottom_5 = sorted_results[-5:]
            
            summary[metric] = {
                'baseline': baseline_value,
                'best_score': best_score,
                'best_ablations': best_ablations,
                'worst_score': worst_score,
                'worst_ablations': worst_ablations,
                'mean': mean_score,
                'std': std_score,
                'beat_baseline': beat_baseline,
                'match_baseline': match_baseline,
                'total_ablations': len(metric_values),
                'top_5': top_5,
                'bottom_5': bottom_5
            }
        
        return summary