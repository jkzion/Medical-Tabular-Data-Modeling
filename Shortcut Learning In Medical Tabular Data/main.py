import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import datetime

from preprocessor import Preprocessor
from models import Models
from evaluator import Evaluator

def menu():
    print("\n"*5 + "_"*30)
    print("TABULAR MEDICAL DATA ABLATION TOOL")
    print("_"*30)
    print("MENU")
    print("1. Standard Run (Baseline Only)")
    print("2. Single Variable Ablation Run")
    print("3. Multi Variable Ablation Run")
    print("4. Exit")
    
    menu_choice = input("\nSelect an option (1-4): ")
    print("_"*30)
    
    model_type = select_model()
    
    column_ablate_no = 1
    if menu_choice == '3':
        while True:
            try:
                column_ablate_no = int(input("\nHow many variables to drop at most: "))
                if column_ablate_no <1:
                    print("Invalid input! Please enter a positive number.")
                    continue
                break
            except ValueError:
                print("Invalid input! Please enter a whole number.")
    print("_"*30)
    

    print("CHOOSE A PREDICTION OUTPUT:")
    print("1. ACC and AUC")
    print("2. ACC only")
    print("3. AUC only")
    output_choice = input("\nSelect an option (1-3): ")
    print("_"*30)
    
    if output_choice == '2':
        output_type = ['acc']
    elif output_choice == '3':
        output_type = ['auc']
    else:
        output_type = ['acc', 'auc']
        
    return menu_choice, output_type, column_ablate_no, model_type
    
def select_model():
    print("Choose a model: ")
    print("1. XGBoost")
    print("2. Random Forest")
    print("3. TabPFN")
    print("4. Stacking Ensemble")
    model_choice = input("\nSelect an option (1-4): ")
    print("_"*30)
    
    if model_choice == '1':
        return 'xgboost'
    elif model_choice == '2':
        return 'rf'
    elif model_choice == '3':
        return 'tabpfn'
    elif model_choice == '4':
        print("Select two base models for stacking:")
        print("1. XGBoost")
        print("2. Random Forest")
        print("3. TabPFN")
        
        base_choice1 = input("\nSelect first base model (1-3): ")
        base_choice2 = input("\nSelect second base model (1-3): ")
        
        model_map = {'1': 'xgboost', '2': 'rf', '3': 'tabpfn'}
        base_model1 = model_map.get(base_choice1, 'xgboost')
        base_model2 = model_map.get(base_choice2, 'tabpfn')
        
        print("_"*30)
        return ('stacking', base_model1, base_model2)
    else:
        print("No valid model selected. Defaulting to XGBoost...")
        return 'xgboost'
    
def config():
    # UCI Heart Disease Preprocessing Config (change this for other datasets)
    file_path = "Shortcut Learning In Medical Tabular Data/heart_disease_uci.csv"
    target_column = "num"
    columns_dropping = ["id", "slope", "ca", "thal", target_column]
    
    
    return file_path, target_column, columns_dropping


def run(menu_choice, output_type, file_path, target_column, columns_dropping, model_type, column_ablate_no):
    
    df = pd.read_csv(file_path)
    preprocessor = Preprocessor(target_column)
    X_full, y = preprocessor.prep(df, columns_dropping, file_path)
    
    evaluator = Evaluator(random_state=42)
    
    
    if isinstance(model_type, tuple) and model_type[0] == 'stacking':
        model_name = f"STACKING ({model_type[1].upper()} + {model_type[2].upper()})"
        model_filename = f"stacking_{model_type[1]}_{model_type[2]}"
    else:
        model_name = model_type.upper()
        model_filename = model_type
    
    print(f"\nResults for {model_name}: ")
    print("_"*30)
    
    baseline_results = evaluator.evaluate(X_full, y, model_type, output_type)
    print_results("Baseline", baseline_results)
    
    
    if menu_choice in ['2', '3']:
        results_map = evaluator.run_ablation(X_full, y, model_type, output_type, column_ablate_no)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"ablation_{model_filename}_{timestamp}.csv"
        evaluator.save_results(results_map, baseline_results, csv_filename)
        
        print("\n" + "_"*30)
        print("ABLATION STUDY SUMMARY")
        print("_"*30)
        
        summary = evaluator.analyse_results(results_map, baseline_results, output_type)
        print_summary(summary, output_type)
        



def print_results(label, scores):
    output_str = f"{label:20} | "
    if 'acc' in scores:
        output_str += f"ACC: {scores['acc']:.4f}"
    if 'auc' in scores:
            if 'acc' in scores:
                output_str += " | "
            output_str += f"AUC: {scores['auc']:.4f}"
            
    print(output_str)
    
def print_summary(summary, output_type):
    for metric in output_type:
        metric_upper = metric.upper()
        stats = summary[metric]
        
        print("\n" + "_"*30)
        print(f"{metric_upper} ANALYSIS")
        print("_"*30)
    
        print("Stats:")
        print(f"   Baseline {metric_upper}: {stats['baseline']:.4f}")
        print(f"   Mean {metric_upper}: {stats['mean']:.4f} (±{stats['std']:.4f})")
        print(f"   Best {metric_upper}:     {stats['best_score']:.4f}")
        print(f"   Worst {metric_upper}:    {stats['worst_score']:.4f}")
        
        print(f"\nBaseline Comparison:")
        print(f"   Ablations beating baseline:   {stats['beat_baseline']}/{stats['total_ablations']}")
        print(f"   Ablations matching baseline:  {stats['match_baseline']}/{stats['total_ablations']}")
        
        print(f"\nBest Performing Ablations (Top 5):")
        for i, (label, scores) in enumerate(stats['top_5'], 1):
            print(f"   {i}. {label:40} | {metric_upper}: {scores[metric]:.4f}")
        
        print(f"\nWorst Performing Ablations (Bottom 5):")
        for i, (label, scores) in enumerate(stats['bottom_5'], 1):
            print(f"   {i}. {label:40} | {metric_upper}: {scores[metric]:.4f}")
        
        print(f"\nKey Insights:")
        if stats['best_score'] > stats['baseline']:
            print(f"   Removing some variables IMPROVED performance!")
            print(f"     Best improvement: +{(stats['best_score'] - stats['baseline']):.4f}")
            print(f"     When removed: {stats['best_ablations'][0].replace('Without ', '')}")
        else:
            print(f"   No ablation improved over baseline")
        
        if stats['worst_score'] < stats['baseline']:
            drop = stats['baseline'] - stats['worst_score']
            print(f"   Worst drop: -{drop:.4f}")
            print(f"     Critical variables: {stats['worst_ablations'][0].replace('Without ', '')}")
    
if __name__ == "__main__":
    file_path, target_column, columns_dropping = config()
    menu_choice, output_type, column_ablate_no, model_type = menu()
    run(menu_choice, output_type, file_path, target_column, columns_dropping, model_type, column_ablate_no)