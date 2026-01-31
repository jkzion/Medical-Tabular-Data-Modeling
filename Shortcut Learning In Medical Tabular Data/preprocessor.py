import pandas as pd

class Preprocessor:
    def __init__(self, target_column):
        self.target_column = target_column
        
    def prep(self, df, columns_dropping, file_path):
        
        X_full = df.drop(columns=columns_dropping, errors='ignore')
        y = (pd.to_numeric(df[self.target_column], errors='coerce') > 0).astype(int) #df[self.] is mean to be for 'num' only
        
        X_full = X_full.apply(pd.to_numeric, errors='coerce')
        
        return X_full, y
    