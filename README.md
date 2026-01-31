Key files:
- main.py
- preprocessor.py
- models.py
- evaluator.py



# History:

**20/1/26**

- Created a basic XGBoost program that outputs ACC and AUC of the UCI heart disease dataset. Basic ablation, dropping a few key demographics.
- Modified the program to try every combination of dropping up to 3 variables. With 14 total variables, this meant running through 14c1+14c2+14c3=469 combinations.
- Experimented with the program and reflected the reliability of XGBoost with regards to algorithmic bias

**27/1/26**

Changes:
- Modular multi-class system (main.py, preprocessor.py, models.py, evaluator.py)
- Easier to swap out and work with diff datasets (still needs manual editing of code)
- Easier to preprocess datasets (still could improve alot)
- Menu that lets you choose:
  - Model (XGBoost, Random Forest, TabPFN, I want to add CatBoost later)
  - Output type (ACC, AUC, both)
  - Up to n variables to be ablated (no limit set, could add warning for high processing time)
- Saves the output in a csv file
- Outputs a summary in the terminal with:
  - Baseline stats (mean, best, worst ACC/AUC)
  - Number of Ablations beating or matching the baseline
  - Top/Bottom 5 ablations in terms of ACC/AUC
  - Some key insights (if removing variables improved performance, critical variables)
 
Notes:
- While XGBoost is more of a gold standard, TabPFN appears to now be outperforming with these smaller medical datasets (0.8711 vs 0.8528 AUC with my initial testing)
- oldpeak and exang were the most biologically significant variables for both XGBoost and TabPFN.
- Removing the hospital location variable “dataset” seems to improve the results for both models.
- Swiss Cheese problem: TabPFN probably is fairly affected by NaN datapoints, so removing some of the variables like “ca” with 611/920 missing entries would be my next step.
- TabPFN is more sensitive to noise, but more resilient to loss of individual important features
- Would the NaN count be worth adding as an extra feature? What about for other datasets than the UCI heart disease one?

Project Directions:
- Do modern Tabular Foundation Models (TFMs) like TabPFN provide a more robust and clinically valid "prior" for medical data than the 2022 state-of-the-art tree-based models?
- Prove that TFMs outperform traditional models with small data
  - Prove that the TabPFN generalizes across different hospitals better than XGBoost; Solving the Transfer Problem
- Identify if 'dataset shortcuts' create hidden biases in medical diagnosis
  - Create a fairness score for each model


**31/1/26**

Added to the code:
- Stacking ensemble option for any 2 models.
- Stratified K-fold cross-validation, 5 folds


Notes:
- I don’t fully understand these results or its implications yet. I am concerned the stacking may not be reliable or have done exactly what I intended. Small datasets might not work well with this kind of ensembling, despite it being better for TabPFN.
- Would just a simple weighted average be better in this case?
- I wonder how much overlap there was in the results between XGBoost and TabPFN, and if XGBoost just ended up polluting the results rather than providing a good alternative variety of options.
- What is the best way to ensemble with TabPFN? Is stacking via k-fold cross-validation the best option for this use case?
