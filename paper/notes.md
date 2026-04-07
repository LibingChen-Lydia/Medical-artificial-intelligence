# Project
Heart Failure Survival Prediction with Classical ML and a Simple MLP Baseline

# Dataset
- Dataset: Heart Failure Clinical Records
- Sample size: 299
- Total variables: 13
- Target: DEATH_EVENT
- Outcome distribution: 203 survived (67.89%), 96 died (32.11%)

# Preprocessing
- No missing values were found.
- Continuous variables: age, creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium, time
- Binary variables: anaemia, diabetes, high_blood_pressure, sex, smoking, DEATH_EVENT
- IQR-based outlier detection identified notable outliers in creatinine_phosphokinase, serum_creatinine, platelets, serum_sodium, and ejection_fraction.
- Outliers were not directly deleted because extreme values may reflect real clinical conditions.
- A winsorized version was created for robustness comparison.
- Continuous variables were standardized for scale-sensitive models such as Logistic Regression, SVM, and MLP.
- Tree-based models were also evaluated because they do not strongly rely on feature scaling.

# Statistical analysis
- Significant continuous variables from Welch's t-test:
  - time: p = 2.343276e-22
  - ejection_fraction: p = 9.647153e-06
  - age: p = 4.735215e-05
  - serum_creatinine: p = 6.398962e-05
  - serum_sodium: p = 1.872325e-03
- No binary variable reached statistical significance in chi-square testing.

# Feature importance
- Random forest top features:
  1. time (0.351394)
  2. serum_creatinine (0.135949)
  3. ejection_fraction (0.121653)
  4. age (0.092868)
  5. creatinine_phosphokinase (0.082592)

# Combined top risk factors
- time
- ejection_fraction
- serum_creatinine

# Modeling: Experiment 1 (scaled data only)
- Logistic Regression: Accuracy 0.8167, Precision 0.7857, Recall 0.5789, F1 0.6667, AUC 0.8614
- SVM: Accuracy 0.7667, Precision 0.6923, Recall 0.4737, F1 0.5625, AUC 0.8742
- KNN: Accuracy 0.7167, Precision 0.6000, Recall 0.3158, F1 0.4138, AUC 0.7824
- Random Forest: Accuracy 0.8167, Precision 0.7857, Recall 0.5789, F1 0.6667, AUC 0.8960
- XGBoost: Accuracy 0.8333, Precision 0.8462, Recall 0.5789, F1 0.6875, AUC 0.8472
- MLP: Accuracy 0.8000, Precision 0.7333, Recall 0.5789, F1 0.6471, AUC 0.8562

# Modeling: Experiment 2 (data sensitivity)
- Random Forest:
  - raw: Accuracy 0.8000, Precision 0.7692, Recall 0.5263, F1 0.6250, AUC 0.8935
  - winsorized: Accuracy 0.8167, Precision 0.7857, Recall 0.5789, F1 0.6667, AUC 0.8864
  - scaled: Accuracy 0.8167, Precision 0.7857, Recall 0.5789, F1 0.6667, AUC 0.8960
- XGBoost:
  - raw: Accuracy 0.8333, Precision 0.8462, Recall 0.5789, F1 0.6875, AUC 0.8472
  - winsorized: Accuracy 0.8333, Precision 0.8462, Recall 0.5789, F1 0.6875, AUC 0.8562
  - scaled: Accuracy 0.8333, Precision 0.8462, Recall 0.5789, F1 0.6875, AUC 0.8472

# Figures and tables already available
- figures/correlation_heatmap.png
- figures/feature_importance.png
- figures/experiment1_roc.png
- figures/experiment2_auc_bar.png
- tables/continuous_statistics_table.tex
- tables/categorical_statistics_table.tex
- results/continuous_t_test_results.csv
- results/binary_chi_square_results.csv
- results/feature_importance_results.csv
- results/experiment1_model_comparison.csv
- results/experiment2_data_sensitivity.csv

# Important writing notes
- This is a compact course paper, not a novelty-driven research paper.
- Do not claim causal inference.
- Be cautious when discussing the variable "time"; note that it is highly predictive in this dataset and should be interpreted carefully in the discussion.
- Use an academic medical ML tone.