# A4 results — auto-generated summary

- Employers retained after target construction: **7,971**
- Features used: **28** (policy flags + engineered + one-hot division)
- Class balance (positive = below industry median): **49.3%** of 7,971 employers

## Hold-out test-set metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RandomForest | 0.505 | 0.499 | 0.557 | 0.526 | 0.512 |
| NaiveBayes | 0.473 | 0.425 | 0.189 | 0.262 | 0.467 |
| LogisticRegression | 0.502 | 0.496 | 0.534 | 0.514 | 0.498 |
| DecisionTree | 0.513 | 0.506 | 0.578 | 0.539 | 0.513 |

**Best model on hold-out F1:** DecisionTree (F1=0.539, ROC-AUC=0.513).

## 5-fold CV (mean ± std)

| Model | F1 | ROC-AUC | Recall |
|---|---|---|---|
| RandomForest | 0.530 ± 0.010 | 0.518 ± 0.011 | 0.563 ± 0.012 |
| NaiveBayes | 0.332 ± 0.239 | 0.467 ± 0.010 | 0.359 ± 0.355 |
| LogisticRegression | 0.521 ± 0.005 | 0.495 ± 0.007 | 0.547 ± 0.011 |
| DecisionTree | 0.528 ± 0.010 | 0.509 ± 0.010 | 0.554 ± 0.012 |

## Pairwise paired t-tests on per-fold F1

| A | B | mean F1 A | mean F1 B | t | p | sig (0.05) | better |
|---|---|---|---|---|---|---|---|
| RandomForest | NaiveBayes | 0.530 | 0.332 | 1.86 | 0.1370 | False | RandomForest |
| RandomForest | LogisticRegression | 0.530 | 0.521 | 1.81 | 0.1440 | False | RandomForest |
| RandomForest | DecisionTree | 0.530 | 0.528 | 0.23 | 0.8292 | False | RandomForest |
| NaiveBayes | LogisticRegression | 0.332 | 0.521 | -1.78 | 0.1495 | False | LogisticRegression |
| NaiveBayes | DecisionTree | 0.332 | 0.528 | -1.78 | 0.1496 | False | DecisionTree |
| LogisticRegression | DecisionTree | 0.521 | 0.528 | -1.61 | 0.1820 | False | DecisionTree |