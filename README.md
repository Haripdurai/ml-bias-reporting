

# Telco Customer Churn: XGBoost + SHAP & LIME Explainability
# ml-bias-reporting

An end-to-end machine learning pipeline that predicts customer churn and explains the model's decisions using **SHAP** and **LIME** — two of the most widely used model explainability techniques.

---

## Objective

- Train a high-performing churn classifier on the IBM Telco Customer Churn dataset
- Use **SHAP** (global + local explanations) to understand *what drives churn across all customers*
- Use **LIME** (local explanations) to explain *why the model made a specific prediction* for an individual customer
- Compare SHAP and LIME side-by-side to build trust in the explanations

---

## Dataset

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

| Property | Value |
|----------|-------|
| Source | IBM Sample Data (Telco Customer Churn) |
| Rows | 7,043 customers |
| Columns | 21 (20 features + 1 target) |
| Target | `Churn` (Yes / No) |
| Churn Rate | ~26.5% |

**Key features:** `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `OnlineSecurity`, `TechSupport`, `PaymentMethod`, and more.

---

## ML Algorithm: XGBoost

**Why XGBoost?**
- Best-in-class performance on tabular data
- Native exact SHAP support via `shap.TreeExplainer` (no approximation)
- Handles class imbalance with `scale_pos_weight`

---

## Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.8282** |
| F1-Score | 0.6117 |
| Recall | 0.7433 |
| Precision | 0.5196 |
| Accuracy | 0.7495 |

---

## Top 5 Churn Drivers (SHAP Global Analysis)

| Rank | Feature | Mean \|SHAP\| | Interpretation |
|------|---------|--------------|----------------|
| 1 | `Contract_Month-to-month` | 0.8254 | No lock-in → easiest to leave |
| 2 | `tenure` | 0.6534 | New customers are highest risk |
| 3 | `MonthlyCharges` | 0.4270 | High bills drive dissatisfaction |
| 4 | `TotalCharges` | 0.3297 | Correlated with tenure |
| 5 | `OnlineSecurity_No` | 0.2764 | Less value from service |

---

## Explainability Methods

### SHAP (SHapley Additive exPlanations)

SHAP uses **game theory (Shapley values)** to fairly assign credit to each feature for a model's prediction.

- Every feature gets a score: how much it pushed the prediction **up** (toward churn) or **down** (away from churn)
- The sum of all SHAP values + the baseline always equals the model's exact output — mathematically guaranteed
- `TreeExplainer` gives **exact** values for tree models (XGBoost, Random Forest, LightGBM)

**Plots generated:**
| File | What it shows |
|------|---------------|
| `shap_summary_beeswarm.png` | Every customer as a dot — colour = feature value, x-axis = SHAP value |
| `shap_bar.png` | Mean \|SHAP\| per feature — overall importance ranking |
| `shap_waterfall_churner.png` | Step-by-step breakdown for one churner |
| `shap_waterfall_non_churner.png` | Step-by-step breakdown for one non-churner |
| `shap_dependence_tenure.png` | How tenure's value affects churn prediction across all customers |
| `shap_dependence_MonthlyCharges.png` | Same for MonthlyCharges |
| `shap_dependence_Contract_Month_to_month.png` | Same for Contract type |

### LIME (Local Interpretable Model-agnostic Explanations)

LIME explains one prediction at a time by **creating ~1,000 slightly-modified versions** of a customer, asking the model for predictions on all of them, then fitting a simple **linear model** to those nearby predictions.

- Works on **any** model — model-agnostic
- Gives an **approximation** (unlike SHAP's exact values)
- Useful for sanity-checking individual predictions

**Plots generated:**
| File | What it shows |
|------|---------------|
| `lime_churner.png` | Feature weights for the churner customer |
| `lime_non_churner.png` | Feature weights for the non-churner customer |
| `shap_vs_lime_comparison.png` | Side-by-side bar chart comparing SHAP and LIME for the same customer |

### SHAP vs LIME Summary

| | SHAP | LIME |
|--|------|------|
| Method | Game theory (Shapley values) | Local linear model |
| Exact? | Yes (TreeExplainer) | No — approximation |
| Scope | Global + Local | Local only |
| Model-agnostic? | No (tree-specific) | Yes (any model) |
| Speed | Fast for trees | Slower (~1k samples per prediction) |
| When to trust | Always for tree models | When it agrees with SHAP |

When both methods **agree on the top features** for a customer, the explanation is robust and can be confidently acted on.

---

## Project Structure

```
handson-bias-reporting/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Source dataset
├── churn_explainability.ipynb              # Main notebook (executed, with outputs)
├── confusion_matrix.png                    # Model evaluation
├── shap_summary_beeswarm.png               # SHAP global — beeswarm
├── shap_bar.png                            # SHAP global — bar chart
├── shap_waterfall_churner.png              # SHAP local — churner
├── shap_waterfall_non_churner.png          # SHAP local — non-churner
├── shap_dependence_tenure.png              # SHAP dependence — tenure
├── shap_dependence_MonthlyCharges.png      # SHAP dependence — monthly charges
├── shap_dependence_Contract_Month_to_month.png  # SHAP dependence — contract
├── lime_churner.png                        # LIME — churner explanation
├── lime_non_churner.png                    # LIME — non-churner explanation
└── shap_vs_lime_comparison.png             # Side-by-side SHAP vs LIME
```

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| 1 | Import libraries |
| 2 | Load & prepare dataset (encode, split, scale) |
| 3 | Train XGBoost classifier + evaluate |
| 4 | Compute SHAP values with `TreeExplainer` |
| 5 | SHAP Summary Plot (global importance) |
| 6 | SHAP Waterfall Plot (individual explanation) |
| 7 | SHAP Dependence Plot (feature effect curves) |
| 8 | LIME explainer setup |
| 9 | LIME individual explanations |
| 10 | LIME visualisations |
| 11 | SHAP vs LIME side-by-side comparison + summary |

---

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost shap lime matplotlib seaborn notebook

# macOS only: XGBoost requires OpenMP
brew install libomp

# Launch notebook
jupyter notebook churn_explainability.ipynb
```

**Python:** 3.14 · **Key packages:** `xgboost==3.2.0`, `shap==0.51.0`, `lime==0.2.0.1`, `scikit-learn==1.8.0`
