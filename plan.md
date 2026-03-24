## Plan: Telco Churn Explainability with SHAP & LIME

**TL;DR**: Build a churn prediction model on the Telco Customer Churn dataset (~7,043 rows, 21 columns) using **Gradient Boosted Trees (XGBoost)**, then apply SHAP and LIME to explain predictions. XGBoost is chosen because it handles mixed feature types well, performs strongly on tabular data, and has native SHAP integration via TreeExplainer (exact, fast).

### Phase 1: Data Preparation

1. **Load & inspect** — Read CSV with pandas. Drop `customerID` (non-predictive identifier).
2. **Handle `TotalCharges`** — Column is stored as string; some rows have whitespace (" ") instead of a number (these correspond to tenure=0 new customers). Convert to float, fill missing/blank with 0.
3. **Encode target** — Map `Churn` column: "Yes" → 1, "No" → 0.
4. **Encode categorical features** — Use `LabelEncoder` or `OrdinalEncoder` for binary columns (gender, Partner, Dependents, PhoneService, PaperlessBilling). Use `pd.get_dummies` / `OneHotEncoder` for multi-class columns (MultipleLines, InternetService, Contract, PaymentMethod, and the six service columns that have "No internet service" as a third value: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies).
5. **Train/test split** — 80/20 stratified split on Churn (preserves ~26.5% churn rate in both sets).
6. **Scale numeric features** — StandardScaler on `tenure`, `MonthlyCharges`, `TotalCharges` (needed for LIME's kernel-based explanations to be well-behaved; XGBoost itself doesn't need it, but won't be hurt).

### Phase 2: Model Training

7. **Train XGBoost classifier** — `XGBClassifier` with reasonable defaults (`n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `scale_pos_weight` ~= ratio of non-churn/churn to handle class imbalance).
8. **Evaluate** — Print accuracy, precision, recall, F1, AUC-ROC, and confusion matrix on test set.

### Phase 3: Explainability — SHAP

9. **SHAP TreeExplainer** — Use `shap.TreeExplainer(model)` for exact Shapley values.
10. **Global explanations** — `shap.summary_plot` (beeswarm) to show feature importance + direction of effect across all test samples.
11. **Single-prediction explanation** — `shap.waterfall_plot` for one churned and one non-churned customer to show per-feature contribution.
12. **Dependence plot** — `shap.dependence_plot` for top 2-3 features (e.g., tenure, Contract, MonthlyCharges) to show interaction effects.

### Phase 4: Explainability — LIME

13. **LIME LimeTabularExplainer** — Initialize with training data, feature names, class names, and `mode='classification'`.
14. **Explain individual predictions** — Pick the same two customers from step 11. Call `explain_instance` and show `.as_pyplot_figure()`.
15. **Compare SHAP vs LIME** — Side-by-side for those same instances; note agreements and divergences.

### Phase 5: Interpretation & Reporting

16. **Summarize findings** — Key drivers of churn (expected: short tenure, month-to-month contracts, high monthly charges, fiber optic internet, no tech support/online security). Note any surprising features.
17. **SHAP vs LIME comparison** — Discuss consistency: SHAP gives exact global + local; LIME approximates locally. Note where they agree (builds trust) and where they diverge (flags model complexity).

### Relevant Files

- [WA_Fn-UseC_-Telco-Customer-Churn.csv](WA_Fn-UseC_-Telco-Customer-Churn.csv) — Source data (7,043 rows, 21 columns). Target: `Churn`. Numeric: `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`. Categorical: remaining 16 columns.
- **New file to create**: `churn_explainability.ipynb` (or `.py`) — All code in a single notebook/script.

### Verification

1. Confirm no data leakage — `customerID` dropped, no target leakage from features.
2. Confirm `TotalCharges` has no NaN/blank after conversion — `df['TotalCharges'].isna().sum() == 0`.
3. Model AUC-ROC should be ≥ 0.80 (typical for this dataset with XGBoost).
4. SHAP values sum to model output minus base value — verify with `shap_values[0].sum() + explainer.expected_value ≈ model.predict_proba(X)[0]`.
5. LIME explanation should show broadly consistent top features with SHAP for the same instance.

### Decisions

- **Algorithm: XGBoost** — Best fit because: (a) strong tabular performance, (b) native `shap.TreeExplainer` gives exact Shapley values in O(TLD²) vs kernel SHAP's approximation, (c) handles class imbalance via `scale_pos_weight`.
- **Alternative considered: Random Forest** — Also supports TreeExplainer, but typically lower AUC on this dataset. Could be swapped easily.
- **Alternative considered: Logistic Regression** — Simpler and inherently interpretable, but less interesting for demonstrating SHAP/LIME's value (explainability tools shine on complex models).
- **Scope**: Single notebook, no hyperparameter tuning (grid search), no deployment. Focus is on explainability, not model optimization.

### Python Dependencies

- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `lime`, `matplotlib`
