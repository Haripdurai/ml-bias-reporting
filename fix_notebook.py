import json

nb_path = '/Users/hari.durai/Documents/AIApprenticeship/Explainability_Bias_Reporting_24March2026/handson-bias-reporting/churn_explainability.ipynb'

with open(nb_path, 'r') as f:
    nb = json.load(f)

NEW_LIME_INIT = [
    "X_train_arr = X_train.values\n",
    "X_test_arr  = X_test.values\n",
    "\n",
    "# Declare all non-numeric-continuous features as categorical\n",
    "# This prevents LIME from sampling truncnorm for binary 0/1 columns\n",
    "continuous_feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
    "categorical_feature_indices = [\n",
    "    i for i, fn in enumerate(feature_names)\n",
    "    if fn not in continuous_feature_names\n",
    "]\n",
    "print(f'Categorical features: {len(categorical_feature_indices)}')\n",
    "print(f'Continuous features: {continuous_feature_names}')\n",
    "\n",
    "# discretize_continuous=False avoids the LIME discretizer's zero-std bug\n",
    "lime_explainer = lime.lime_tabular.LimeTabularExplainer(\n",
    "    training_data=X_train_arr,\n",
    "    feature_names=feature_names,\n",
    "    class_names=['No Churn', 'Churn'],\n",
    "    categorical_features=categorical_feature_indices,\n",
    "    discretize_continuous=False,\n",
    "    mode='classification',\n",
    "    random_state=42\n",
    ")\n",
    "print('LIME explainer initialized.')\n",
    "print(f'  Training data shape: {X_train_arr.shape}')\n",
]

for cell in nb['cells']:
    if cell['id'] == 'cell-lime-init':
        cell['source'] = NEW_LIME_INIT
        print("Fixed cell-lime-init")
        break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
print("Saved OK")
