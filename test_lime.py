import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import lime, lime.lime_tabular

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling']:
    df[col] = LabelEncoder().fit_transform(df[col])
df = pd.get_dummies(df, columns=[
    'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
    'Contract','PaymentMethod'])
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
X = df.drop('Churn', axis=1)
y = df['Churn']
feature_names = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for col in ['tenure','MonthlyCharges','TotalCharges']:
    s = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[col] = s.fit_transform(X_train[[col]])
    X_test[col] = s.transform(X_test[[col]])

model = XGBClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
print('Model trained')

# Test 1: No discretize, no categorical_features (baseline)
print('\n--- Test 1: discretize_continuous=False, no categorical_features ---')
try:
    exp = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=feature_names,
        class_names=['No Churn','Churn'], mode='classification',
        random_state=42, discretize_continuous=False)
    churner_idx = int(np.where((y_test.values==1) & (model.predict(X_test)==1))[0][0])
    result = exp.explain_instance(X_test.values[churner_idx], model.predict_proba,
                                   num_features=5, num_samples=200)
    print('SUCCESS:', result.as_list(label=1)[:3])
except Exception as e:
    print('FAIL:', type(e).__name__, str(e)[:200])

# Test 2: With categorical_features, discretize_continuous=False
print('\n--- Test 2: discretize_continuous=False, with categorical_features ---')
try:
    continuous_feature_names = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_idx = [i for i, fn in enumerate(feature_names) if fn not in continuous_feature_names]
    exp2 = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=feature_names,
        class_names=['No Churn','Churn'], mode='classification',
        categorical_features=cat_idx,
        random_state=42, discretize_continuous=False)
    result2 = exp2.explain_instance(X_test.values[churner_idx], model.predict_proba,
                                     num_features=5, num_samples=200)
    print('SUCCESS:', result2.as_list(label=1)[:3])
except Exception as e:
    print('FAIL:', type(e).__name__, str(e)[:200])
