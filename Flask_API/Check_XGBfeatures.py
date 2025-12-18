import joblib, os
BASE = os.path.join(os.path.dirname(__file__))
m = joblib.load(os.path.join(BASE,'model_xgb_final.joblib'))
print('model.feature_names_in_ (XGBoost/sklearn):', getattr(m, 'feature_names_in_', None))
try:
    cols = joblib.load(os.path.join(BASE,'X_train_columns.joblib'))
    print('X_train_columns.joblib (saved):', cols)
except Exception as e:
    print('Failed to load X_train_columns.joblib:', e)