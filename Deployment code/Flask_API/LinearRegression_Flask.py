from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the pre-trained model, scaler, and encoded columns (file-relative paths)
BASE_DIR = Path(__file__).resolve().parent
final_model_lr = joblib.load(str(BASE_DIR / 'linear_regression_model.joblib'))
scaler = joblib.load(str(BASE_DIR / 'scaler.joblib'))
encoded_columns = joblib.load(str(BASE_DIR / 'encoded_columns.joblib'))

# Determine model expected feature names if available
try:
    MODEL_FEATURES = list(final_model_lr.feature_names_in_)
except Exception:
    MODEL_FEATURES = None

app = Flask(__name__)
# allow routes with or without trailing slash
app.url_map.strict_slashes = False

# Define the non-encoded numerical columns that need scaling
numerical_features_for_scaling = ['Store_id', 'Year', 'Month', 'DayOfWeek']

# Define the categorical columns that were one-hot encoded
categorical_features_for_encoding = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']

# Build the full expected training columns for the model
# Prefer the model's saved feature names when available to avoid using stale encoded_columns
if MODEL_FEATURES:
    X_train_columns_full = list(MODEL_FEATURES)
else:
    X_train_columns_full = numerical_features_for_scaling + ['Holiday'] + list(encoded_columns)


# 2. Define the preprocessing function
def preprocess_input(data: dict, scaler, expected_cols):
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([data])

    # Extract date-based features if Date provided
    if 'Date' in input_df.columns:
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        input_df['Year'] = input_df['Date'].dt.year
        input_df['Month'] = input_df['Date'].dt.month
        input_df['DayOfWeek'] = input_df['Date'].dt.dayofweek
        input_df = input_df.drop(columns=['Date'], errors='ignore')

    # One-hot encode categorical columns present in input
    categorical_features = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']
    input_encoded = pd.get_dummies(input_df, columns=[c for c in categorical_features if c in input_df.columns], drop_first=False)

    # Create final frame with exact columns expected by model/training
    expected_cols = list(expected_cols)
    final_df = pd.DataFrame(columns=expected_cols, index=[0]).fillna(0)

    # Fill what we know from the input_encoded (align by column name)
    for col in input_encoded.columns:
        if col in final_df.columns:
            final_df.at[0, col] = input_encoded.at[0, col]

    # Ensure numeric columns are numeric
    numerics = ['Store_id', 'Year', 'Month', 'DayOfWeek']
    for n in numerics:
        if n in final_df.columns:
            final_df[n] = pd.to_numeric(final_df[n], errors='coerce').fillna(0)

    # Scale only the numerical feature columns that scaler expects (if scaler present)
    if scaler is not None:
        cols_to_scale = [c for c in numerics if c in final_df.columns]
        if cols_to_scale:
            try:
                final_df[cols_to_scale] = scaler.transform(final_df[cols_to_scale])
            except Exception:
                # If scaler cannot transform (shape mismatch), leave values as-is; model will likely raise an error later
                pass

    # If any columns are boolean, convert to int (0/1)
    for c in final_df.columns:
        if final_df[c].dtype == 'bool':
            final_df[c] = final_df[c].astype(int)

    # Reorder columns exactly as expected
    final_df = final_df[expected_cols]

    return final_df


# 3. Prediction endpoint for Linear Regression
@app.route('/predictSales', methods=['GET', 'POST'])
@app.route('/predictSales/', methods=['GET', 'POST'])
def predictSales():
    # Minimal GET for health-check
    if request.method == 'GET':
        return jsonify({'status': 'ok'})

    try:
        data = request.get_json(force=True)
        # Preprocess using the full expected columns
        preprocessed = preprocess_input(data, scaler, X_train_columns_full)

        # Debug prints
        print('Received JSON:', data)
        print('Preprocessed columns:', preprocessed.columns.tolist())

        # Determine model-expected columns
        expected_cols = None
        try:
            expected_cols = list(final_model_lr.feature_names_in_)
        except Exception:
            expected_cols = list(X_train_columns_full)

        received_set = set(preprocessed.columns.tolist())
        expected_set = set(expected_cols)
        missing = sorted(list(expected_set - received_set))
        extra = sorted(list(received_set - expected_set))
        if missing or extra:
            return jsonify({
                'error': 'Feature mismatch',
                'missing_features': missing,
                'extra_features': extra,
                'note': 'Model was trained with different features. Retrain model or provide the expected features.'
            }), 400

        # Make prediction
        pred = final_model_lr.predict(preprocessed)
        return jsonify({'predicted_sales': float(pred[0])})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


# root health endpoint listing available endpoints
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'ok',
        'endpoints': {
            'lr_predict': '/predictSales',
            'arima_predict': '/predict_arima'
        },
        'message': 'Visit /predictSales (GET) for usage or POST JSON to that endpoint.'
    })

# To run the Flask app, you would typically save this code as app.py and run `flask run`
# For demonstration purposes, we can include a main block here, but it's not ideal for production.
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    print("Flask app loaded. Starting development server...")
    print(f"Endpoint: http://{host}:{port}/predictSales")
    

    # Print registered routes to help debug 404s when Run-button is used
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f" {rule} -> methods: {methods}")

    app.run(host=host, port=port, debug=True)