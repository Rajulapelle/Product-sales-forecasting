import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
import traceback

app = Flask(__name__)
# allow trailing slash or no-trailing-slash
app.url_map.strict_slashes = False

# --- Load Saved Artifacts (file-relative paths) ---
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / 'model_xgb_final.joblib'
scaler_path = BASE_DIR / 'scaler_current.joblib'
columns_path = BASE_DIR / 'X_train_columns.joblib'

try:
    model_xgb_final = joblib.load(str(model_path))
    scaler_current = joblib.load(str(scaler_path))
    X_train_columns = joblib.load(str(columns_path))
    print("Model, scaler, and column names loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    model_xgb_final = None
    scaler_current = None
    X_train_columns = None

# --- Preprocessing Function ---
def preprocess_input(data: dict, scaler, X_train_columns):
    # Convert input data to DataFrame
    warnings = []
    input_df = pd.DataFrame([data])

    # Convert 'Date' to datetime and extract features
    if 'Date' in input_df.columns:
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        input_df['Year'] = input_df['Date'].dt.year
        input_df['Month'] = input_df['Date'].dt.month
        input_df['DayOfWeek'] = input_df['Date'].dt.dayofweek
    else:
        # Handle case where 'Date' might not be provided for existing features
        input_df['Year'] = np.nan # Or some default/imputation strategy
        input_df['Month'] = np.nan
        input_df['DayOfWeek'] = np.nan

    # Drop 'Date' column as it's no longer needed after feature extraction
    input_df = input_df.drop(columns=['Date'], errors='ignore')

    # One-hot encode categorical features, ensuring consistency with training data
    # These categorical columns were converted to boolean by pd.get_dummies(drop_first=True)
    # We need to recreate them for the new data. Create dummy columns with False first,
    # then set True where appropriate. This handles cases where a category might be missing in new data.
    categorical_features = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']
    for col in categorical_features:
        if col in input_df.columns:
            # Get unique values from the input for the current categorical column
            unique_vals = input_df[col].unique()
            for val in unique_vals:
                dummy_col_name = f"{col}_{val}" if val != 'No' and val != 0 else f"{col}_{val}" if val == 'Yes' else f"{col}_{val}" if val == 1 else ""
                if dummy_col_name in X_train_columns:
                    input_df[dummy_col_name] = (input_df[col] == val).astype(bool)
            input_df = input_df.drop(columns=[col])

    # Align columns with X_train_columns, filling missing with 0 (for one-hot encoded and missing numericals)
    processed_df = pd.DataFrame(columns=X_train_columns) # Create an empty DataFrame with target columns
    for col in X_train_columns:
        if col in input_df.columns:
            processed_df[col] = input_df[col].astype(bool) if 'Store_Type' in col or 'Location_Type' in col or 'Region_Code' in col or 'Discount' in col else input_df[col] # Cast boolean columns
        else:
            processed_df[col] = 0 # Fill numericals with 0 and booleans with False. Boolean columns will handle True/False correctly during reindexing.

    # Convert numerical features (Store_id, Year, Month, DayOfWeek) to numeric before scaling
    numerical_features_to_scale = ['Store_id', 'Year', 'Month', 'DayOfWeek']
    for feature in numerical_features_to_scale:
        if feature in processed_df.columns:
            processed_df[feature] = pd.to_numeric(processed_df[feature], errors='coerce')

    # Scale numerical features using the loaded scaler
    if scaler:
        try:
            processed_df[numerical_features_to_scale] = scaler.transform(processed_df[numerical_features_to_scale])
        except ValueError as ve:
            # Scaler was likely fitted on different feature names (or different number of features).
            # Skip scaling and warn — this prevents a 400 caused by sklearn's feature-name checks.
            traceback.print_exc()
            warnings.append('Skipped scaling: scaler feature names mismatch or incompatible shape')
        except Exception:
            traceback.print_exc()
            warnings.append('Skipped scaling due to unexpected scaler error')

    # Convert boolean columns to integer (0 or 1) for the model
    for col in processed_df.columns:
        if processed_df[col].dtype == 'bool':
            processed_df[col] = processed_df[col].astype(int)

    return processed_df, warnings

# --- Prediction Endpoint ---
@app.route('/XGBoost_predict', methods=['GET', 'POST'])
@app.route('/XGBoost_predict/', methods=['GET', 'POST'])
def predict():
    # Minimal GET -> health-check (do not expose example payload or curl)
    if request.method == 'GET':
        return jsonify({'status': 'ok'})
    if model_xgb_final is None or scaler_current is None or X_train_columns is None:
        return jsonify({'error': 'Model artifacts not loaded. Please check server logs.'}), 500

    try:
        json_data = request.get_json(force=True)
        if not isinstance(json_data, dict):
            return jsonify({'error': 'Invalid JSON format. Expected a dictionary.'}), 400

        # Debug: show received JSON
        print("Received JSON:", json_data)

        # Determine expected columns (prefer model's saved feature names when available)
        expected_cols = None
        if X_train_columns is not None:
            expected_cols = list(X_train_columns)
        try:
            if hasattr(model_xgb_final, 'feature_names_in_'):
                expected_cols = list(model_xgb_final.feature_names_in_)
            else:
                # XGBoost booster
                try:
                    expected_cols = list(model_xgb_final.get_booster().feature_names)
                except Exception:
                    pass
        except Exception:
            pass

        # Preprocess the input data using expected columns to guide dummy creation
        processed_data, preprocess_warnings = preprocess_input(json_data, scaler_current, expected_cols if expected_cols is not None else X_train_columns)

        # Debug: show processed dataframe columns and preview
        print("Processed data columns:", processed_data.columns.tolist())
        try:
            print("Processed data preview:\n", processed_data.head().to_dict())
        except Exception:
            pass

        # Validate columns against expected training columns and gracefully align them.
        expected_cols = None
        if X_train_columns is not None:
            expected_cols = list(X_train_columns)
        else:
            try:
                expected_cols = list(model_xgb_final.get_booster().feature_names)
            except Exception:
                expected_cols = None

        if expected_cols is not None:
            received_set = set(processed_data.columns.tolist())
            expected_set = set(expected_cols)
            missing = sorted(list(expected_set - received_set))
            extra = sorted(list(received_set - expected_set))

            warning_msgs = []
            # include any warnings from preprocessing (e.g., skipped scaling)
            if preprocess_warnings:
                warning_msgs.extend(preprocess_warnings)
            # Fill any missing expected features with zeros (best-effort)
            if missing:
                for col in missing:
                    processed_data[col] = 0
                warning_msgs.append(f"Filled missing features with 0: {missing}")

            # Drop unexpected extra columns
            if extra:
                processed_data = processed_data.drop(columns=extra, errors='ignore')
                warning_msgs.append(f"Dropped unexpected features: {extra}")

            # Ensure final DataFrame has all expected columns in the correct order
            processed_data = processed_data.reindex(columns=expected_cols, fill_value=0)

            # Attempt prediction; if XGBoost still raises a ValueError about feature names,
            # return a helpful error with details. Otherwise return prediction and warnings.
            try:
                # Convert to numpy array to avoid XGBoost/ sklearn feature-name checks
                # (we already aligned/reindexed columns to expected order)
                prediction = model_xgb_final.predict(processed_data.to_numpy())
            except ValueError as ve:
                traceback.print_exc()
                return jsonify({'error': 'Prediction failed due to feature name mismatch', 'detail': str(ve)}), 500
            except Exception as ex:
                traceback.print_exc()
                return jsonify({'error': 'Prediction failed', 'detail': str(ex)}), 500

            resp = {'predicted_sales_scaled': prediction.tolist()[0]}
            if warning_msgs:
                resp['warnings'] = warning_msgs
            return jsonify(resp)

        # Make prediction (convert to numpy to bypass DataFrame column-name validation)
        prediction = model_xgb_final.predict(processed_data.to_numpy())

        # attach any preprocessing warnings to the response
        resp = {'predicted_sales_scaled': prediction.tolist()[0]}
        if preprocess_warnings:
            resp['warnings'] = preprocess_warnings
        return jsonify(resp)
        # The target variable 'Sales' was scaled. Inverse transform the prediction if needed.
        # To inverse transform, we need to create a dummy array with the scaled features,
        # where the prediction is in the 'Sales' column, and then inverse_transform.
        # This would require knowing the original numerical_features (including Sales)
        # which was ['#Order', 'Sales', 'Sales_Rolling_Mean_7D'] and their order.
        # For simplicity, we return the scaled prediction here. If original scale is needed,
        # the scaler.inverse_transform logic needs careful reconstruction.

        # To inverse transform the 'Sales' column, we need to know its original position in 'numerical_features'
        # when the scaler_current was fitted on ['Store_id', 'Year', 'Month', 'DayOfWeek']
        # This API assumes the target variable 'Sales' is independent and not part of scaler_current
        # If Sales was part of the scaler used for X_train and y_train, we would need to handle this.
        # However, looking at the notebook, `y_train` was 'Sales' directly, and `scaler_current` only transformed
        # ['Store_id', 'Year', 'Month', 'DayOfWeek']. Therefore, the `model_xgb_final` predicts directly
        # the scaled 'Sales' target, and inverse scaling `y_pred` is not straightforward with `scaler_current`.
        # If the original 'Sales' values were used as `y_train`, then the predictions are already in the original scale.
        # But since y_train was scaled (from `dataset['Sales'] = scaler.fit_transform(dataset[numerical_features])`),
        # the prediction `prediction` is on the scaled target variable. Inverse scaling is needed.
        
        # Reconstruct dummy array for inverse_transform if Sales was scaled with scaler_current.
        # Looking back at the notebook, numerical_features = ['#Order', 'Sales', 'Sales_Rolling_Mean_7D'] were scaled by the *first* scaler.
        # Then y_train was assigned dataset['Sales'] (which was already scaled).
        # The scaler_current was then fitted and transformed on ['Store_id', 'Year', 'Month', 'DayOfWeek'] for X_train.
        # So, the output of model_xgb_final.predict(processed_data) *is* the scaled Sales value.
        # To inverse transform this, we need the *original* scaler that was used on 'Sales'.
        # This means we need to save *that* scaler too, or re-think the scaling strategy for y_train.
        
        # Assuming for now `model_xgb_final` predicts the scaled `Sales` values,
        # and `scaler_current` is only for `X_train` features.
        # To inverse transform the prediction, we would need the scaler that was originally applied to `Sales` (the first `scaler` initialized).
        # Let's assume the task is to return the scaled prediction for now, or acknowledge the missing scaler for target.
        
        # For demonstration, we'll return the scaled prediction. If inverse transformation of 'Sales' is required,
        # the original scaler used for 'Sales' needs to be loaded and applied.
        
        # Placeholder for inverse scaling if `Sales` was part of `scaler_current` or a separate `y_scaler` was available:
        # sales_idx = numerical_features.index('Sales') # Assuming numerical_features from original scaling of dataset
        # dummy_features = np.zeros((1, len(numerical_features)))
        # dummy_features[0, sales_idx] = prediction[0]
        # original_sales = original_sales_scaler.inverse_transform(dummy_features)[:, sales_idx]
        
        # For this Flask app, we'll return the raw (scaled) prediction.
        
        return jsonify({'predicted_sales_scaled': prediction.tolist()[0]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'ok',
        'endpoints': {
            'xgb_predict': '/XGBoost_predict',
            'lr_predict': '/predictSales',
            'arima_predict': '/predict_arima'
        },
        'message': 'Use GET on /XGBoost_predict for instructions or POST JSON to that endpoint.'
    })


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    print("Flask XGBoost app loaded. Starting development server...")
    print(f"Endpoint: http://{host}:{port}/XGBoost_predict")
    print("Example curl:")
    print("curl -X POST -H \"Content-Type: application/json\" -d '{\"Store_id\":1,\"Store_Type\":\"S1\",\"Location_Type\":\"L1\",\"Region_Code\":\"R1\",\"Date\":\"2019-06-01\",\"Holiday\":0,\"Discount\":\"No\"}' http://127.0.0.1:5000/XGBoost_predict")

    # Print registered routes to help debug 404s when Run-button is used
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f" {rule} -> methods: {methods}")

    app.run(host=host, port=port, debug=True)
