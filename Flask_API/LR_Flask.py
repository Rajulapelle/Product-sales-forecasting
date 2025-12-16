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

# Define the non-encoded numerical columns that need scaling
numerical_features_for_scaling = ['Store_id', 'Year', 'Month', 'DayOfWeek']

# Define the categorical columns that were one-hot encoded
categorical_features_for_encoding = ['Store_Type', 'Location_Type', 'Region_Code', 'Discount']

# 2. Define the preprocessing function
def preprocess_input(data):
    # Convert the input dictionary to a pandas DataFrame
    input_df = pd.DataFrame([data])

    # Convert 'Date' column to datetime objects
    input_df['Date'] = pd.to_datetime(input_df['Date'])

    # Extract 'Year', 'Month', and 'DayOfWeek' features
    input_df['Year'] = input_df['Date'].dt.year
    input_df['Month'] = input_df['Date'].dt.month
    input_df['DayOfWeek'] = input_df['Date'].dt.dayofweek

    # Drop the original 'Date' column as it's no longer needed
    input_df = input_df.drop('Date', axis=1)

    # Apply one-hot encoding to categorical features
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_features_for_encoding, drop_first=True)

    # Align columns with the training data's encoded columns
    # First, identify all columns that should be present in the final DataFrame
    expected_columns = numerical_features_for_scaling + ['Holiday'] + encoded_columns
    
    # Reindex the input_df_encoded to match the expected columns. Fill missing with 0 and remove extra.
    input_df_final = input_df_encoded.reindex(columns=expected_columns, fill_value=0)

    # Scale the numerical features
    input_df_final[numerical_features_for_scaling] = scaler.transform(input_df_final[numerical_features_for_scaling])

    return input_df_final

# 3. Initialize the Flask application
app = Flask(__name__)
# allow routes to be requested with or without trailing slashes
app.url_map.strict_slashes = False

# 4. Create a prediction endpoint
@app.route('/predictSales', methods=['GET', 'POST'])
@app.route('/predictSales/', methods=['GET', 'POST'])
def predictSales():
    # GET -> health-check and instructions
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Send a POST with JSON payload matching training features to get a forecast',
            'endpoint': '/predictSales',
            'example_curl': "curl -X POST -H \"Content-Type: application/json\" -d '{\"Store_id\":1,\"Store_Type\":\"S1\",\"Location_Type\":\"L1\",\"Region_Code\":\"R1\",\"Date\":\"2019-06-01\",\"Holiday\":0,\"Discount\":\"No\"}' http://127.0.0.1:5000/predictSales"
        })

    try:
        data = request.get_json(force=True)
        preprocessed_data = preprocess_input(data)
        prediction = final_model_lr.predict(preprocessed_data)
        return jsonify({'predicted_sales': prediction[0].item()})
    except Exception as e:
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
    print("Example curl:")
    print("curl -X POST -H \"Content-Type: application/json\" -d '{\"Store_id\": 1, \"Store_Type\": \"S1\", \"Location_Type\": \"L1\", \"Region_Code\": \"R1\", \"Date\": \"2019-06-01\", \"Holiday\": 0, \"Discount\": \"No\"}' http://127.0.0.1:5000/predictSales")
    app.run(host=host, port=port, debug=True)