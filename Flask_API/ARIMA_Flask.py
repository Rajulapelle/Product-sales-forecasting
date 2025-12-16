from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the pre-trained ARIMA model (file-relative path)
BASE_DIR = Path(__file__).resolve().parent
model_arima_fit = joblib.load(str(BASE_DIR / 'arima_model.joblib'))

# NOTE: For ARIMA, the historical data (train_data) is crucial for determining
# the forecast start/end points. We need to recreate or load this.
# Assuming 'daily_sales' and 'train_data' from the notebook context are available
# or can be recreated from the original dataset.
# For a standalone Flask app, you'd typically save this alongside the model.
# Let's assume for this example, we re-create it minimally.
# In a real scenario, you would save and load the train_data's length or last_date.

# Re-create daily_sales and train_data context for determining forecast indices
# In a production setup, you would load these pre-computed values or indices.
# This is a simplification based on the previous notebook cells.
# Here, we will just use the length of the train_data from the notebook state
# (which was 412 entries up to 2019-02-16).
# The ARIMA model's predictions are based on its internal state, but for 'predict'
# with start/end indices, those indices relate to the original series length.
# The length of `train_data` was 412.
# The length of `validation_data` was 104.

# We'll need the total length of the series used to train the model to correctly
# calculate the `start` index for `predict` in the Flask app.
# In the notebook, `train_data` had 412 entries.
ARIMA_TRAIN_DATA_LENGTH = 412 # This should ideally be loaded from a saved config or determined dynamically
ARIMA_LAST_TRAIN_DATE = pd.to_datetime('2019-02-16') # This should ideally be loaded

# 2. Define the preprocessing function for ARIMA
def preprocess_input_for_arima(data):
    # Extract the 'Date' from the input data
    input_date_str = data.get('Date')
    if not input_date_str:
        raise ValueError("'Date' field is missing in the input.")

    input_date = pd.to_datetime(input_date_str)

    # Calculate the number of days between the last training date and the input date
    # This determines how many steps into the future we need to forecast.
    forecast_horizon = (input_date - ARIMA_LAST_TRAIN_DATE).days

    if forecast_horizon <= 0:
        raise ValueError("Forecast date must be after the last training date (2019-02-16).")

    # The 'start' index for prediction is the length of the training data
    start_index = ARIMA_TRAIN_DATA_LENGTH
    # The 'end' index is start_index + forecast_horizon - 1
    end_index = start_index + forecast_horizon - 1

    return start_index, end_index

# 3. Initialize the Flask application
app = Flask(__name__)
# allow routes to be requested with or without trailing slashes
app.url_map.strict_slashes = False

# 4. Create a prediction endpoint for ARIMA
@app.route('/predict_arima', methods=['GET', 'POST'])
def predict_arima():
    # If called with GET, return a health-check / usage message so browsers and Run buttons don't get 404/405
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Send a POST with JSON {"Date": "YYYY-MM-DD"} to get a forecast',
            'endpoint': '/predict_arima',
            'example_curl': "curl -X POST -H \"Content-Type: application/json\" -d '{\"Date\": \"2019-06-01\"}' http://127.0.0.1:5000/predict_arima"
        })

    try:
        data = request.get_json(force=True)
        start_index, end_index = preprocess_input_for_arima(data)

        # Generate ARIMA predictions
        arima_prediction_series = model_arima_fit.predict(start=start_index, end=end_index, dynamic=False)

        # The last value corresponds to the requested input_date.
        predicted_sales = arima_prediction_series.iloc[-1].item()

        return jsonify({'predicted_sales_arima': predicted_sales})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# root health endpoint listing available endpoints
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'ok',
        'endpoints': {
            'arima_predict': '/predict_arima',
            'lr_predict': '/predictSales'
        },
        'message': 'Use GET on /predict_arima for instructions or POST JSON to that endpoint.'
    })

# To run the Flask app, you would typically save this code as app_arima.py and run `flask run`
# For demonstration purposes, we can include a main block here.
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    print("Flask ARIMA app loaded. Starting development server...")
    print(f"Endpoint: http://{host}:{port}/predict_arima")
    print("Example curl:")
    print("curl -X POST -H \"Content-Type: application/json\" -d '{\"Date\": \"2019-06-01\"}' http://127.0.0.1:5000/predict_arima")
    # Print registered routes to help debug 404s when Run-button is used
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f" {rule} -> methods: {methods}")

    app.run(host=host, port=port, debug=True)