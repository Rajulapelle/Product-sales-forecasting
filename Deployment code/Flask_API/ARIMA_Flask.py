from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the pre-trained ARIMA model (file-relative path)
BASE_DIR = Path(__file__).resolve().parent
model_arima_fit = None
try:
    model_arima_fit = joblib.load(str(BASE_DIR / 'arima_model.joblib'))
except Exception as e:
    print(f"Warning: could not load ARIMA model: {e}")



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

    # We only need the forecast horizon (number of steps ahead)
    return int(forecast_horizon)

# 3. Initialize the Flask application
app = Flask(__name__)

# 4. Create a prediction endpoint for ARIMA (support GET for health-check + usage)
@app.route('/predict_arima', methods=['GET', 'POST'])
def predict_arima():
    # GET -> simple health-check and usage example
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'usage': 'POST JSON {"Date":"YYYY-MM-DD"} to this endpoint to get forecast',
            'example': '/predict_arima with POST body {"Date":"2019-06-01"}'
        })

    try:
        data = request.get_json(force=True)
        forecast_horizon = preprocess_input_for_arima(data)
        if forecast_horizon <= 0:
            return jsonify({'error': 'Forecast horizon must be positive.'}), 400
        if model_arima_fit is None:
            return jsonify({'error': 'ARIMA model not loaded on server.'}), 500

        # Use get_forecast to produce out-of-sample forecasts for the horizon
        try:
            forecast = model_arima_fit.get_forecast(steps=forecast_horizon)
            # predicted_mean is a pandas Series; the last element corresponds to the requested date
            predicted_sales = float(forecast.predicted_mean.iloc[-1])
            return jsonify({'predicted_sales_arima': predicted_sales})
        except AttributeError:
            # Some saved models may not implement get_forecast; fall back to predict with indices
            try:
                start_index = ARIMA_TRAIN_DATA_LENGTH
                end_index = start_index + forecast_horizon - 1
                arima_prediction_series = model_arima_fit.predict(start=start_index, end=end_index, dynamic=False)
                predicted_sales = float(arima_prediction_series.iloc[-1])
                return jsonify({'predicted_sales_arima': predicted_sales})
            except Exception as ex:
                return jsonify({'error': f'ARIMA fallback prediction failed: {ex}'}), 500
        except Exception as ex:
            return jsonify({'error': f'ARIMA forecast failed: {ex}'}), 500
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# To run the Flask app, you would typically save this code as app_arima.py and run `flask run`
# For demonstration purposes, we can include a main block here.
if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5001
    print("Flask ARIMA app loaded. Starting development server...")
    print(f"Endpoint: http://{host}:{port}/predict_arima")
    print("Example curl:")
    print("curl -X POST -H \"Content-Type: application/json\" -d '{\"Date\": \"2019-06-01\"}' http://127.0.0.1:5001/predict_arima")
    # Print registered routes to help debug method/404 issues
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        methods = ','.join(sorted(rule.methods - {"HEAD", "OPTIONS"}))
        print(f" {rule} -> methods: {methods}")

    app.run(host=host, port=port, debug=True)