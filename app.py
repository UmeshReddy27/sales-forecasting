from flask import Flask, render_template, request, jsonify, make_response
import pandas as pd
import joblib
import traceback
from sklearn.metrics import mean_absolute_error
import numpy as np

# Create a Flask web application instance
app = Flask(__name__)

# Load the pre-trained LightGBM model
try:
    model = joblib.load('lgbm_forecaster.pkl')
except FileNotFoundError:
    model = None
    print("ERROR: Model file 'lgbm_forecaster.pkl' not found.")
    print("Please make sure you have trained and saved the model from your notebook.")

# Define the features the model was trained on
MODEL_FEATURES = ['store', 'item', 'month', 'year', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter']

# --- UPGRADE: Simple in-memory cache for the last forecast ---
forecast_cache = {}

def feature_engineer(df):
    """Creates time-series features from a datetime index."""
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    return df

@app.route('/')
def home():
    """Serves the main HTML page of the application."""
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    """Handles file upload, runs forecasting, and returns results."""
    if model is None:
        return jsonify({'error': 'Model is not loaded. Check server logs.'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        
        forecast_days = request.form.get('forecast_days', default=90, type=int)
        store_filter = request.form.get('store_filter', 'all')
        item_filter = request.form.get('item_filter', 'all')

        required_columns = ['date', 'store', 'item', 'sales']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': f'Invalid CSV format. Missing: {required_columns}'}), 400

        df = feature_engineer(df.copy())

        filtered_df = df.copy()
        if store_filter != 'all':
            filtered_df = filtered_df[filtered_df['store'] == int(store_filter)]
        if item_filter != 'all':
            filtered_df = filtered_df[filtered_df['item'] == int(item_filter)]
        
        if filtered_df.empty:
            return jsonify({'error': 'No data for the selected store/item filter.'}), 400
        
        mae = None
        if len(filtered_df) > forecast_days:
            test_set = filtered_df.tail(forecast_days)
            test_predictions = model.predict(test_set[MODEL_FEATURES])
            mae = mean_absolute_error(test_set['sales'], test_predictions)

        historical_plot_df = filtered_df.groupby('date')['sales'].sum().reset_index()

        last_date = filtered_df['date'].max()
        future_dates = pd.date_range(start=last_date, periods=forecast_days + 1, freq='D')[1:]
        
        if store_filter != 'all' and item_filter != 'all':
            future_combos = pd.DataFrame([{'store': int(store_filter), 'item': int(item_filter)}])
        else:
            future_combos = filtered_df[filtered_df['date'] == last_date][['store', 'item']].drop_duplicates()

        future_df = pd.concat([future_combos.assign(date=date) for date in future_dates], ignore_index=True)
        future_df = feature_engineer(future_df)
        
        future_predictions = model.predict(future_df[MODEL_FEATURES])
        future_df['predicted_sales'] = np.maximum(0, future_predictions)
        
        daily_forecast = future_df.groupby('date')['predicted_sales'].sum().reset_index()
        daily_forecast.rename(columns={'predicted_sales': 'sales'}, inplace=True)
        
        # --- UPGRADE: Store the forecast data in the cache for download ---
        forecast_cache['last_forecast'] = daily_forecast[['date', 'sales']].copy()

        importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': MODEL_FEATURES, 'importance': importances}).sort_values('importance', ascending=False)

        response_data = {
            'labels': historical_plot_df['date'].dt.strftime('%Y-%m-%d').tolist() + daily_forecast['date'].dt.strftime('%Y-%m-%d').tolist(),
            'historical_sales': historical_plot_df['sales'].tolist(),
            'forecast': ([None] * len(historical_plot_df)) + daily_forecast['sales'].tolist(),
            'mae': f'{mae:.2f}' if mae is not None else 'N/A',
            'feature_importance': {'labels': importance_df['feature'].tolist(), 'scores': importance_df['importance'].tolist()}
        }
        
        return jsonify(response_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during processing.'}), 500

# --- UPGRADE: New endpoint to handle CSV download ---
@app.route('/download_forecast')
def download_forecast():
    """Serves the last generated forecast data as a downloadable CSV file."""
    if 'last_forecast' in forecast_cache:
        df = forecast_cache['last_forecast']
        
        # Format date for readability in CSV
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df['sales'] = df['sales'].round(2)

        csv_data = df.to_csv(index=False)
        response = make_response(csv_data)
        response.headers["Content-Disposition"] = "attachment; filename=forecast_data.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    else:
        return "No forecast data available to download.", 404

if __name__ == '__main__':
    app.run(debug=True)

