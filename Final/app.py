import os


from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('eurovision_xgb_model.joblib')
model_features = joblib.load('feature_columns.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        input_df = pd.DataFrame([data])
        
    
        input_encoded = pd.get_dummies(input_df, columns=['Year', 'Is.Final', 'Song.In.English'])
        
        for col in model_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[model_features]
        
        prediction = model.predict(input_encoded)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400
    

@app.route('/historical')
def historical_page():
    return render_template('historical.html')

@app.route('/api/countries')
def get_countries():
    df_hist = pd.read_csv('top_providers.csv')
    countries = sorted(df_hist['To Country'].unique().tolist())
    return jsonify(countries)

@app.route('/api/top-provider')
def get_top_provider():
    target_country = request.args.get('country')
    df_hist = pd.read_csv('top_providers.csv')
    row = df_hist[df_hist['To Country'] == target_country]
    if not row.empty:
        return jsonify({
            'provider': row.iloc[0]['From Country'],
            'points': int(row.iloc[0]['Points'])
        })
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)


