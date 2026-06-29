from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
from utils.prediction_utils import predict_points, validate_input

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model and preprocessing info
MODEL_PATH = 'models/gradient_boosting_eurovision_model.pkl'
PREPROCESSING_PATH = 'models/preprocessing_info.pkl'

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(PREPROCESSING_PATH, 'rb') as f:
    preprocessing_info = pickle.load(f)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            # Get JSON data
            data = request.get_json()
            
            # Validate input
            validation_result = validate_input(data)
            if not validation_result['valid']:
                return jsonify({
                    'success': False,
                    'error': validation_result['message']
                }), 400
            
            # Prepare features
            features = np.array([[
                data['danceability'],
                data['energy'],
                data['valence'],
                data['tempo'],
                data['acousticness']
            ]])
            
            # Make prediction
            predicted_points = model.predict(features)[0]
            
            # Calculate confidence interval (using MAE ±24.5)
            mae = preprocessing_info['mae_test']
            lower_bound = max(0, predicted_points - mae)
            upper_bound = predicted_points + mae
            
            return jsonify({
                'success': True,
                'predicted_points': round(predicted_points, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'mae': round(mae, 2)
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return render_template('predict.html')

@app.route('/analysis')
def analysis():
    """Analysis and insights page"""
    return render_template('analysis.html', 
                         mae=preprocessing_info['mae_test'],
                         r2=preprocessing_info['r2_score'])

@app.route('/api/feature-info')
def feature_info():
    """Get feature information"""
    features_info = {
        'danceability': {
            'description': 'Describes how suitable a track is for dancing',
            'range': [0, 1],
            'unit': 'score'
        },
        'energy': {
            'description': 'Intensity and activity measure',
            'range': [0, 1],
            'unit': 'score'
        },
        'valence': {
            'description': 'Musical positiveness conveyed by the track',
            'range': [0, 1],
            'unit': 'score'
        },
        'tempo': {
            'description': 'Overall estimated tempo of the track',
            'range': [0, 300],
            'unit': 'BPM'
        },
        'acousticness': {
            'description': 'Confidence measure of whether track is acoustic',
            'range': [0, 1],
            'unit': 'score'
        }
    }
    return jsonify(features_info)

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)