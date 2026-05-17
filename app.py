
import os
import joblib
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and vectorizer
MODEL_PATH = 'fake_news_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

# ============================================
# HOME ROUTE
# ============================================
@app.route('/', methods=['GET'])
def home():
    """Home route with API information"""
    return jsonify({
        'status': 'success',
        'message': 'Fake News Detector API is running!',
        'version': '1.0',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'home': '/ (GET)'
        }
    })

# ============================================
# HEALTH CHECK ROUTE
# ============================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None and vectorizer is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'service': 'Fake News Detector API'
    })

# ============================================
# PREDICTION ROUTE (MAIN ENDPOINT)
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if news is fake or real
    
    Expected JSON input:
    {
        "text": "news article text here"
    }
    
    Returns:
    {
        "prediction": 0 (real) or 1 (fake),
        "label": "REAL NEWS" or "FAKE NEWS",
        "confidence": 0.95,
        "status": "success"
    }
    """
    
    try:
        # 1️⃣ Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded',
                'prediction': None
            }), 500
        
        # 2️⃣ Get JSON data from request
        data = request.get_json()
        
        # 3️⃣ Validate input
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Please provide "text" field in JSON body',
                'example': {'text': 'Your news text here'}
            }), 400
        
        text = data['text'].strip()
        
        # 4️⃣ Validate text length
        if len(text) < 10:
            return jsonify({
                'status': 'error',
                'message': 'Text too short. Minimum 10 characters required'
            }), 400
        
        if len(text) > 5000:
            return jsonify({
                'status': 'error',
                'message': 'Text too long. Maximum 5000 characters allowed'
            }), 400
        
        # 5️⃣ Vectorize the text
        text_vectorized = vectorizer.transform([text])
        
        # 6️⃣ Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = float(probabilities[prediction])
        
        # 7️⃣ Map prediction to label
        label = "FAKE NEWS ❌" if prediction == 1 else "REAL NEWS ✅"
        
        # 8️⃣ Return response
        return jsonify({
            'status': 'success',
            'text_length': len(text),
            'prediction': int(prediction),
            'label': label,
            'confidence': round(confidence, 4),
            'confidence_percentage': round(confidence * 100, 2),
            'probabilities': {
                'real_news': round(float(probabilities[0]), 4),
                'fake_news': round(float(probabilities[1]), 4)
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction error: {str(e)}',
            'prediction': None
        }), 500

# ============================================
# ERROR HANDLERS
# ============================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'available_endpoints': ['/predict', '/health', '/']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ============================================
# RUN THE APP
# ============================================
if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )
