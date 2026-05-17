# Fake News Detector API

A machine learning-based API to detect fake news using TF-IDF and Logistic Regression.

## Model Performance

- Accuracy: 87.50%+
- Precision: 85%+
- Recall: 85%+
- F1-Score: 85%+

## Features

Detects fake news and misinformation
Provides confidence scores
Fast predictions
RESTful API
CORS enabled
Production ready

## Architecture

- Model: Logistic Regression
- Vectorizer: TF-IDF
- Backend: Flask
- Deployment: Railway.app

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python app.py
```

## API Endpoints

### 1. Home Route
GET /

### 2. Health Check
GET /health

### 3. Prediction
POST /predict

Input JSON:
```json
{
    "text": "Your news text here"
}
```

Response:
```json
{
    "status": "success",
    "prediction": 1,
    "label": "FAKE NEWS",
    "confidence": 0.8842,
    "confidence_percentage": 88.42
}
```

## Testing the API

### Using cURL
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "Your text"}'
```

### Using Python
```python
import requests
url = 'http://localhost:5000/predict'
response = requests.post(url, json={'text': 'Your text'})
print(response.json())
```

## Deployment on Railway.app

1. Create GitHub repository
2. Push code to GitHub
3. Go to railway.app
4. Deploy from GitHub repo
5. Get your API URL

## Input Validation

- Minimum text length: 10 characters
- Maximum text length: 5000 characters
- Required format: JSON with "text" field

## Prediction Classes

- 0 (REAL NEWS): Legitimate news
- 1 (FAKE NEWS): Misinformation or hoax

## License

MIT License
