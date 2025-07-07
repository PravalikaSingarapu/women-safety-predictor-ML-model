# app.py

from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from scipy.sparse import hstack

app = Flask(__name__)

# Load model, vectorizer, encoders
with open('model/safety_model.pkl', 'rb') as f:
    model, vectorizer, label_encoders, target_le = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Safety Tips
safety_tips = {
    'Safe': "You seem to be in a safe environment. Stay alert and keep your phone charged.",
    'Caution': "Be cautious. Share your location and avoid isolated areas.",
    'Danger': "You're in a potentially risky situation. Call someone you trust or seek help immediately!"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    time_of_day = request.form['time_of_day']
    location_type = request.form['location_type']
    crowd_density = request.form['crowd_density']
    is_alone = request.form['is_alone']
    battery_level = int(request.form['battery_level'])
    mood_text = request.form['mood_text']

    # Encode categorical values
    encoded = [
        label_encoders['time_of_day'].transform([time_of_day])[0],
        label_encoders['location_type'].transform([location_type])[0],
        label_encoders['crowd_density'].transform([crowd_density])[0],
        label_encoders['is_alone'].transform([is_alone])[0],
        battery_level
    ]

    # Clean and vectorize text
    cleaned_text = clean_text(mood_text)
    text_vector = vectorizer.transform([cleaned_text])

    # Combine text vector and numerical features
    combined_input = hstack([text_vector, np.array([encoded])])

    # Predict
    prediction_index = model.predict(combined_input)[0]
    prediction_label = target_le.inverse_transform([prediction_index])[0]
    tip = safety_tips[prediction_label]

    return render_template('result.html', risk=prediction_label, tip=tip)

if __name__ == '__main__':
    app.run(debug=True)

