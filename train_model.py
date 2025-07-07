# train_model.py

import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Dataset
df = pd.read_csv('safety_data.csv')

# Preprocessing categorical features
categorical_cols = ['time_of_day', 'location_type', 'crowd_density', 'is_alone']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['mood_text'] = df['mood_text'].apply(clean_text)

# Combine all features
X_numeric = df[['time_of_day', 'location_type', 'crowd_density', 'is_alone', 'battery_level']]
X_text = df['mood_text']
y = df['risk_level']

# Encode target
target_le = LabelEncoder()
y = target_le.fit_transform(y)  # 0: Caution, 1: Danger, 2: Safe

# Split
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
    X_text, X_numeric, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Vectorize text
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# Combine text and numeric features
from scipy.sparse import hstack

X_train_combined = hstack([X_train_vec, X_train_num])
X_test_combined = hstack([X_test_vec, X_test_num])

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_combined, y_train)

# Evaluate
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# Save model and vectorizer
with open('model/safety_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer, label_encoders, target_le), f)

print("✅ Model trained and saved to model/safety_model.pkl")

