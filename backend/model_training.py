# backend/model_training.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
df = pd.read_csv("Food Ingredients and Recipe Dataset with Images.csv")
df.dropna(subset=["Ingredients"], inplace=True)
df['Ingredients'] = df['Ingredients'].astype(str).str.lower()

# 2. Fake labels for demo (you can replace with real data later)
import numpy as np
df['label'] = np.random.randint(0, 2, len(df))

# 3. Train TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['Ingredients'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Save model + vectorizer
joblib.dump(model, "recipe_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
print("✅ Model and vectorizer saved.")
