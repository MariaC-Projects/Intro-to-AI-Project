# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# ----- Load trained artifacts -----
# Make sure you have these from model_training.py
tfidf = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("recipe_model.joblib")

# Load your dataset (so we can show recipe names)
df = pd.read_csv("Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
df = df[['Recipe_name', 'Ingredients']].dropna()
df['Ingredients'] = df['Ingredients'].astype(str).str.lower()

# ----- FastAPI setup -----
app = FastAPI(title="AI-Powered Grocery & Recipe Recommender")

# Request body schema
class PantryInput(BaseModel):
    pantry: str
    top_k: int = 5  # how many recipes to return

# ----- Helper: recommend recipes -----
@app.post("/recommend")
def recommend_recipes(request: PantryInput):
    # 1. Transform the pantry text using the same TF-IDF vectorizer
    pantry_text = request.pantry.lower()
    pantry_vec = tfidf.transform([pantry_text])

    # 2. Compute cosine similarity between pantry and all recipes
    recipe_vecs = tfidf.transform(df["Ingredients"])
    similarities = cosine_similarity(pantry_vec, recipe_vecs).ravel()

    # 3. Get top-k most similar recipes
    top_k = min(request.top_k, len(df))
    top_indices = similarities.argsort()[::-1][:top_k]
    results = []
    for i in top_indices:
        results.append({
            "recipe_name": df.iloc[i]["Recipe_name"],
            "ingredients": df.iloc[i]["Ingredients"],
            "similarity": float(similarities[i])
        })

    return {"results": results}

# ----- Root route -----
@app.get("/")
def home():
    return {"message": "AI-Powered Grocery & Recipe Recommender API is running!"}
