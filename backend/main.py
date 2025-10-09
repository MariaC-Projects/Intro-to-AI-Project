# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Load data -----------------
HERE = Path(__file__).parent                      # .../backend
DATA_FILE = HERE / "recipes.csv"                  # keep it simple

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Could not find {DATA_FILE}. Put recipes.csv in the backend folder.")

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig").dropna(subset=["Ingredients"])
df["Ingredients"] = df["Ingredients"].astype(str).str.lower()

# If your CSV has a different title column name, adapt here:
title_col = "Recipe_name" if "Recipe_name" in df.columns else ("Title" if "Title" in df.columns else None)
if title_col is None:
    # create a fallback title
    df["Title"] = df.index.to_series().apply(lambda i: f"Recipe #{i}")
    title_col = "Title"

# ----------------- Fit TF-IDF once -----------------
vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w\-]+\b")
X_recipes = vec.fit_transform(df["Ingredients"])

# ----------------- FastAPI app -----------------
app = FastAPI(title="AI-Powered Grocery & Recipe Recommender")

# (optional) allow your React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PantryInput(BaseModel):
    pantry: str
    top_k: int = 5

@app.get("/")
def root():
    return {"ok": True, "message": "Recommender API running"}

@app.post("/recommend")
def recommend(body: PantryInput):
    pantry_text = (body.pantry or "").lower()
    q = vec.transform([pantry_text])
    sims = cosine_similarity(q, X_recipes).ravel()

    k = max(1, min(body.top_k, len(df)))
    top_idx = sims.argsort()[::-1][:k]

    results = []
    for i in top_idx:
        results.append({
            "recipe_name": df.iloc[i][title_col],
            "ingredients": df.iloc[i]["Ingredients"],
            "similarity": float(sims[i])
        })
    return {"results": results}
