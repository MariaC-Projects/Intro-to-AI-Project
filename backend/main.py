# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Paths ----------
HERE = Path(__file__).parent
DATA_FILE = HERE / "recipes.csv"             # keep the name simple: recipes.csv
IMAGES_DIR = HERE / "Food Images"            # folder with your JPGs/PNGs

# ---------- Load data ----------
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Could not find {DATA_FILE}. Place recipes.csv in backend/")

df = pd.read_csv(DATA_FILE, encoding="utf-8-sig").dropna(subset=["Ingredients"])
df["Ingredients"] = df["Ingredients"].astype(str).str.lower()

# pick a title column or create one
title_col = (
    "Recipe_name" if "Recipe_name" in df.columns else
    ("Title" if "Title" in df.columns else None)
)
if title_col is None:
    df["Title"] = df.index.to_series().apply(lambda i: f"Recipe {i}")
    title_col = "Title"

# ---------- TF-IDF once ----------
vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w\-]+\b")
X_recipes = vec.fit_transform(df["Ingredients"])

# ---------- Image matching helpers ----------
def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")

df["title_slug"] = df[title_col].astype(str).apply(slugify)

# build lookup from filename stem -> relative path inside IMAGES_DIR
image_lookup = {}
if IMAGES_DIR.exists():
    # search recursively to handle nested folders (some datasets have an extra 'Food Images' subfolder)
    for p in IMAGES_DIR.rglob("*"):
        if p.is_file():
            # store the path relative to the images directory so we can mount IMAGES_DIR at /images
            rel_path = p.relative_to(IMAGES_DIR).as_posix()
            # use the filename stem as the lookup key (lowercase) -> relative path as value
            image_lookup[p.stem.lower()] = rel_path

def find_image_for_title(slug: str):
    # exact match first
    if slug in image_lookup:
        return image_lookup[slug]
    # loose contains match (look in keys which are filename stems)
    for key in image_lookup:
        if slug in key or key in slug:
            return image_lookup[key]
    return None

df["image_file"] = df["title_slug"].apply(find_image_for_title)

# ---------- FastAPI app ----------
app = FastAPI(title="AI-Powered Grocery & Recipe Recommender")

# CORS (allow your React dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve images at /images/<filename>
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

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
        row = df.iloc[i]
        img_file = row.get("image_file")
        image_url = f"/images/{img_file}" if img_file else None
        results.append({
            "recipe_name": row[title_col],
            "ingredients": row["Ingredients"],
            "similarity": float(sims[i]),
            "image_url": image_url
        })
    return {"results": results}
