# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import re
import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Paths ----------
HERE = Path(__file__).parent
DATA_FILE = HERE / "recipes.csv"             
IMAGES_DIR = HERE / "Food Images"           

VEC_PATH = HERE / "tfidf_vectorizer.joblib"
MATRIX_PATH = HERE / "recipe_tfidf.npz"
META_PARQUET_PATH = HERE / "recipes_metadata.parquet"
META_CSV_PATH = HERE / "recipes_metadata.csv"


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def load_artifacts():
    if VEC_PATH.exists() and MATRIX_PATH.exists():
        meta = None
        if META_PARQUET_PATH.exists():
            meta = pd.read_parquet(META_PARQUET_PATH)
        elif META_CSV_PATH.exists():
            meta = pd.read_csv(META_CSV_PATH, encoding="utf-8")
        if meta is None:
            return None, None, None
        vec = joblib.load(VEC_PATH)
        X = sparse.load_npz(MATRIX_PATH)
        if "Ingredients" in meta.columns:
            meta["Ingredients"] = meta["Ingredients"].astype(str).str.lower()
        if "Title" in meta.columns:
            meta["Title"] = meta["Title"].astype(str)
        return vec, X, meta
    return None, None, None


def load_fallback_dataframe():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find {DATA_FILE}. Place recipes.csv in backend/")
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig").dropna(subset=["Ingredients"])
    df["Ingredients"] = df["Ingredients"].astype(str).str.lower()
    title_col = (
        "Recipe_name" if "Recipe_name" in df.columns else
        ("Title" if "Title" in df.columns else None)
    )
    if title_col is None:
        df["Title"] = df.index.to_series().apply(lambda i: f"Recipe {i}")
        title_col = "Title"
    vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w\-]+\b")
    X = vec.fit_transform(df["Ingredients"])
    df = df.reset_index().rename(columns={"index": "recipe_idx"})
    df["title_slug"] = df[title_col].astype(str).apply(slugify)
    return vec, X, df, title_col


# ---------- Load model + metadata ----------
vec, X_recipes, df_meta = load_artifacts()
if vec is None:
    vec, X_recipes, df_meta, _title_col = load_fallback_dataframe()
    title_col = _title_col
else:
    title_col = "Title" if "Title" in df_meta.columns else "Recipe_name"
    df_meta = df_meta.rename(columns={"Title": "Title"})  
    if "recipe_idx" not in df_meta.columns:
        df_meta = df_meta.reset_index().rename(columns={"index": "recipe_idx"})
    df_meta["title_slug"] = df_meta["Title"].apply(slugify)
    df_meta = df_meta.reset_index(drop=True)

df = df_meta
df["recipe_idx"] = df["recipe_idx"].astype(int)

# build lookup from filename stem -> relative path inside IMAGES_DIR
image_lookup = {}
if IMAGES_DIR.exists():
    # search recursively to handle nested folders (since some datasets have an extra 'Food Images' subfolder)
    for p in IMAGES_DIR.rglob("*"):
        if p.is_file():
            # store the path relative to the images directory so we can mount IMAGES_DIR at /images
            rel_path = p.relative_to(IMAGES_DIR).as_posix()
            # use the filename stem as the lookup key (lowercase) -> relative path as value
            image_lookup[p.stem.lower()] = rel_path

def find_image(row):
    candidates = []
    if "Image_Name" in row and pd.notna(row["Image_Name"]):
        candidates.append(str(Path(row["Image_Name"]).stem).lower())
    candidates.append(str(row.get("title_slug", "")).lower())
    for slug in candidates:
        if not slug:
            continue
        if slug in image_lookup:
            return image_lookup[slug]
        for key in image_lookup:
            if slug in key or key in slug:
                return image_lookup[key]
    return None

df["image_file"] = df.apply(find_image, axis=1)

# ---------- FastAPI app ----------
app = FastAPI(title="AI-Powered Grocery & Recipe Recommender")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
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
            "recipe_id": int(row["recipe_idx"]),
            "recipe_name": row[title_col],
            "ingredients": row["Ingredients"],
            "instructions": row.get("Instructions") if "Instructions" in row else None,
            "similarity": float(sims[i]),
            "image_url": image_url
        })
    return {"results": results}


# ----- Favorites persistence (simple JSON file) -----
FAV_FILE = HERE / "favorites.json"

def load_favorites():
    if not FAV_FILE.exists():
        return []
    try:
        import json
        with open(FAV_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_favorites(favs):
    import json
    with open(FAV_FILE, "w", encoding="utf-8") as f:
        json.dump(favs, f, ensure_ascii=False, indent=2)


@app.get("/favorites")
def get_favorites():
    favs = load_favorites()
    # return the full recipe details for each favorite id
    result = []
    for rid in favs:
        if rid in df["recipe_idx"].values:
            row = df[df["recipe_idx"] == rid].iloc[0]
            img_file = row.get("image_file")
            image_url = f"/images/{img_file}" if img_file else None
            result.append({
                "recipe_id": int(row["recipe_idx"]),
                "recipe_name": row[title_col],
                "ingredients": row["Ingredients"],
                "instructions": row.get("Instructions") if "Instructions" in row else None,
                "image_url": image_url,
            })
    return {"favorites": result}


@app.post("/favorites")
def add_favorite(item: dict):
    # expects JSON body: { "recipe_id": <int> }
    favs = load_favorites()
    rid = int(item.get("recipe_id"))
    if rid not in favs:
        favs.append(rid)
        save_favorites(favs)
    return {"favorites": favs}


@app.delete("/favorites/{recipe_id}")
def remove_favorite(recipe_id: int):
    favs = load_favorites()
    recipe_id = int(recipe_id)
    if recipe_id in favs:
        favs = [r for r in favs if r != recipe_id]
        save_favorites(favs)
    return {"favorites": favs}
