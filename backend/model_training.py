"""
Train a TF-IDF model for recipe retrieval and persist artifacts for the API.

Outputs (all in backend/):
- tfidf_vectorizer.joblib  (fitted TfidfVectorizer)
- recipe_tfidf.npz         (sparse TF-IDF matrix aligned to metadata rows)
- recipes_metadata.parquet (recipe ids + text fields + image name + slug)
"""

from pathlib import Path
import re
import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).parent  # backend folder

# Try to find a data file automatically (CSV or Excel)
CANDIDATES = [
    HERE / "recipes.csv",
    HERE / "recipes.xlsx",
    HERE.parent / "recipes.csv",
    HERE.parent / "recipes.xlsx",
    *HERE.glob("Food*Recipe*Image*.csv"),
    *HERE.glob("Food*Recipe*Image*.xlsx"),
    *HERE.parent.glob("Food*Recipe*Image*.csv"),
    *HERE.parent.glob("Food*Recipe*Image*.xlsx"),
]


def find_data_file() -> Path:
    data_file = next((p for p in CANDIDATES if p.exists()), None)
    if data_file is None:
        raise FileNotFoundError(
            "Could not find your dataset. Put 'recipes.csv' or 'recipes.xlsx' "
            "in the backend folder, or keep the Kaggle file name and place it in backend."
        )
    return data_file


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path)
    print(f"✅ Loaded data file: {path}  rows={len(df)}")
    return df


def pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    # prefer cleaned ingredients if present
    text_col = "Cleaned_Ingredients" if "Cleaned_Ingredients" in df.columns else "Ingredients"
    title_col = (
        "Recipe_name"
        if "Recipe_name" in df.columns
        else ("Title" if "Title" in df.columns else None)
    )
    if title_col is None:
        df["Title"] = df.index.to_series().apply(lambda i: f"Recipe {i}")
        title_col = "Title"
    required = {text_col, title_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns not found: {missing}")

    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.lower()
    df[title_col] = df[title_col].astype(str)

    # keep optional columns if present
    keep_cols = [title_col, text_col]
    if "Instructions" in df.columns:
        keep_cols.append("Instructions")
    if "Image_Name" in df.columns:
        keep_cols.append("Image_Name")

    df = df[keep_cols]
    df = df.reset_index(drop=True).reset_index().rename(columns={"index": "recipe_idx"})
    df["title_slug"] = df[title_col].apply(slugify)
    df = df.rename(columns={text_col: "Ingredients", title_col: "Title"})
    return df


def train_vectorizer(text_series: pd.Series) -> TfidfVectorizer:
    vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w\-]+\b", ngram_range=(1, 2), min_df=2)
    matrix = vec.fit_transform(text_series)
    print(f" Trained TF-IDF: vocab={len(vec.vocabulary_)}  shape={matrix.shape}")
    return vec, matrix


def save_artifacts(vec: TfidfVectorizer, matrix, meta: pd.DataFrame):
    joblib.dump(vec, HERE / "tfidf_vectorizer.joblib")
    sparse.save_npz(HERE / "recipe_tfidf.npz", matrix)
    try:
        meta.to_parquet(HERE / "recipes_metadata.parquet", index=False)
        saved_meta = "recipes_metadata.parquet"
    except Exception:
        meta.to_csv(HERE / "recipes_metadata.csv", index=False, encoding="utf-8")
        saved_meta = "recipes_metadata.csv"
    print(f" Saved artifacts: tfidf_vectorizer.joblib, recipe_tfidf.npz, {saved_meta}")


def main():
    data_file = find_data_file()
    raw_df = load_data(data_file)
    df = pick_columns(raw_df)
    vec, X = train_vectorizer(df["Ingredients"])
    save_artifacts(vec, X, df)


if __name__ == "__main__":
    main()
