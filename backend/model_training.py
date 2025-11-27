import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent  # the backend folder

# Try to find a data file automatically (CSV or Excel)
candidates = [
    HERE / "recipes.csv",
    HERE / "recipes.xlsx",
    HERE.parent / "recipes.csv",
    HERE.parent / "recipes.xlsx",
    *HERE.glob("Food*Recipe*Image*.csv"),
    *HERE.glob("Food*Recipe*Image*.xlsx"),
    *HERE.parent.glob("Food*Recipe*Image*.csv"),
    *HERE.parent.glob("Food*Recipe*Image*.xlsx"),
]

DATA_FILE = next((p for p in candidates if p.exists()), None)
if DATA_FILE is None:
    raise FileNotFoundError(
        "Could not find your dataset. Put 'recipes.csv' or 'recipes.xlsx' "
        "in the backend folder, or keep the Kaggle file name and place it in backend."
    )

# Read depending on extension
if DATA_FILE.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
else:
    df = pd.read_excel(DATA_FILE)

print(f"? Loaded data file: {DATA_FILE}  rows={len(df)}")

# ---------------------------------------------------------------------------
# Keyword-based filtering instead of model training
# ---------------------------------------------------------------------------
KEYWORDS = ["low-fat", "vegan", "salad"]  # extend with any tags you need


def filter_by_keywords(dataframe: pd.DataFrame, keywords) -> pd.DataFrame:
    kw = [str(k).strip().lower() for k in keywords if str(k).strip()]
    if not kw:
        return dataframe.copy()
    pattern = "|".join(re.escape(k) for k in kw)
    # Keep rows whose Ingredients contain any of the keywords
    return dataframe[dataframe["Ingredients"].astype(str).str.lower().str.contains(pattern, na=False)]


# Clean and filter
df = df.dropna(subset=["Ingredients"]).copy()
df["Ingredients"] = df["Ingredients"].astype(str).str.lower()
filtered = filter_by_keywords(df, KEYWORDS)

print(f"Recipes matching keywords ({', '.join(KEYWORDS)}): {len(filtered)} / {len(df)}")

# Save filtered set so the API can load/use it
out_path = HERE / "filtered_recipes.parquet"
filtered.to_parquet(out_path, index=False)
print(f"? Saved filtered recipes to {out_path}")
