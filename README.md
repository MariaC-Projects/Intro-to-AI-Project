# PantryChef  
### *AI-Powered Grocery & Recipe Recommender*

PantryChef is an AI-driven web application that recommends recipes based on the ingredients you already have at home.  
It uses Natural Language Processing (TF–IDF vectorization + cosine similarity) to match your pantry items with the most relevant recipes from a dataset of ~13,000 recipes.

Link for Presentation and Slides: https://drive.google.com/drive/folders/13wKEiGZXcPGe7NkelWh8slM3nRaGVM7u

---
## Features
- Ingredient-based recipe recommendations  
- Built-in TF–IDF model trained on real recipe data  
- Recipe images, ingredients, instructions  
- Save & view favorite recipes  
- Clean React frontend + FastAPI backend  
- Lightweight, fast, and fully local
---
##  Requirements
- Python 3.10+  
- Node.js & npm  
- Dataset: *Food Ingredients and Recipes Dataset* (Kaggle)  
  - Place `recipes.csv` inside your `backend/` folder  
  - Place food images inside `backend/Food Images/`

---

## AI Model
The AI uses:
- **TF-IDF Vectorization** – converts recipe ingredients into numerical vectors  
- **Cosine Similarity** – measures closeness between user ingredients and recipe vectors  

These models are trained once using `model_training.py`, which outputs:
- `tfidf_vectorizer.joblib`  
- `recipe_tfidf.npz`  
- `recipes_metadata.parquet`  

---

##  How to Run the Project

### 1. Start the Backend (FastAPI)
```bash
cd backend
python -m venv .venv
# activate virtual environment:
# PowerShell:
.\.venv\Scripts\Activate.ps1
# OR CMD:
.\.venv\Scripts\activate.bat

pip install -r requirements.txt  # or manually install fastapi, uvicorn, pandas, scikit-learn
uvicorn main:app --reload --port 8000
```
### 2. Frontend(React)
```bash
cd groceryweb
npm install
npm run dev
```
Open app in browser : http://localhost:5173

### Notes for installation
- Ensure the recipes.csv file is inside backend/
- Ensure all image files are inside backend/Food Images/
- The backend must be running before the React app
