import { useEffect, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function normalizeIngredients(raw) {
  if (Array.isArray(raw)) {
    return raw.map((i) => String(i).trim()).filter(Boolean);
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        return parsed.map((i) => String(i).trim()).filter(Boolean);
      }
    } catch (_) {
      // not JSON, fall back to string cleanup
    }
    return trimmed
      .replace(/^\[|\]$/g, "")
      .replace(/['"]/g, "")
      .split(/\s*,\s*/)
      .filter(Boolean)
      .map((i) => i.trim());
  }
  return [];
}

export default function App() {
  const [pantry, setPantry] = useState("tomato, pasta, garlic");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    // load favorites from backend on mount
    fetch(`${API}/favorites`).then((res) => res.json()).then((data) => {
      setFavorites((data && data.favorites) || []);
    }).catch(() => {});
  }, []);

  async function recommend() {
    setLoading(true);
    setError("");
    setResults([]);
    try {
      const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pantry, top_k: Number(topK) || 5 }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      setResults(data.results || []);
    } catch (e) {
      setError(`Could not get recommendations. ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-container fade-in">
      <h1>AI-Powered Grocery & Recipe Recommender</h1>

      <div className="card" style={{ marginTop: 16 }}>
        <label>Your Pantry</label>
        <input
          className="input-field"
          value={pantry}
          onChange={(e) => setPantry(e.target.value)}
          placeholder="e.g. tomato, pasta, garlic"
        />

        <label style={{ marginTop: 12 }}>How many recipes (Top-K)</label>
        <input
          type="number"
          min="1"
          max="20"
          value={topK}
          onChange={(e) => setTopK(e.target.value)}
          className="input-field"
        />

        <button onClick={recommend} style={{ marginTop: 14 }}>
          {loading ? "Finding recipes..." : "Recommend"}
        </button>

        {error && <p style={{ color: "#b91c1c", marginTop: 12 }}>{error}</p>}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>Top Recipes</h2>
        {results.length === 0 && !loading ? (
          <p>No results yet. Try adding ingredients above.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Recipe</th>
                <th>Ingredients</th>
                <th>Similarity</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, idx) => {
                const isFav = favorites.some((f) => f.recipe_id === r.recipe_id);
                const ingredientsList = normalizeIngredients(r.ingredients);
                const ingredientsText = ingredientsList.join(" • ");
                const similarityPercent = `${((r.similarity || 0) * 100).toFixed(1)}%`;
                return (
                  <tr key={idx}>
                    <td style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      {r.image_url ? (
                        <img
                          src={API.replace(/\/$/, "") + r.image_url}
                          alt={r.recipe_name}
                          style={{ width: 96, height: 96, objectFit: 'cover', borderRadius: 8, cursor: 'pointer' }}
                          onClick={() => setSelectedRecipe(r)}
                          onError={(e) => { e.currentTarget.style.display = 'none'; }}
                        />
                      ) : (
                        <div style={{ width: 96, height: 96, background: '#eee', borderRadius: 8 }} />
                      )}
                      <div>
                        <div>{r.recipe_name}</div>
                        <div style={{ marginTop: 6 }}>
                          {isFav ? (
                            <button onClick={() => {
                              // remove favorite
                              fetch(`${API}/favorites/${r.recipe_id}`, { method: 'DELETE' })
                                .then(res => res.json())
                                .then(data => {
                                  // update favorites detail list by refetching
                                  return fetch(`${API}/favorites`).then(r => r.json());
                                })
                                .then(d => setFavorites(d.favorites || []))
                                .catch(() => {});
                            }}>Unsave</button>
                          ) : (
                            <button onClick={() => {
                              fetch(`${API}/favorites`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ recipe_id: r.recipe_id })
                              }).then(res => res.json())
                                .then(() => fetch(`${API}/favorites`).then(r => r.json()))
                                .then(d => setFavorites(d.favorites || []))
                                .catch(() => {});
                            }}>Save</button>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="ingredients-cell">
                      {ingredientsList.length ? (
                        <div className="ingredient-inline">
                          {ingredientsText}
                        </div>
                      ) : (
                        <span style={{ color: '#666' }}>No ingredients listed</span>
                      )}
                    </td>
                    <td>{similarityPercent}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h2>Favorites</h2>
        {favorites.length === 0 ? (
          <p>No favorites yet. Save recipes from results above.</p>
        ) : (
          <ul>
            {favorites.map((f) => {
              const favIngredients = normalizeIngredients(f.ingredients);
              const favText = favIngredients.join(" • ");
              return (
                <li key={f.recipe_id} style={{ marginBottom: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
                  {f.image_url ? (
                    <img src={API.replace(/\/$/, '') + f.image_url} alt={f.recipe_name} style={{ width: 64, height: 64, objectFit: 'cover', borderRadius: 6, cursor: 'pointer' }} onClick={() => setSelectedRecipe(f)} />
                  ) : (
                    <div style={{ width: 64, height: 64, background: '#eee', borderRadius: 6 }} />
                  )}
                  <div style={{ flex: 1 }}>
                    <div>{f.recipe_name}</div>
                    <div className="ingredient-inline small">
                      {favIngredients.length ? favText : <span style={{ color: '#666', fontSize: 12 }}>No ingredients listed</span>}
                    </div>
                  </div>
                  <button onClick={() => {
                    fetch(`${API}/favorites/${f.recipe_id}`, { method: 'DELETE' })
                      .then(res => res.json())
                      .then(() => fetch(`${API}/favorites`).then(r => r.json()))
                      .then(d => setFavorites(d.favorites || []))
                      .catch(() => {});
                  }}>Remove</button>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      {/* Lightbox / modal for selected recipe */}
      {selectedRecipe && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999 }} onClick={() => setSelectedRecipe(null)}>
          <div style={{ background: '#fff', borderRadius: 8, maxWidth: '90%', maxHeight: '90%', overflow: 'auto', padding: 20 }} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', gap: 16 }}>
              {selectedRecipe.image_url ? (
                <img src={API.replace(/\/$/, '') + selectedRecipe.image_url} alt={selectedRecipe.recipe_name} style={{ maxWidth: 420, maxHeight: '70vh', objectFit: 'cover', borderRadius: 8 }} />
              ) : (
                <div style={{ width: 420, height: 260, background: '#eee', borderRadius: 8 }} />
              )}
              <div style={{ flex: 1 }}>
                <h3 style={{ marginTop: 0 }}>{selectedRecipe.recipe_name}</h3>
                <div className="ingredient-inline" style={{ marginBottom: 8 }}>
                  {normalizeIngredients(selectedRecipe.ingredients).join(" • ")}
                </div>
                {selectedRecipe.instructions && (
                  <div style={{ marginTop: 12 }}>
                    <h4 style={{ margin: '8px 0' }}>Instructions</h4>
                    <div style={{ whiteSpace: 'pre-wrap', color: '#222' }}>{selectedRecipe.instructions}</div>
                  </div>
                )}
                <div style={{ marginTop: 12 }}>
                  <button onClick={() => setSelectedRecipe(null)}>Close</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
