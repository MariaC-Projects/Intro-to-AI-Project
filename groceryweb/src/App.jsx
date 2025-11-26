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
  const [activeTab, setActiveTab] = useState("results");
  const [shoppingList, setShoppingList] = useState([]);

  useEffect(() => {
    // load favorites from backend on mount
    fetch(`${API}/favorites`).then((res) => res.json()).then((data) => {
      setFavorites((data && data.favorites) || []);
    }).catch(() => {});
  }, []);

  async function recommend(customPantry) {
    setLoading(true);
    setError("");
    setResults([]);
    const pantryValue = (customPantry ?? pantry).trim();
    setPantry(pantryValue);
    try {
      const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pantry: pantryValue, top_k: Number(topK) || 5 }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      setResults(data.results || []);
      setActiveTab("results");
    } catch (e) {
      setError(`Could not get recommendations. ${e.message}`);
    } finally {
      setLoading(false);
    }
  }

  function addToShoppingList(items) {
    const incoming = Array.isArray(items) ? items : [items];
    setShoppingList((prev) => {
      const next = [...prev];
      incoming.forEach((raw) => {
        const item = String(raw || "").trim();
        if (!item) return;
        const exists = next.some((i) => i.toLowerCase() === item.toLowerCase());
        if (!exists) next.push(item);
      });
      return next;
    });
    setActiveTab("shopping");
  }

  function removeFromShoppingList(item) {
    setShoppingList((prev) => prev.filter((i) => i.toLowerCase() !== item.toLowerCase()));
  }

  function clearShoppingList() {
    setShoppingList([]);
  }

  return (
    <div className="app-container fade-in">
      <h1>PantryChef</h1>
      <h2>AI-Powered Grocery & Recipe Recommender</h2>

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
        <div className="tab-list">
          <button className={`tab ${activeTab === "results" ? "active" : ""}`} onClick={() => setActiveTab("results")}>Top Recipes</button>
          <button className={`tab ${activeTab === "favorites" ? "active" : ""}`} onClick={() => setActiveTab("favorites")}>Favorites</button>
          <button className={`tab ${activeTab === "shopping" ? "active" : ""}`} onClick={() => setActiveTab("shopping")}>Shopping List</button>
        </div>

        {activeTab === "results" && (
          <>
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
                                    .then(() => fetch(`${API}/favorites`).then(r => r.json()))
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
                            <>
                              <div className="ingredient-inline">
                                {ingredientsList.map((ing, i) => (
                                  <button
                                    key={`${r.recipe_id || idx}-ing-${i}`}
                                    className="ingredient-pill"
                                    onClick={() => addToShoppingList(ing)}
                                    type="button"
                                  >
                                    + {ing}
                                  </button>
                                ))}
                              </div>
                              <button className="link-button" type="button" onClick={() => addToShoppingList(ingredientsList)}>
                                Add all to shopping list
                              </button>
                            </>
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
          </>
        )}

        {activeTab === "favorites" && (
          <>
            {favorites.length === 0 ? (
              <p>No favorites yet. Save recipes from results above.</p>
            ) : (
              <ul>
                {favorites.map((f) => {
                  const favIngredients = normalizeIngredients(f.ingredients);
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
                          {favIngredients.length ? favIngredients.join(" • ") : <span style={{ color: '#666', fontSize: 12 }}>No ingredients listed</span>}
                        </div>
                        {favIngredients.length > 0 && (
                          <button className="link-button" type="button" onClick={() => addToShoppingList(favIngredients)}>
                            Add all to shopping list
                          </button>
                        )}
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
          </>
        )}

        {activeTab === "shopping" && (
          <div>
            {shoppingList.length === 0 ? (
              <p>Your shopping list is empty. Click ingredients to add them.</p>
            ) : (
              <>
                <ul className="shopping-list">
                  {shoppingList.map((item, i) => (
                    <li key={`${item}-${i}`}>
                      <span>{item}</span>
                      <button className="ghost" type="button" onClick={() => removeFromShoppingList(item)}>Remove</button>
                    </li>
                  ))}
                </ul>
                <div style={{ marginTop: 10 }}>
                  <button className="ghost" type="button" onClick={clearShoppingList}>Clear list</button>
                  <button
                    style={{ marginLeft: 8 }}
                    onClick={() => recommend(shoppingList.join(", "))}
                    disabled={loading || shoppingList.length === 0}
                  >
                    {loading ? "Recommending..." : "Recommend from shopping list"}
                  </button>
                </div>
              </>
            )}
          </div>
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
                  {normalizeIngredients(selectedRecipe.ingredients).map((ing, i) => (
                    <button
                      key={`modal-ing-${i}`}
                      className="ingredient-pill"
                      onClick={() => addToShoppingList(ing)}
                      type="button"
                    >
                      + {ing}
                    </button>
                  ))}
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
