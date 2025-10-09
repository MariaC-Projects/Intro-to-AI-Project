import { useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [pantry, setPantry] = useState("tomato, pasta, garlic");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");

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
              {results.map((r, idx) => (
                <tr key={idx}>
                  <td>{r.recipe_name}</td>
                  <td style={{ maxWidth: 450 }}>{r.ingredients}</td>
                  <td>{r.similarity.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
