import './App.css';

export default function App() {
  return (
    <div className="app-container fade-in">
      <h1>AI-Powered Grocery & Recipe Recommender</h1>
      <label>Your Pantry</label>
      <input placeholder="tomato, pasta, garlic" />
      <button>Recommend</button>
      <div className="card">
        <h2>Top Recipes</h2>
        <p>Results will appear here...</p>
      </div>
    </div>
  );
}