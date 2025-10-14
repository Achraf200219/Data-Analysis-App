Upload CSV, Excel, JSON, Parquet, SQLite, or DuckDB files and analyze them using natural language queries. Backend: FastAPI + OpenRouter API + Pandas/SQLAlchemy/DuckDB. Frontend: React 18 + Vite + Plotly.js for interactive charts and summaries.

Setup: .env with OPENROUTER_API_KEY, then cd backend && pip install -r requirements.txt && uvicorn main:app --reload and cd frontend && npm install && npm run dev.

Access the app at http://localhost:5173 to chat, generate SQL/Pandas queries, visualize results, and explore follow-up questions.