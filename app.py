"""
Minimal web UI + JSON API for the search engine built in indexer.py / search_engine.py.

Run:
    python3 app.py

Then open http://127.0.0.1:5000

The index paths can be overridden via env vars (useful for pointing the demo
at a differently-located index without touching code):
    POSTINGS_PATH, SEEK_MAP_PATH, DOCMAP_PATH
"""

import os

from flask import Flask, jsonify, render_template_string, request

from search_engine import SearchEngine

app = Flask(__name__)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = SearchEngine(
            postings_path=os.environ.get("POSTINGS_PATH", "final_postings.txt"),
            seek_map_path=os.environ.get("SEEK_MAP_PATH", "seek_map.json"),
            docmap_path=os.environ.get("DOCMAP_PATH", "doc_id_map.json"),
        ).load()
    return _engine


INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Search Engine</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto; color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    .search-row { display: flex; gap: 8px; }
    input { flex: 1; padding: 10px 12px; font-size: 1rem; border: 1px solid #ccc; border-radius: 6px; }
    button { padding: 10px 16px; font-size: 1rem; border: none; border-radius: 6px; background: #1a1a1a; color: #fff; cursor: pointer; }
    #meta { color: #666; font-size: 0.85rem; margin-top: 10px; }
    ul { padding-left: 0; list-style: none; margin-top: 14px; }
    li { padding: 8px 0; border-bottom: 1px solid #eee; word-break: break-all; }
  </style>
</head>
<body>
  <h1>Search Engine</h1>
  <div class="search-row">
    <input id="q" placeholder="Search the index..." autofocus>
    <button onclick="doSearch()">Search</button>
  </div>
  <p id="meta"></p>
  <ul id="results"></ul>
  <script>
    async function doSearch() {
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      const start = performance.now();
      const res = await fetch('/search?q=' + encodeURIComponent(q));
      const data = await res.json();
      const elapsed = (performance.now() - start).toFixed(1);
      document.getElementById('meta').textContent =
        data.results.length + ' result(s) in ' + elapsed + ' ms';
      const list = document.getElementById('results');
      list.innerHTML = '';
      data.results.forEach(url => {
        const li = document.createElement('li');
        if (/^https?:\\/\\//.test(url)) {
          const a = document.createElement('a');
          a.href = url;
          a.textContent = url;
          a.target = '_blank';
          a.rel = 'noopener noreferrer';
          li.appendChild(a);
        } else {
          li.textContent = url;
        }
        list.appendChild(li);
      });
    }
    document.getElementById('q').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') doSearch();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/search")
def search_route():
    query = request.args.get("q", "").strip()
    top_k = max(1, min(request.args.get("k", default=5, type=int) or 5, 20))
    if not query:
        return jsonify({"query": query, "results": []})
    results = get_engine().search(query, top_k=top_k)
    return jsonify({"query": query, "results": results})


if __name__ == "__main__":
    get_engine()  # fail fast if the index files are missing, before serving requests
    app.run(debug=True)
