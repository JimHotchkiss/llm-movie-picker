Here’s a polished, recruiter-friendly **README.md** you can drop into your repo. It’s eye-catching, clear, and shows off real data-science chops while staying easy to run.

---

# 🎬 MoodMatch: VAD-Powered Movie/Series Recommender

> *“Find the right **vibe**, not just the right **genre**.”*
> Given a short mood description (e.g., *“feel-good underdog sports story—rousing and triumphant; movie or miniseries both work”*), **MoodMatch** recommends titles that match both the **semantics** (plot/topic) and the **emotion** (VAD: Valence–Arousal–Dominance).

![screenshot-placeholder](docs/screenshot.png)

---

## ✨ Highlights

* **Emotion-aware search** using **VAD** (Valence, Arousal, Dominance)
* **Hybrid ranking**: semantic similarity (embeddings) + mood similarity (VAD) + genre match
* **Streamlit UI** with:

  * VAD sliders (or auto-detect VAD from your text via LLM)
  * “Why this matched” explanations (VAD closeness, genre overlap, semantic score)
  * Optional playlist export / shortlist save
* **Efficient data science pipeline**:

  * Precompute embeddings & VAD for your catalog once
  * Fast retrieval with NumPy (no vector DB required at 8.5k rows)
  * Optional Qdrant/Chroma integration

---

## 🧠 What makes this a Data Science project?

* **Problem framing → metrics**: turn vague “vibe” into numeric targets (VAD) and rank with a blended score
* **Feature engineering**: embeddings (semantics), VAD (mood), genre signals
* **Unsupervised learning & retrieval**: similarity search, clustering (optional)
* **Evaluation**: precision@K, ablations (weight sweeps, with/without VAD)
* **MLOps-lite**: cached artifacts (embeddings/VAD), reproducible scripts, config

---

## 🏗️ Architecture

```
User Text  ──► Parse Intent + VAD (LLM) ──► Filters (genre/type/audience)
                           │
                           ▼
                    Query Embedding (same model as catalog)
                           │
      ┌────────────────────┴──────────────────────┐
      ▼                                           ▼
Movie Embeddings (N×d, .npy)                Movie VAD (N×3, in parquet)
      │                                           │
      └─────────────► Hybrid Scoring ◄────────────┘
                     score = 0.5*sem + 0.3*vad + 0.2*genre
                                   │
                                   ▼
                            Top-K Recommendations
```

---

## 🧰 Tech Stack

* **Python**, **pandas**, **NumPy**, **scikit-learn**
* **Embeddings:** OpenAI `text-embedding-3-small` (or local `sentence-transformers`)
* **LLM (optional):** OpenAI for structured extraction + text→VAD
* **UI:** Streamlit
* **Storage:** `movies.parquet` + `embeddings.npy` (fast & simple)
  *Optional:* Qdrant/Chroma for vector search

---

## 📦 Repository Structure

```
.
├─ app/
│  └─ streamlit_app.py           # Streamlit UI
├─ data/
│  ├─ movies.parquet             # catalog + VAD columns
│  └─ embeddings.npy             # (N, d) float32, unit-normalized
├─ scripts/
│  ├─ precompute.py              # clean, embed, VAD, save artifacts
│  ├─ evaluate.py                # precision@K, ablations
│  └─ tmdb_enrich.py             # (optional) poster/keywords
├─ src/
│  ├─ extract.py                 # 1-call LLM: genres/type/audience + VAD
│  ├─ features.py                # embedding adapters, normalization
│  ├─ vad.py                     # LLM or lexicon; teacher→student option
│  ├─ rank.py                    # hybrid scoring + MMR diversity
│  └─ utils.py
├─ docs/
│  ├─ screenshot.png
│  └─ model_card.md
├─ .env.example
└─ README.md
```

---

## 🔧 Setup

1. **Clone & env**

```bash
git clone <your-repo-url>
cd moodmatch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

2. **Configure** `.env`

```
# Choose ONE embedding source
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai            # or: local
EMBEDDING_MODEL=text-embedding-3-small
# LLM for parsing/VAD (optional but nice)
LLM_MODEL=gpt-4o-mini
```

3. **Data**

* Place your CSV in `data/raw/` (e.g., Netflix catalog).
* (Optional) Add TMDB API key to enrich posters/keywords.

---

## 🏗️ One-time Precompute

Creates `movies.parquet` (with `vad_v, vad_a, vad_d`) and `embeddings.npy`.

```bash
python scripts/precompute.py \
  --input data/raw/netflix_titles.csv \
  --output-parquet data/movies.parquet \
  --output-embeddings data/embeddings.npy \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  --vad-mode teacher_student --sample-size 400
```

**How VAD is computed:**

* **Teacher–Student (recommended):**
  LLM labels ~400 samples → train a small regressor (Ridge/MLP) mapping **embedding → VAD** → predict VAD for all titles. Quality ≈ LLM, cost tiny.
* **Lexicon fallback:** fast/zero-cost baseline.
* Write VAD to `movies.parquet` as columns: `vad_v`, `vad_a`, `vad_d` (in [0,1]).

---

## ▶️ Run the App

```bash
streamlit run app/streamlit_app.py
```

**Features**

* Enter a mood prompt (or just use sliders)
* Auto-parse **genres/type/audience** + **VAD** (with confidence)
* Filter catalog, compute **semantic + mood** similarity, and **rank**
* See top picks with **explanations** and posters

---

## 🧪 Evaluation (optional but impressive)

```bash
python scripts/evaluate.py \
  --catalog data/movies.parquet \
  --embeddings data/embeddings.npy \
  --queries tests/queries.yml \
  --k 10 --weights "0.5,0.3,0.2"
```

* **Metrics:** precision@K, coverage, diversity (artist/franchise cap), silhouette (if clustering)
* **Ablations:** sweep weights `α, β, γ`; compare with/without VAD; cosine vs L2 for VAD

---

## 🧩 Scoring Details

* **Semantic similarity**: cosine between **query embedding** and **movie embedding**
* **Mood similarity**: cosine (or 1 − L2/√3) between **user VAD** and **movie VAD**
* **Genre overlap**: Jaccard between requested genres and movie genres

```
score = 0.50 * sim_embed
      + 0.30 * sim_vad
      + 0.20 * genre_overlap
# optional gate: require sim_vad ≥ 0.6
```

*(Tune weights; log runs in `experiments/results.csv`.)*

---

## 🧑‍🍳 Configuration Tips

* **No vector DB needed** at 8.5k rows (NumPy is blazing fast).
* Want résumé flair? Swap in **Qdrant**: upsert embeddings + metadata and use server-side filtering.
* **Normalize** all vectors to unit length → dot product = cosine.
* Cache all LLM outputs by a stable hash (title+year+overview).

---

## 🔒 Privacy & Safety

* Store **derived VAD** and embeddings; raw user prompts optional (off by default).
* Provide a “Demo Mode” using a static sample so anyone can try without API keys.

---

## 🗺️ Roadmap

* [ ] Cross-encoder rerank on top-50 for sharper final ordering
* [ ] Cluster explorer (“browse by vibe”)
* [ ] Playlist export / “Watchlist” persistence
* [ ] Multi-language support
* [ ] Deploy to Streamlit Community Cloud (one-click demo)

---

## 🙏 Acknowledgments

* VAD (Valence–Arousal–Dominance) concept from affective computing literature
* TMDB for metadata/posters (if used), Netflix CSV (sample catalogs), and open-source embedding models

---

## 📝 License

MIT (see `LICENSE`)

---

### Quick Start (TL;DR)

```bash
# 1) Install + configure
pip install -r requirements.txt
cp .env.example .env  # add OPENAI key or choose local embeddings

# 2) Precompute once
python scripts/precompute.py --input data/raw/netflix_titles.csv \
  --output-parquet data/movies.parquet --output-embeddings data/embeddings.npy

# 3) Run UI
streamlit run app/streamlit_app.py
```

Enjoy finding the **perfect vibe**. 🎥💫
