# Movie Recommender System

> A content-based movie recommendation engine built on the TMDB 5000 dataset — combining NLP feature engineering, Porter stemming, and cosine similarity to surface the five most relevant films for any given title.

---

## Overview

This project implements a **content-based filtering** recommender system. Unlike collaborative filtering (which relies on user behaviour data), this approach builds a rich semantic representation of each movie — called a **tag** — from its plot, genres, keywords, top cast, and director. Movies are then compared purely on the basis of what they *are*, not who watched them.

The result: type in a movie title, get back five thematically and contextually similar recommendations.

---

## How It Works

```
TMDB Movies CSV  ──┐
                   ├──► Merge on title ──► Feature Selection ──► Tag Construction
TMDB Credits CSV ──┘
                                                    │
                             ┌──────────────────────┘
                             ▼
                    Text Normalisation
                    (lowercase + Porter stemming)
                             │
                             ▼
                    CountVectorizer
                    (Bag of Words, top 5000 features,
                     English stop words removed)
                             │
                             ▼
                    Cosine Similarity Matrix
                    (4800 × 4800)
                             │
                             ▼
                    Ranked Top-5 Recommendations
```

---

## Feature Engineering

Each movie is reduced to a single `tags` string — a concatenation of five feature streams:

| Feature | Source | Processing |
|---|---|---|
| `overview` | Plot synopsis | Split into word tokens |
| `genres` | Genre list (JSON) | Extract `name` field |
| `keywords` | Keyword list (JSON) | Extract `name` field |
| `cast` | Cast list (JSON) | Top 3 actor names only |
| `crew` | Crew list (JSON) | Director name only |

Multi-word names (e.g. `"Sam Raimi"`, `"Science Fiction"`) have spaces removed (`"SamRaimi"`, `"ScienceFiction"`) to prevent them being split into unrelated tokens during vectorisation.

All tags are lowercased and passed through a **Porter Stemmer** — reducing inflected words to their root form (`"running"` → `"run"`, `"loves"` → `"love"`) so semantically equivalent terms are treated as identical.

---

## Similarity Computation

```python
# Bag-of-Words vectorisation
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()   # shape: (4800, 5000)

# Pairwise cosine similarity
similarity = cosine_similarity(vectors)             # shape: (4800, 4800)
```

**Cosine similarity** measures the angle between two tag vectors, making it robust to document length — a long plot summary doesn't artificially inflate similarity scores over a short one.

---

## Recommendation Function

```python
def movies_recommender(movie):
    movie_index = df[df['title'] == movie].index[0]
    distances   = similarity[movie_index]
    top_movies  = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]                          # exclude self (index 0)

    for i in top_movies:
        print(df.iloc[i[0]].title)
```

**Example output:**
```
movies_recommender('Inception')
─────────────────────────────
The Dark Knight
Interstellar
The Prestige
Shutter Island
The Dark Knight Rises
```

---

## Dataset

**Source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)

| File | Description |
|---|---|
| `tmdb_5000_movies.csv` | Movie metadata — overview, genres, keywords, budget, revenue |
| `tmdb_5000_credits.csv` | Cast and crew information per movie |

After merging and dropping 3 rows with missing overviews, the working dataset contains approximately **4,800 movies**.

---

## Project Structure

```
movie-recommender-system/
│
├── Movies_Recommender_system.ipynb   ← Main notebook (EDA + pipeline + inference)
├── tmdb_5000_movies.csv              ← Movie metadata
├── tmdb_5000_credits.csv             ← Cast & crew data
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn nltk
```

### NLTK Data

```python
import nltk
nltk.download('punkt')
```

### Run

1. Download both CSV files from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place them in the project root.
2. Open `Movies_Recommender_system.ipynb` and run all cells.
3. Call the recommender at the bottom of the notebook:

```python
movies_recommender('The Dark Knight')
```

---

## Pipeline Summary

```
Raw CSVs
  └─► Merge on title
        └─► Select: movie_id, title, overview, genres, keywords, cast, crew
              └─► Parse JSON columns with ast.literal_eval
                    └─► Extract: genres(all), keywords(all), cast(top 3), director(1)
                          └─► Remove spaces from multi-word tokens
                                └─► Concatenate into tags string
                                      └─► Lowercase + Porter stemming
                                            └─► CountVectorizer (top 5000)
                                                  └─► Cosine similarity matrix
                                                        └─► Top-5 lookup
```

---

## Key Concepts

**Content-Based Filtering** — Recommendations are driven entirely by movie attributes, not user history. This means the system works cold-start — no user data needed.

**Porter Stemming** — Normalises word variants to a common root before vectorisation, improving token overlap between semantically related movies.

**Bag of Words (CountVectorizer)** — Represents each tag document as a sparse vector of token frequencies. Limiting to `max_features=5000` keeps the vocabulary manageable while retaining the most discriminative terms.

**Cosine Similarity** — Direction-based distance metric that is length-agnostic, making it well-suited for text where document size varies significantly.

---

## Limitations & Future Improvements

- **Bag of Words loses word order** — TF-IDF or sentence embeddings (e.g. SBERT) would capture semantics more richly
- **No personalisation** — Adding collaborative filtering (user ratings) would enable hybrid recommendations
- **Cast truncated to top 3** — Expanding cast coverage may improve recommendations for ensemble films
- **Stemming can over-reduce** — Lemmatisation (WordNet) would be more linguistically accurate
- **No UI** — A Streamlit or FastAPI frontend would make this demo-ready

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-4EA94B?style=flat)

---

## License

This project is released under the [MIT License](LICENSE).
