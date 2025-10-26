# The Movie Recommender

Welcome to **The Movie Recommender**, a Streamlit web application that provides movie recommendations for both new and existing users using hybrid content-based and collaborative filtering.

---

## Features

- **New User Flow:** Choose up to 3 favorite movies and receive recommendations based on similar content features like actors, directors, producers, writers, and genres.
- **Existing User Flow:** Explore already-rated movies, receive user-based collaborative recommendations, and hybrid predictions based on historical preferences.
- **Movie Filtering:** Refine the movie selection by actors, categories, producers, and more.
- **Movie Posters:** Posters are fetched dynamically from the OMDB API.
- **User Session Management:** A random `user_id` is generated per session for temporary identification.

---

## Recommender System

The app uses a `FilmRecommender` class that combines:

- **Content-Based Filtering** using movie metadata.
- **Hybrid Approach** that combines Content-Based & Collaborative Filtering & Popularity
- **Predict Movie rating function**

---

## Directory Structure

```
project/
│
├── code/
│   └── notebooks/
│       ├── 01.Imdb_pipeline.ipynb
│       ├── 02.Tmdb_pipeline.ipynb
│       ├── 03.Omdb_pipeline.ipynb
│       ├── 04.Movielens_pipeline.ipynb
│       ├── 05.Pipelines_combined.ipynb
│       ├── 06.Preprocessed_dataframe.ipynb
│       ├── 07.Model_content_based.ipynb
│       ├── 08.Model_collab_user_based.ipynb
│       ├── 09.Model_hybrid.ipynb
│       ├── 10.Model_random_forest.ipynb
│   └── streamlit_app
│       ├── app.py
│       ├── film_recommender.py
│
├── data/
│   └── cleaned/
│       └──  joblib_dataframes/
│            ├── df_final.joblib
│            ├── df_final_matrix.joblib
│            ├── df_movies_combined_rf.joblib
│       └── dim & bridge.csv files
│
│   └── raw/
│       ├── box_office_mapping
│       ├── imdb
│       ├── movielens
│       ├── omdb
│       ├── tmdb
│
└── README.md 
```

---

## Api Key

The app fetches movie posters using the OMDB API:

```
OMDB_API_KEY = "###################"
```

You can set your own OMDB API key if needed.

---

## Data Caching & Loading

The app uses:
- `st.cache_data` for dimensional data
- `st.cache_resource` for loading models and ratings

---

## App Pages

- **Welcome Page:** Introduction and guidance.
- **New User:** Select up to 3 liked movies and get content-based recommendations.
- **Existing User:** View past ratings and get hybrid recommendations.

---

## Notes

- Ratings and user preferences are stored using `joblib`.
- IDs for actors, writers, etc., are mapped to human-readable names using dimension tables.
- User ID is generated randomly unless selected manually from existing IDs.

---


