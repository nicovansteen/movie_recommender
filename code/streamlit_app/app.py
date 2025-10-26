import streamlit as st
import pandas as pd
import joblib
import os
import random
import requests
from film_recommender import FilmRecommender

# ================================================================================= CONFIG =================================================================================
st.set_page_config(page_title="The Movie Recommender", layout="wide")

OMDB_API_KEY = "a039f4dd"
OMDB_BASE_URL = "http://www.omdbapi.com/"

DATA_DIR = os.path.join('..', '..', 'data', 'cleaned')
JOBLIB_DIR = os.path.join(DATA_DIR, 'joblib_dataframes')
RATINGS_PATH = os.path.join(JOBLIB_DIR, 'df_final_matrix.joblib')

DIM_ACTORS_PATH = os.path.join(DATA_DIR, 'dim_movie_actors.csv')
DIM_DIRECTORS_PATH = os.path.join(DATA_DIR, 'dim_movie_directors.csv')
DIM_PRODUCERS_PATH = os.path.join(DATA_DIR, 'dim_movie_producers.csv')
DIM_WRITERS_PATH = os.path.join(DATA_DIR, 'dim_movie_writers.csv')
DIM_CATEGORIES_PATH = os.path.join(DATA_DIR, 'dim_movie_categories.csv')

# ========================= User Session =========================
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = random.randint(100000, 999999)

if 'selected_movies_titles' not in st.session_state:
    st.session_state['selected_movies_titles'] = []

user_id = st.session_state['user_id']

# =============== Help function map id to name ===================
@st.cache_data(show_spinner=False)
def load_dim_data(path, id_col, name_col):
    """Loads a dimension CSV and returns a dictionary mapping IDs to names."""
    try:
        df = pd.read_csv(path)
        return dict(zip(df[id_col], df[name_col]))
    except FileNotFoundError:
        st.error(f"Dimension file not found: {path}. Filters may not work correctly.")
        return {}
    except KeyError as e:
        st.error(f"Missing expected column in {path}: {e}. Please check your dimension CSVs.")
        return {}
    except Exception as e:
        st.error(f"Error loading dimension file {path}: {e}")
        return {}

actor_id_to_name = load_dim_data(DIM_ACTORS_PATH, 'actor_name_id', 'actor_name')
director_id_to_name = load_dim_data(DIM_DIRECTORS_PATH, 'director_name_id', 'director_name') 
producer_id_to_name = load_dim_data(DIM_PRODUCERS_PATH, 'producer_name_id', 'producer_name')
writer_id_to_name = load_dim_data(DIM_WRITERS_PATH, 'writer_name_id', 'writer_name')
category_id_to_name = load_dim_data(DIM_CATEGORIES_PATH, 'movie_category_id', 'movie_category')

def map_ids_to_names_str(id_string, id_to_name_map):
    """Converts a pipe-separated string of IDs to a pipe-separated string of names using a map."""
    if pd.isna(id_string) or id_string == '':
        return ''
    ids = id_string.split('|')
    # Use .get() with a fallback to the original ID if name not found, to avoid errors
    names = [id_to_name_map.get(id_val.strip(), id_val.strip()) for id_val in ids]
    return '|'.join(names)

# ============== Load recommender & data =================
@st.cache_resource(show_spinner="Loading recommender model and data...")
def load_recommender_and_data():
    try:
        recommender_instance = FilmRecommender(DATA_DIR, JOBLIB_DIR)
        movies_df_loaded = recommender_instance.movies.copy()
        
        movies_df_loaded['actors'] = movies_df_loaded['actors'].apply(lambda x: map_ids_to_names_str(x, actor_id_to_name))
        movies_df_loaded['directors'] = movies_df_loaded['directors'].apply(lambda x: map_ids_to_names_str(x, director_id_to_name))
        movies_df_loaded['producers'] = movies_df_loaded['producers'].apply(lambda x: map_ids_to_names_str(x, producer_id_to_name))
        movies_df_loaded['writers'] = movies_df_loaded['writers'].apply(lambda x: map_ids_to_names_str(x, writer_id_to_name))
        movies_df_loaded['categories'] = movies_df_loaded['categories'].apply(lambda x: map_ids_to_names_str(x, category_id_to_name))

        full_ratings_df_loaded = joblib.load(RATINGS_PATH)
        return recommender_instance, movies_df_loaded, full_ratings_df_loaded
    except FileNotFoundError as e:
        st.error(f"Error loading data files. Please ensure '{e.filename}' exists in the correct directory. Check paths: {DATA_DIR}, {JOBLIB_DIR} and dimension files.")
        st.stop() 
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}. Please check your data files and column names in the dimension CSVs.")
        st.stop()

recommender, movies_df, full_ratings_df = load_recommender_and_data()

def get_random_existing_user_id(ratings_df):
    return random.choice(ratings_df.index.tolist())

# ============== Fetch posters =================

@st.cache_data(show_spinner=False)
def get_poster_url(imdb_id):
    """Fetches movie poster URL from OMDB API."""
    formatted_id = f"tt{str(imdb_id).zfill(7)}"  # Ensure 7-digit and add 'tt' prefix
    url = f"{OMDB_BASE_URL}?i={formatted_id}&apikey={OMDB_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        poster = data.get("Poster", "")
        return poster if poster and poster != "N/A" else ""
    except Exception:
        return ""

# ============== Background image =================
        
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Force all general text elements to black for readability */
        .stApp, .stApp p, .stApp li, .stApp label, .stApp div, .stApp span, 
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp .stMarkdown, .stApp .stInfo, .stApp .stWarning {{
            color: #000000 !important; 
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3); 
        }}

        /* Adjust specific Streamlit input/display elements if needed, ensuring they remain black */
        .stApp .stSelectbox label, .stApp .stTextInput label,
        .stApp .stCheckbox label, .stApp .stRadio label {{
            color: #000000 !important; 
        }}

        /* Adjust background for sidebar to make it readable and ensure its text is black */
        .stSidebar > div:first-child {{
            background-color: rgba(255, 255, 255, 0.8); 
            color: #000000 !important; 
        }}
        /* Ensure specific elements within the sidebar also follow black text */
        .stSidebar .stRadio div[role="radiogroup"] label {{
            color: #000000 !important; 
        }}
        
        /* Make main content background opaque enough for readability */
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.7); 
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

UNSPLASH_IMAGE_URL = "https://images.unsplash.com/photo-1628417009859-1e8e39837bd8?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

set_background_image(UNSPLASH_IMAGE_URL)

# ================================================================================= MAIN APP LOGIC =================================================================================

st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",["Welcome", "New User", "Existing User"])
if page_selection == "Existing User" and 'existing_user_id' not in st.session_state:
    st.session_state['existing_user_id'] = get_random_existing_user_id(full_ratings_df)

if page_selection == "Welcome":
    st.title("🎬 Welcome to The Movie Recommender!")
    st.write("""Discover your next favorite movie with our intelligent recommendation system.
        Choose an option from the sidebar to get started:
        """
    )
    st.markdown(
        """
        ---
        **How it works:**
        * **New User:** Explore movies, pick a few you like, and get instant recommendations based on similar movie characteristics (content-based).
        * **Existing User:** See your already seen movies, get personalized recommendations based on your unique taste (collaborative filtering) & (hybrid filtering), and even predict how much you'd like an by you unrated movie.
        """
    )

# ================================================================================= PAGE: NEW USER =================================================================================

elif page_selection == "New User":
    st.title("✨ Discover Movies (New User)")
    st.write("Select up to 3 movies you like, and we'll recommend similar ones!")

    # ============== Movie filtering & selection =================
    st.subheader("Explore Movies")

    # Extract all unique filter values
    def extract_unique_values(column):
        return sorted(list(set([
            item.strip()
            for sublist in movies_df[column].apply(lambda x: x.split('|') if isinstance(x, str) else [])
            for item in sublist if item.strip()
        ])))

    all_categories = extract_unique_values("categories")
    all_actors = extract_unique_values("actors")
    all_directors = extract_unique_values("directors")
    all_producers = extract_unique_values("producers")
    all_writers = extract_unique_values("writers")

    with st.expander("Filter Options"):
        col1, col2, col3, col4, col5 = st.columns(5)
        selected_category = col1.selectbox("Category", ["All"] + all_categories, key="nu_category")
        selected_actor = col2.selectbox("Actor", ["All"] + all_actors, key="nu_actor")
        selected_director = col3.selectbox("Director", ["All"] + all_directors, key="nu_director")
        selected_producer = col4.selectbox("Producer", ["All"] + all_producers, key="nu_producer")
        selected_writer = col5.selectbox("Writer", ["All"] + all_writers, key="nu_writer")
        search_title = st.text_input("Search by Movie Title:", key="nu_title_search")

    filtered_movies = movies_df.copy()

    filters = [
        (selected_category, "categories"),
        (selected_actor, "actors"),
        (selected_director, "directors"),
        (selected_producer, "producers"),
        (selected_writer, "writers"),
    ]

    for selected_value, column in filters:
        if selected_value != "All":
            filtered_movies = filtered_movies[
                filtered_movies[column].apply(
                    lambda x: selected_value in [i.strip() for i in x.split('|')] if isinstance(x, str) else False
                )
            ]

    if search_title:
        filtered_movies = filtered_movies[filtered_movies['title'].str.contains(search_title, case=False, na=False)]

    # ============== Show filtered movies =================
    if filtered_movies.empty:
        st.info("No movies match your filters.")
    else:
        num_display = min(20, len(filtered_movies))
        display_movies = filtered_movies.sort_values(by='title').head(num_display)

        st.markdown("**Click a movie poster to select (up to 3):**")
        for chunk in [display_movies.iloc[i:i+5] for i in range(0, len(display_movies), 5)]:
            cols = st.columns(5)
            for col, (_, movie) in zip(cols, chunk.iterrows()):
                imdb_id = movie['imdb_id']
                title = movie['title']
                is_selected = title in st.session_state['selected_movies_titles']
                checkbox_key = f"select_{imdb_id}_{title}"

                if col.checkbox(title, key=checkbox_key, value=is_selected):
                    if not is_selected and len(st.session_state['selected_movies_titles']) < 3:
                        st.session_state['selected_movies_titles'].append(title)
                    elif not is_selected:
                        st.warning("Maximum of 3 movies can be selected.")
                        st.rerun()
                else:
                    if is_selected:
                        st.session_state['selected_movies_titles'].remove(title)

                col.image(get_poster_url(imdb_id), width=150, caption=title)

    # ============== Selected movie list =================
    st.subheader("Your Selected Movies:")
    selected_titles = st.session_state['selected_movies_titles']
    if selected_titles:
        st.write(", ".join(selected_titles))
        if st.button("Clear Selections", key="nu_clear_selections"):
            st.session_state['selected_movies_titles'] = []
            st.rerun()
    else:
        st.write("No movies selected yet.")

    
    # ============== Content Based Recommendations =================
    if st.button("Get Recommendations", key="nu_get_recommendations"):
        if selected_titles:
            st.subheader("Recommended Movies (Content-Based)")
            first_selected_title = selected_titles[0]
            with st.spinner(f"Finding similar movies to '{first_selected_title }'..."):
                recommendations = recommender.content_recommendations(first_selected_title, top_n=5)
                
                # Check if recommendations is a DataFrame and not empty
                if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                    # Assuming the DataFrame has a 'title' column
                    recommended_titles = recommendations['title'].drop_duplicates().tolist()
    
                    cols_per_row = 5
                    chunks = [recommended_titles[i:i + cols_per_row] for i in range(0, len(recommended_titles), cols_per_row)]
                    
                    for chunk in chunks:
                        cols = st.columns(len(chunk))
                        for i, title in enumerate(chunk):
                            with cols[i]:
                                movie_data = movies_df[movies_df['title'] == title]
                                if not movie_data.empty:
                                    movie = movie_data.iloc[0]
                                    st.image(get_poster_url(movie['imdb_id']), width=150, caption=movie['title'])
                                else:
                                    st.write(title)
                elif recommendations is None or (isinstance(recommendations, list) and recommendations == ["Title not found."]):
                    st.warning(f"No recommendations found for '{first_selected_title}'.")
                else:
                    st.warning(f"Could not retrieve recommendations for '{first_selected_title}'. Unexpected format of recommendations.")
        else:
            st.warning("Select at least one movie to get recommendations.")


# ================================================================================= PAGE: EXISTING USER =================================================================================
elif page_selection == "Existing User":
    st.title("🤝 Welcome Back! (Existing User)")
    st.markdown("Hello, VDO-Data-Scientist, explore userId ratings and get new recommendations.")

    all_user_ids = full_ratings_df.index.tolist()

    # Logic to initialize automatic userid swap
    if 'existing_user_id' not in st.session_state:
        st.session_state['existing_user_id'] = random.choice(all_user_ids)

    if st.button("Switch User"):
        current_user = st.session_state['existing_user_id']
        other_users = [u for u in all_user_ids if u != current_user]
        st.session_state['existing_user_id'] = random.choice(other_users)
        st.rerun()  # Make sure app reloads with new user
    
    # Now that the user ID is potentially updated, assign it here
    selected_user_id = st.session_state['existing_user_id']
    st.info(f"User ID: **{selected_user_id}**")
      
    # Use selected_user_id for your logic
    user_ratings = full_ratings_df.loc[selected_user_id]
    user_id_str = str(selected_user_id)


    st.header("Movies you have already watched")

    full_ratings_df.columns = full_ratings_df.columns.astype(str)

    if not full_ratings_df.empty and int(selected_user_id) in full_ratings_df.index:
        user_ratings_series = full_ratings_df.loc[int(selected_user_id)]
    
        # Convert to numeric before filtering
        user_ratings_series_numeric = pd.to_numeric(user_ratings_series, errors='coerce')
    
        # Filter for rated movies (rating > 0)
        user_rated = user_ratings_series_numeric[user_ratings_series_numeric > 0].dropna()
    
        if not user_rated.empty:
            user_rated_movies_long = user_rated.reset_index()
            user_rated_movies_long.columns = ['imdb_id', 'rating']
    
            # Ensure imdb_id columns are strings
            user_rated_movies_long['imdb_id'] = user_rated_movies_long['imdb_id'].astype(str)
            movies_df['imdb_id'] = movies_df['imdb_id'].astype(str)
    
            rated_movies_details = pd.merge(
                user_rated_movies_long,
                movies_df[['imdb_id', 'title']],
                on='imdb_id',
                how='left'
            ).dropna(subset=['title'])
            rated_movies_details = rated_movies_details.sort_values(by='title')
   
            # st.subheader("Your Rated Movies (with Posters)")
            cols_per_row = 6
            chunks = [rated_movies_details.iloc[i:i + cols_per_row] for i in range(0, len(rated_movies_details), cols_per_row)]
    
            for chunk in chunks:
                cols = st.columns(cols_per_row)
                for i, movie in enumerate(chunk.itertuples()):
                    with cols[i]:
                        st.image(get_poster_url(movie.imdb_id), width=100, caption=f"{movie.title} ({movie.rating} Stars)")
        else:
            st.info("You haven't rated any movies yet.")
    else:
        st.info("No ratings found for this user ID.")
    
        st.markdown("---")

    # ============== Hybrid Recommendations =================

    st.header("Other movies you might like -- with Hybrid Filtering")
    st.write("Get recommendations that are based on a combined approach (content-based, user-based and popularity-based filtering)")
    
    movie_for_hybrid = st.selectbox(
        "Optionally, select a movie you like to influence hybrid recommendations:",
        ["None"] + sorted(movies_df['title'].tolist()),
        key="hybrid_movie_select"
    )

    # Run hybrid recommender
    if st.button("Get Hybrid Recommendations", key="eu_get_hybrid_recs"):
        with st.spinner("Generating intelligent hybrid recommendations..."):
            selected_movie_title = movie_for_hybrid if movie_for_hybrid != "None" else None
            hybrid_recs_titles = recommender.hybrid_recommendations(user_id=user_id_str,
                                                                    movie_title=selected_movie_title,
                                                                    top_n=10,
                                                                    w_content=0.4, w_collab=0.4, w_pop=0.2,
                                                                    exclude_rated=True,
                                                                    diversity_pool=100
                                                                    )

            if hybrid_recs_titles:
                st.subheader(
                    f"Hybrid Recommendations for User {user_id_str}"
                    + (f" (influenced by '{selected_movie_title}')" if selected_movie_title else "")
                    + ":"
                )
                cols_hybrid = st.columns(5)
                for i, rec_title in enumerate(hybrid_recs_titles):
                    rec_movie_data = movies_df[movies_df['title'] == rec_title]
                    if not rec_movie_data.empty:
                        rec_movie = rec_movie_data.iloc[0]
                        with cols_hybrid[i % 5]:
                            st.image(get_poster_url(rec_movie['imdb_id']), width=150, caption=rec_movie['title'])
                    else:
                        with cols_hybrid[i % 5]:
                            st.write(rec_title)
            else:
                st.info("No hybrid recommendations available.")

    st.markdown("---")

    # ============== Look up a movies predicted rating =================
    st.header("Will you love the movie? Find out!")
    st.write("Curious about a movie? Select the movie title and see how our model predicts the rating you would give, even before watching!")

    movie_to_predict = st.selectbox(
        "",
        sorted(movies_df['title'].tolist()),
        key="predict_movie_select"
    )
    if st.button("Get personal score", key="eu_predict_rating"):
        if movie_to_predict:
            movie_imdb_id = movies_df[movies_df['title'] == movie_to_predict]['imdb_id'].iloc[0]
            try:
                prediction = recommender.algo.predict(user_id_str, str(movie_imdb_id))
                rating = round(prediction.est, 2)
    
                # Build star visuals
                stars_html = ""
                remaining = rating
                for _ in range(5):
                    if remaining >= 0.75:
                        stars_html += "<span style='color: black; font-size: 50px;'>★</span>"
                        remaining -= 1
                    elif remaining >= 0.25:
                        stars_html += "<span style='color: black; font-size: 50px;'>⯨</span>"  
                        remaining -= 0.5
                    else:
                        stars_html += "<span style='color: black; font-size: 50px;'>☆</span>"
    
                rating_display = f"<span style='font-size: 30px;'>&nbsp;({rating}/5)</span>"
    
                st.success(f"Predicted rating for **{movie_to_predict}**:")
                st.markdown(f"<div style='display: flex; align-items: center;'>{stars_html}{rating_display}</div>", unsafe_allow_html=True)
    
            except Exception as e:
                st.error(f"Could not predict rating for '{movie_to_predict}'. This might happen if the movie or user is not in the model's training data. Error: {e}")
        else:
            st.warning("Please select a movie to predict its rating.")