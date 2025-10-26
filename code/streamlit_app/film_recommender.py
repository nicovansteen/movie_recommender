import os
import joblib
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

class FilmRecommender:

    # ================================================== INIT ==================================================
    def __init__(self, data_dir, joblib_dir):
        self.data_dir = data_dir
        self.joblib_dir = joblib_dir
        self.movies = joblib.load(os.path.join(self.joblib_dir, 'df_final.joblib'))
        
        self.ratings = joblib.load(os.path.join(self.joblib_dir, 'df_final_matrix.joblib'))
        # self.df_ratings = self.ratings.apply(pd.to_numeric, errors='coerce').fillna(0)   
        # if not isinstance(self.ratings, pd.DataFrame):
        #     print("WARNING: self.ratings is not a DataFrame, attempting conversion.")
        #     try:
        #         self.ratings = pd.DataFrame(self.ratings.toarray())
        #     except AttributeError: 
        #         print("ERROR: self.ratings is not a DataFrame and not a sparse matrix.")                
        #         raise TypeError("Expected self.ratings to be a DataFrame or sparse matrix.")

        if isinstance(self.ratings, pd.DataFrame):
            self.df_ratings = self.ratings.apply(pd.to_numeric, errors='coerce').fillna(0)
        else:
            print("WARNING: self.ratings is not a DataFrame, attempting conversion.")
            try:
                self.ratings = pd.DataFrame(self.ratings.toarray())
                self.df_ratings = self.ratings.apply(pd.to_numeric, errors='coerce').fillna(0)
            except AttributeError:
                print("ERROR: self.ratings is not a DataFrame and not a sparse matrix.")                
                raise TypeError("Expected self.ratings to be a DataFrame or sparse matrix.")
        
        
        # self.trainset = None
        self._prepare_surprise_data()
        self.fit_surprise_model()
        self._compute_popularity()
        self._prepare_content_data()
        self._build_combined_features()
        self._build_tfidf()
        
        self.title_to_index = pd.Series(self.movies.index, index=self.movies['title'])

    # ================================================== COMBINE FEAUTURES ==================================================    
    def _build_combined_features(self):
        feature_cols = ['actors', 'directors', 'producers', 'writers', 'categories']

        # Convert lists to strings in these columns, keep strings as is
        for col in feature_cols:
            self.movies[col] = self.movies[col].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else (str(x) if pd.notnull(x) else '')
            )
            
        # Now join the columns with '|'
        self.movies['combined_features'] = self.movies[feature_cols].fillna('').agg('|'.join, axis=1)

    # ========================================= MAKE TABLE WITH WEIGHTS ON FEAUTURES ==============================================
    def _build_tfidf(self):
        self.tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['combined_features'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    # ========================================= SURPRISE MODEL ==============================================
    def _prepare_surprise_data(self):
        if self.df_ratings.empty:
            raise ValueError("The input DataFrame for ratings (self.df_ratings) is empty.")

        df_long = self.df_ratings.reset_index().melt(id_vars=['user_ids'], var_name='itemId', value_name='rating')
        df_long = df_long.rename(columns={'user_ids': 'userId'}) # Rename the 'user_ids' column to 'userId'
    
        # Filter out unrated movies (where rating is 0, assuming 0 means not rated)
        self.df_ratings_long = df_long[df_long['rating'] > 0].copy()
    
        # Ensure userId and itemId are strings, as Surprise handles them as 'raw' IDs
        self.df_ratings_long['userId'] = self.df_ratings_long['userId'].astype(str)
        self.df_ratings_long['itemId'] = self.df_ratings_long['itemId'].astype(str)
    
        reader = Reader(rating_scale=(0, 5)) # Assuming ratings are 0-5
        self.data = Dataset.load_from_df(self.df_ratings_long[['userId', 'itemId', 'rating']], reader)
        self.trainset = self.data.build_full_trainset()
    
    def fit_surprise_model(self, test_size: float = 0.2, random_state: int = 42,
                               n_factors: int = 50, n_epochs: int = 20, resplit: bool = False):
        
        if resplit or not hasattr(self, 'trainset'):
            self._prepare_surprise_data(test_size=test_size, random_state=random_state)
        
        self.algo = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=random_state)
        self.algo.fit(self.trainset)
        
    def _compute_popularity(self):
           
        if self.df_ratings_long.empty:
            print("ERROR: self.df_ratings_long is empty. Cannot compute popularity.")
            # Handle this case, perhaps by returning or raising an error
            self.movie_popularity = pd.Series(dtype=float) # Initialize as empty Series
            return
    
        # Group by 'itemId' (the actual column name) and calculate the mean rating
        grouped_ratings = self.df_ratings_long.groupby('itemId')['rating'].mean()
    
        # Calculate the count of ratings for each movie
        rating_counts = self.df_ratings_long.groupby('itemId')['rating'].count()
    
        # For simplicity, if we just want popularity based on the ratings data:
        popularity_df = pd.DataFrame({
            'mean_rating': grouped_ratings,
            'rating_count': rating_counts
        }).reset_index() # Resets itemId as a column
    
        # Ensure itemId is string for consistent merges/lookups
        popularity_df['itemId'] = popularity_df['itemId'].astype(str)
    
        self.movie_popularity = grouped_ratings.sort_values(ascending=False)
        self.avg_ratings = self.df_ratings_long.groupby('itemId')['rating'].mean()
        self.movie_rating_counts = self.df_ratings_long.groupby('itemId')['rating'].count()

        if not self.avg_ratings.empty:
            self.popularity_max = self.avg_ratings.max()
        else:
            self.popularity_max = 1.0  # Default to 1.0 to avoid division by zero if no ratings
      
    def content_recommendations(self, title, top_n=10):
        """
        Generates content-based recommendations for a given movie title.
        Args:
            title (str): The title of the movie for which to find recommendations.
            top_n (int): The number of top recommendations to return.
            
        Returns:
            pandas.DataFrame: DataFrame containing recommended movies.
        """
        if self.movies.empty or self.content_similarity_matrix is None:
            print("ERROR: Content data not prepared. Cannot provide content recommendations.")
            return pd.DataFrame()

        # Find the index of the movie that matches the title
        # Case-insensitive search might be good here, and handle multiple matches/no matches
        movie_row = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]

        if movie_row.empty:
            print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
        
        # If multiple movies match (e.g., "Cinderella"), pick the first one or ask user to disambiguate
        movie_idx = movie_row.index[0]
        
        # Get the similarity scores for this movie
        # content_similarity_matrix is indexed by the DataFrame's original integer index
        similar_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
        
        # Sort movies by similarity score in descending order
        # Exclude the movie itself (similarity score will be 1.0)
        sorted_similar_movies = sorted(similar_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top_n similar movies (excluding the first one which is the movie itself)
        recommended_movie_indices = [i[0] for i in sorted_similar_movies[1:top_n+1]]
        
        # Retrieve the movie details for the recommendations
        recommended_movies_df = self.movies.iloc[recommended_movie_indices].copy()
        recommended_movies_df['similarity_score'] = [i[1] for i in sorted_similar_movies[1:top_n+1]]

        # Drop the temporary 'categories_processed' column if you added it
        if 'categories_processed' in recommended_movies_df.columns:
            recommended_movies_df = recommended_movies_df.drop(columns=['categories_processed'])

        return recommended_movies_df[['title', 'categories', 'similarity_score']] # You can adjust columns

    def _prepare_content_data(self):
        """
        Prepares data for content-based recommendations.
        Uses movie categories to create a TF-IDF matrix and cosine similarity.
        """        

        processed_categories = []
        for item in self.movies['categories'].fillna('').tolist():
            if isinstance(item, list):                
                processed_categories.append(' '.join(item))
            elif isinstance(item, str):
                processed_categories.append(' '.join(item.split('|')))
            else:                
                processed_categories.append('')

        self.movies['categories_processed'] = processed_categories

        self.tfidf_vectorizer = TfidfVectorizer()  # No custom tokenizer or lowercase=False needed

        # Fit and transform the category data to create the TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies['categories_processed'])

        # Compute cosine similarity between movies based on their category TF-IDF vectors
        self.content_similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    # ================================================== USER BASED RECOMMENDATIONS =================================================   
    
    def svd_recommendations(self, user_id, top_n: int = 10):        
        if not hasattr(self, 'algo') or self.algo is None:
            raise ValueError("SVD model must be trained before making recommendations.")
        if self.trainset is None:             
             raise ValueError("Surprise trainset is not prepared. Ensure fit_surprise_model() was called correctly.")

        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return []
        
        rated_movie_ids = self.df_ratings_long[self.df_ratings_long['userId'] == user_id]['itemId'].tolist()        
        if len(rated_movie_ids) == 0:
            return []

        all_item_ids = self.df_ratings_long['itemId'].unique()
        movies_to_predict = [imdb_id for imdb_id in all_item_ids if imdb_id not in rated_movie_ids]

        predictions = []

        for itemId in movies_to_predict:            
            if itemId not in rated_movie_ids:
                try:
                    pred = self.algo.predict(str(user_id), str(itemId))
                    predictions.append(pred)
                except Exception as e:
                    print(f"WARNING: Could not predict for user {user_id} and movie {itemId}: {e}")
        
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get top N movie titles
        top_n_predictions = predictions[:top_n]
        top_n_movie_ids = [pred.iid for pred in top_n_predictions]

        top_n_titles = []
        for movie_id in top_n_movie_ids:
            movie_row = self.movies[self.movies['imdb_id'].astype(str) == str(movie_id)]
            if not movie_row.empty:
                top_n_titles.append(movie_row['title'].iloc[0])
            else:
                print(f"WARNING: Movie ID {movie_id} not found in self.movies DataFrame.")

        return top_n_titles
        
    # ================================================== HYBRID RECOMMENDATIONS =================================================        
    def hybrid_recommendations(self, user_id, movie_title=None, top_n=10,w_content=0.4,
                               w_collab=0.4, w_pop=0.2,exclude_rated=True, diversity_pool=100):
        user_str = str(user_id)

        candidates = set()
        content_candidates = []

        # Get movies the user has already rated
        all_movies = set(self.movies['imdb_id'])
        seen = set(self.df_ratings_long[self.df_ratings_long['userId'] == user_str]['itemId']) if exclude_rated else set()
        unseen = all_movies - seen if exclude_rated else all_movies

        # ===== Content-based scores (influenced by a selected movie) =====
        
        if movie_title and movie_title in self.title_to_index:
            idx = self.title_to_index[movie_title]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            content_candidates = [(self.movies['imdb_id'].iloc[i], score)
                                  for i, score in sim_scores[1:300]  # Slightly deeper pool
                                  if self.movies['imdb_id'].iloc[i] in unseen]
            candidates.update([c[0] for c in content_candidates])

        # ===== Collaborative Filtering (SVD Predictions) =====
        
        collab_preds = [(mid, self.algo.predict(user_str, mid).est) for mid in unseen]
        collab_preds.sort(key=lambda x: x[1], reverse=True)
        collab_candidates = collab_preds[:300]  # Slightly deeper pool
        candidates.update([c[0] for c in collab_candidates])

        collab_min = min(score for _, score in collab_candidates) if collab_candidates else 0
        collab_max = max(score for _, score in collab_candidates) if collab_candidates else 1

        # ===== Score Aggregation =====
        content_dict = dict(content_candidates)
        collab_dict = dict(collab_candidates)
        combined_scores = []

        for mid in candidates:
            c_score = content_dict.get(mid, 0)

            # Normalize collaborative score
            collab_raw = collab_dict.get(mid, 0)
            collab_score = ((collab_raw - collab_min) / (collab_max - collab_min)
                            if collab_max > collab_min else 0)

            # Normalize popularity score
            p_score = self.avg_ratings.get(mid, 0) / self.popularity_max if self.popularity_max > 0 else 0

            # Weighted hybrid score
            hybrid_score = w_content * c_score + w_collab * collab_score + w_pop * p_score
            combined_scores.append((mid, hybrid_score))

        # Sort by hybrid score
        combined_scores.sort(key=lambda x: x[1], reverse=True)

        # Diversity: sample from top N pool
        top_pool = combined_scores[:diversity_pool]
        final_recs = random.sample(top_pool, min(top_n, len(top_pool)))

        recommended_ids = [mid for mid, _ in final_recs]
        return self.movies[self.movies['imdb_id'].isin(recommended_ids)]['title'].tolist()


if __name__ == '__main__':
    DATA_DIR = os.path.join('..', '..', 'data', 'cleaned')
    JOBLIB_DIR = os.path.join('..', '..', 'data', 'cleaned', 'joblib_dataframes')

    recommender = FilmRecommender(DATA_DIR, JOBLIB_DIR)

    print("\nContent-based recommendation based on movie:")
    print(recommender.content_recommendations(title=None , top_n=5))
    print(50 * '*')

    print("\nSurprise collaborative recommendations based on user:")
    print(recommender.svd_recommendations(user_id='7', top_n=5))
    print(50 * '*')

    print(recommender.hybrid_recommendations(user_id='7', movie_title=None,
                                             top_n=5, w_content=0.4,
                                             w_collab=0.4, w_pop=0.2,
                                             exclude_rated=True, diversity_pool=100))
    print(50 * '*')
    print("\nOverall hybrid recommendations for one user:")
    print(recommender.hybrid_recommendations(user_id='7', movie_title=None,
                                             top_n=5, w_content=0.4,
                                             w_collab=0.4, w_pop=0.2,
                                             exclude_rated=True, diversity_pool=100))
    print(50 * '*')
