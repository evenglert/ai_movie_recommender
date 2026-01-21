# # AI-POWERED MOVIE RECOMMENDATION ENGINE  
# ## USER-BASED COLLABORATIVE FILTERING WITH PRECOMPUTED PEARSON SIMILARITY & STRATIFIED 5-FOLD CV  
# ### Author: Evgeniya Englert  
# ### Last update: 2026-01-21  
# 
# ---
# 
# ## Summary  
# 
# This notebook implements an **optimized User-Based Collaborative Filtering (UBCF)** pipeline using **precomputed Pearson correlation** to find similar users and generate personalized movie recommendations.  
# 
# It leverages **stratified 5-fold cross-validation** to evaluate recommendation quality robustly across users, masking test ratings in each fold instead of recomputing similarity for efficiency.  
# 
# Evaluation metrics include **Precision@K**, **Recall@K**, and **F1-Score@K**, aggregated over all folds.  
# 
# ---
# 
# ### Data Description  
# 
# The analysis uses the **MovieLens 1M dataset**, containing ~1 million ratings from 6,040 users across ~3,900 movies.  
# 
# | Dataset | File | Attributes | Role in Analysis |
# | :--- | :--- | :--- | :--- |
# | **Ratings** | `ratings.dat` | UserID, MovieID, Rating, Timestamp (dropped) | Used to construct the full User-Item Matrix and for stratified 5-fold CV. |
# | **Movies** | `movies.dat` | MovieID, Title, Genres | Used to enrich recommendations with titles and genres. |
# 
# ---
# 
# ### Purpose  
# 
# The system implements **user-based collaborative filtering with precomputed Pearson similarity** and evaluates it with **stratified 5-fold cross-validation**:
# 
# 1. **Compute full user-user similarity** using Pearson correlation once.  
# 2. **Cache top-N neighbors** for each user for faster recommendation generation.  
# 3. **Generate Top-N recommendations** based on neighbors’ ratings of unrated movies.  
# 4. **Assess recommendation quality** using Precision@K, Recall@K, and F1-Score@K across 5 stratified folds.  
# 
# ---
# 
# ### Key Steps and Methodology  
# 
# 1. **Data Loading and Preparation**
#    - Load `ratings.dat` and `movies.dat`.
#    - Drop timestamps.
#    - Construct the **full User-Item matrix**.
# 
# 2. **User Similarity Precomputation**
#    - Compute **Pearson correlation** between all users.
#    - Cache **top-K nearest neighbors** for each user to avoid recomputation during CV.
# 
# 3. **Recommendation Generation**
#    - Predict ratings for unrated movies using **weighted averages of neighbors’ ratings**.
#    - Select **Top-N movies** with the highest predicted ratings per user.
# 
# 4. **Stratified 5-Fold Cross-Validation**
#    - Split ratings into 5 folds **stratified by user**.
#    - Mask test ratings in each fold without recalculating similarity.
#    - Generate recommendations and compute ranking metrics per fold.
#    - Aggregate results across folds.
# 
# 5. **Evaluation**
#    - Metrics: **Precision@K**, **Recall@K**, **F1@K**, number of evaluated users.
#    - Report **mean metrics** across folds.
# 
# ---
# 
# ### Ranking Evaluation  
# 
# - **Relevant movies:** Rating ≥ 4.0 (`RELEVANCE_THRESHOLD`)  
# - **Top-K cutoff:** `TOP_N_RECOMMENDATIONS` = 10  
# 
# | Metric | Description |
# | ------ | ----------- |
# | Precision@K | Fraction of top-K recommended movies that are relevant. |
# | Recall@K | Fraction of relevant movies retrieved in top-K recommendations. |
# | F1-Score@K | Harmonic mean of Precision@K and Recall@K. |
# 
# ---
# 
# ### Example Cross-Validation Results  
# 
# **User-Based Collaborative Filtering with Precomputed Pearson Similarity (5-Fold CV)**
# 
# - Top-N Cutoff (K): 10  
# - Number of Folds: 5  
# 
# ---
# 
# ### High-Level Architecture: Optimized Stratified 5-Fold UBCF Workflow  
#               MovieLens Ratings Data
#                      │
#                      ▼
#              Load & Prepare Data
#      ─────────────────────────────
#      - Drop timestamp
#      - Construct full User-Item matrix
#                      │
#                      ▼
#         Stratified 5-Fold Cross-Validation
#      ─────────────────────────────────────
#      For each fold:
#        1. Mask test ratings (do not recompute similarity)
#        2. Use precomputed Pearson similarity
#        3. Identify top-N neighbors per user
#        4. Identify top-N neighbors
#        5. Predict ratings for unrated movies
#        6. Select Top-N recommendations
#        7. Compute Precision@K, Recall@K, F1@K
#                      │
#                      ▼
#            Aggregate Metrics Across Folds
#      ─────────────────────────────────────
#      - Mean & Std for Precision@K
#      - Mean & Std for Recall@K
#      - Mean & Std for F1-Score@K
#      - Fold execution times
#                      │
#                      ▼
#       Output: Personalized Top-N Recommendations
# 	  
# 
# ---
# 
# ### Configuration Parameters
# 
# ```python
# RATING_FILE_PATH = 'ratings.dat'       
# MOVIES_FILE_PATH = 'movies.dat'        
# TOP_N_RECOMMENDATIONS = 10             
# RELEVANCE_THRESHOLD = 4.0              
# NUM_NEIGHBORS = 10                     
# N_SPLITS = 5                            
# 
# =============================== BEGIN ===============================
# ===================================================================================================================
# Optimized User-Based Collaborative Filtering with Precomputed Similarity using Pearson Correlation & Stratified CV
# ===================================================================================================================
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from pandas.errors import SettingWithCopyWarning
import warnings
import time

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

# ----------------------- Configuration -----------------------
RATING_FILE_PATH = 'ratings.dat'       # Path to ratings dataset
MOVIES_FILE_PATH = 'movies.dat'        # Path to movies metadata
TOP_N_RECOMMENDATIONS = 10             # Number of top recommendations to evaluate per user
RELEVANCE_THRESHOLD = 4.0              # Ratings >= this are considered relevant
NUM_NEIGHBORS = 10                     # Number of neighbors to consider in UBCF
N_SPLITS = 5                            # Number of folds in Stratified K-Fold CV

# ----------------------- Evaluation Metrics -----------------------
def evaluate_ranking_metrics(predictions_df, test_ratings_df, relevance_threshold, K):
    """
    Evaluate Precision@K, Recall@K, F1@K based on top-K recommendations per user.
    Only considers items with ratings >= relevance_threshold as relevant.
    """
    # Map: UserID -> set of relevant MovieIDs in test set
    relevant_test_items = test_ratings_df[test_ratings_df['Rating'] >= relevance_threshold] \
        .groupby('UserID')['MovieID'].apply(set).to_dict()

    # Rank predicted ratings per user
    predictions_df = predictions_df.copy()
    predictions_df['rank'] = predictions_df.groupby('UserID')['Predicted_Rating'] \
        .rank(method='first', ascending=False)

    # Keep only top-K predictions per user
    top_k_predictions = predictions_df[predictions_df['rank'] <= K]

    precisions, recalls, f1s = [], [], []

    # Evaluate metrics per user
    for user_id, group in top_k_predictions.groupby('UserID'):
        relevant = relevant_test_items.get(user_id)
        if not relevant:
            continue  # Skip users with no relevant items in test set

        recommended = group['MovieID'].tolist()
        y_true = [1 if movie_id in relevant else 0 for movie_id in recommended]  # Ground truth
        y_pred = [1] * len(y_true)  # All recommended items are predicted positive

        # Compute precision, recall, F1
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    # Return mean metrics and number of evaluated users
    return {
        f'Precision@{K}': np.mean(precisions) if precisions else 0.0,
        f'Recall@{K}': np.mean(recalls) if recalls else 0.0,
        f'F1-Score@{K}': np.mean(f1s) if f1s else 0.0,
        'Evaluated Users': len(precisions)
    }

# ----------------------- Data Loading -----------------------
def load_data():
    """
    Load ratings and movies datasets into pandas DataFrames.
    """
    print("Loading data...")
    rating_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    movie_cols = ['MovieID', 'Title', 'Genres']

    # Load ratings data and drop timestamp
    ratings_df = pd.read_csv(RATING_FILE_PATH, sep='::', engine='python',
                             names=rating_cols, encoding='latin-1').drop('Timestamp', axis=1)
    # Load movies metadata
    movies_df = pd.read_csv(MOVIES_FILE_PATH, sep='::', engine='python',
                            names=movie_cols, encoding='latin-1')

    print(f"Total Ratings: {len(ratings_df)}")
    return ratings_df, movies_df

# ----------------------- Similarity Precomputation -----------------------
def precompute_similarity_full(user_item_matrix):
    """
    Compute user-user Pearson correlation similarity matrix once for all users.
    Also caches top-N neighbors per user for faster recommendation generation.
    """
    print("Precomputing full user-user Pearson similarity matrix...")
    similarity_matrix = user_item_matrix.T.corr(method='pearson').fillna(0)

    # Cache top-N neighbors per user to avoid repeated computation
    neighbors_dict = {
        user_id: similarity_matrix[user_id].drop(user_id, errors='ignore').nlargest(NUM_NEIGHBORS)
        for user_id in user_item_matrix.index
    }
    return similarity_matrix, neighbors_dict

# ----------------------- Recommendation Generation -----------------------
def generate_recommendations(user_item_matrix, neighbors_dict):
    """
    Generate user-based CF recommendations using precomputed neighbors.
    Predictions are weighted averages of neighbor ratings based on similarity.
    """
    recommendations = []

    # Loop over each user
    for user_id in user_item_matrix.index:
        user_ratings = user_item_matrix.loc[user_id]

        # Identify unrated movies for this user
        unrated_movies = user_ratings[user_ratings.isna()].index
        top_neighbors = neighbors_dict[user_id]

        # Skip if no neighbors or no unrated movies
        if top_neighbors.empty or unrated_movies.empty:
            continue

        neighbors_ratings = user_item_matrix.loc[top_neighbors.index]
        sim_scores = top_neighbors.values

        # Predict rating for each unrated movie
        for movie_id in unrated_movies:
            ratings = neighbors_ratings[movie_id]
            mask = ratings.notna()  # Only consider neighbors who rated the movie
            if not mask.any():
                continue

            valid_ratings = ratings[mask].values
            valid_sim = sim_scores[mask.values]

            sim_sum = np.sum(np.abs(valid_sim))
            if sim_sum == 0:
                continue  # Avoid division by zero

            # Weighted average prediction
            pred_rating = np.dot(valid_sim, valid_ratings) / sim_sum
            recommendations.append((user_id, movie_id, pred_rating))

    return pd.DataFrame(recommendations, columns=['UserID', 'MovieID', 'Predicted_Rating'])

# ----------------------- Optimized Stratified Cross-Validation -----------------------
def run_cross_validation_optimized(ratings_df, movies_df):
    """
    Perform Stratified K-Fold cross-validation using precomputed similarity.
    Test ratings are masked in each fold instead of recomputing similarity.
    """
    print(f"\n--- Running {N_SPLITS}-Fold Stratified Cross-Validation (Optimized) ---")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    precision_list, recall_list, f1_list = [], [], []

    # Stratify by UserID to maintain user distribution across folds
    X = ratings_df.drop('Rating', axis=1)
    y = ratings_df['UserID']

    # Full user-item matrix for all ratings
    full_user_item_matrix = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')

    # Precompute similarity and neighbors once
    _, neighbors_dict = precompute_similarity_full(full_user_item_matrix)

    fold_num = 1
    for train_index, test_index in skf.split(X, y):
        print(f"\n--- Fold {fold_num} ---")
        train_df = ratings_df.iloc[train_index]
        test_df = ratings_df.iloc[test_index]

        # Copy full matrix and mask test ratings for this fold
        user_item_matrix = full_user_item_matrix.copy()
        test_ratings = test_df.set_index(['UserID', 'MovieID'])
        for (u, m) in test_ratings.index:
            if u in user_item_matrix.index and m in user_item_matrix.columns:
                user_item_matrix.at[u, m] = np.nan  # Treat as unknown during training

        # Generate recommendations using precomputed neighbors
        recommendations_df = generate_recommendations(user_item_matrix, neighbors_dict)

        # Evaluate metrics for this fold
        metrics = evaluate_ranking_metrics(recommendations_df, test_df, RELEVANCE_THRESHOLD, TOP_N_RECOMMENDATIONS)
        print(f"Precision@{TOP_N_RECOMMENDATIONS}: {metrics[f'Precision@{TOP_N_RECOMMENDATIONS}']:.4f}")
        print(f"Recall@{TOP_N_RECOMMENDATIONS}:    {metrics[f'Recall@{TOP_N_RECOMMENDATIONS}']:.4f}")
        print(f"F1-Score@{TOP_N_RECOMMENDATIONS}:  {metrics[f'F1-Score@{TOP_N_RECOMMENDATIONS}']:.4f}")

        precision_list.append(metrics[f'Precision@{TOP_N_RECOMMENDATIONS}'])
        recall_list.append(metrics[f'Recall@{TOP_N_RECOMMENDATIONS}'])
        f1_list.append(metrics[f'F1-Score@{TOP_N_RECOMMENDATIONS}'])

        fold_num += 1

    # Print overall cross-validation results
    print("\n--- Cross-Validation Results (Optimized) ---")
    print(f"Average Precision@{TOP_N_RECOMMENDATIONS}: {np.mean(precision_list):.4f}")
    print(f"Average Recall@{TOP_N_RECOMMENDATIONS}:    {np.mean(recall_list):.4f}")
    print(f"Average F1-Score@{TOP_N_RECOMMENDATIONS}:  {np.mean(f1_list):.4f}")

# ----------------------- Main Execution -----------------------
def main():
    start_time = time.time()
    try:
        # Load datasets
        ratings_df, movies_df = load_data()

        # Run optimized cross-validation
        run_cross_validation_optimized(ratings_df, movies_df)

    except FileNotFoundError:
        print("\nFATAL ERROR: Ensure 'ratings.dat' and 'movies.dat' exist in the working directory.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

    end_time = time.time()
    print(f"\nTotal Execution Time: {end_time - start_time:.2f} sec")

# Entry point
if __name__ == "__main__":
    main()

# =============================== END ===============================