# # Hybrid User-Based Movie Recommendation System
# ## Using Ratings and Demographic Profiles
# **Author:** Evgeniya Englert  
# **Last update:** 2026-01-27
# ---
# ## Summary
# This code implements a **hybrid user-based collaborative filtering** system for movie recommendation.
# 
# Key features:
# 1. **User-Based Collaborative Filtering**
#    - Uses **Pearson correlation** on user ratings
#    - Predicts ratings via weighted neighbor ratings
# 2. **Profile-Based Similarity**
#    - Converts **user demographic data** (Gender, Age, Occupation) into numeric features
#    - Computes **Pearson correlation** similarity between user profiles
# 3. **Hybrid Similarity**
#    - Combines rating-based and profile-based similarities with configurable weights
# 4. **Evaluation**
#    - Ranking metrics: Precision@K, Recall@K, F1-Score@K
#    - Relevant items: Rating ≥ 4.0
# 
# Additional steps:
# - Stratified train/test split by user
# - User–Item rating matrix construction
# - Sample personalized recommendations
# ---

# ## Dataset Description
# We use the **MovieLens 1M dataset**, which includes:
# | Dataset | File | Description |
# |--------|------|------------|
# | Ratings | ratings.dat | User ratings for movies |
# | Movies  | movies.dat | Movie titles and genres |
# | Users   | users.dat | User demographic data (Gender, Age, Occupation, ZipCode) |

# ## Configuration & Hyperparameters
# - **TOP_N_RECOMMENDATIONS:** Number of movies recommended per user (default 10)  
# - **TEST_SIZE:** Fraction of ratings held out for testing (default 0.2)  
# - **RELEVANCE_THRESHOLD:** Rating threshold for relevant items (default 4.0)  
# - **NUM_NEIGHBORS:** Number of nearest neighbors used for predictions (default 10)  
# - **WEIGHT_PROFILE:** Weight for profile similarity in hybrid score (default 0.3)  
# - **WEIGHT_RATING:** Weight for rating similarity in hybrid score (default 0.7)  

# ## Notes
# - Similarity matrices are computed **once** and reused  
# - Evaluation uses **single stratified split** (no k-fold cross-validation)  
# - Predictions ignore items without any neighbor ratings  
# - Metrics computed only for users with at least one relevant test item  
# - Random sampling is used to display example recommendations  

# ## Execution & Evaluation
# After running the code:
# - Similarity matrices are printed (first 5x5 entries)  
# - Ranking metrics (Precision@K, Recall@K, F1@K) are displayed  
# - Sample Top-N recommendations are shown for a randomly selected user  
# - Total execution time is printed  

# Workflow Overview
# -----------------
#                 MovieLens Datasets
#           (ratings.dat, movies.dat, users.dat)
#                            │
#                            ▼
#                     Load Data
#         ─────────────────────────────────
#         - Read ratings, movies, users
#         - Drop rating timestamp column
#                            │
#                            ▼
#               Stratified Train/Test Split
#         ─────────────────────────────────
#         - Split ratings by UserID
#         - Hold out TEST_SIZE for evaluation
#                            │
#                            ▼
#             Build User–Item Rating Matrix
#         ─────────────────────────────────
#         - Use training data only
#         - Rows: Users
#         - Columns: Movies
#         - Values: Ratings
#                            │
#                            ▼
#             Preprocess User Profiles
#         ─────────────────────────────────
#         - Encode Gender as binary
#         - Keep Age as numeric
#         - One-hot encode Occupation
#         - Drop original demographic columns
#                            │
#                            ▼
#          Compute Similarities
#         ─────────────────────────────────
#         - Rating-based similarity (Pearson correlation)
#         - Profile-based similarity (Pearson correlation)
#                            │
#                            ▼
#            Combine Similarities (Hybrid)
#         ─────────────────────────────────
#         - Weighted sum: rating-based * w_rating + profile-based * w_profile
#                            │
#                            ▼
#            User-Based CF Recommendations
#         ─────────────────────────────────
#         For each user:
#           1. Identify top-N similar users
#           2. Find unrated movies
#           3. Predict ratings using weighted neighbor ratings
#           4. Store predicted scores
#                            │
#                            ▼
#             Top-K Recommendation Selection
#         ─────────────────────────────────
#         - Rank predicted movies per user
#         - Select Top-K recommendations
#                            │
#                            ▼
#                 Ranking Evaluation
#         ─────────────────────────────────
#         - Compare Top-K lists with test set
#         - Relevant items: Rating ≥ 4.0
#         - Compute:
#             • Precision@K
#             • Recall@K
#             • F1-Score@K
#                            │
#                            ▼
#       Output: Personalized Top-N Movies

# Hybrid User-Based Collaborative Filtering for Movie Recommendation System Using Ratings and Demographic Profiles

# =============================== BEGIN ===============================

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------


import time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
RATING_FILE_PATH = BASE_DIR / "ratings.dat"
MOVIES_FILE_PATH = BASE_DIR / "movies.dat"
USERS_FILE_PATH = BASE_DIR / "users.dat"

TOP_N_RECOMMENDATIONS = 10
TEST_SIZE = 0.2
RELEVANCE_THRESHOLD = 4.0
NUM_NEIGHBORS = 10
WEIGHT_PROFILE = 0.3  # weight given to profile similarity
WEIGHT_RATING = 0.7   # weight given to rating-based similarity

# ----------------------------------------------------------------------
def evaluate_ranking_metrics(predictions_df, test_ratings_df, relevance_threshold, K):
    relevant_test = test_ratings_df[test_ratings_df['Rating'] >= relevance_threshold] \
        .groupby('UserID')['MovieID'].apply(set).to_dict()

    recs = defaultdict(list)
    for _, row in predictions_df.iterrows():
        recs[row['UserID']].append(row['MovieID'])

    precisions, recalls, f1s = [], [], []
    evaluated = 0

    for user_id, relevant in relevant_test.items():
        if user_id not in recs:
            continue
        preds = recs[user_id][:K]
        true_labels = [1 if m in relevant else 0 for m in preds]
        pred_labels = [1] * len(preds)
        if len(true_labels) == 0:
            continue
        precisions.append(precision_score(true_labels, pred_labels, zero_division=0))
        recalls.append(recall_score(true_labels, pred_labels, zero_division=0))
        f1s.append(f1_score(true_labels, pred_labels, zero_division=0))
        evaluated += 1

    return {
        f'Precision@{K}': np.mean(precisions) if precisions else 0,
        f'Recall@{K}': np.mean(recalls) if recalls else 0,
        f'F1-Score@{K}': np.mean(f1s) if f1s else 0,
        'Evaluated Users': evaluated
    }

# ----------------------------------------------------------------------
def load_data():
    print("Loading rating, movie, and user data...")

    rating_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    movie_cols = ['MovieID', 'Title', 'Genres']
    user_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode']

    ratings = pd.read_csv(RATING_FILE_PATH, sep='::', names=rating_cols, engine='python', encoding='latin-1') \
        .drop(columns='Timestamp')
    movies = pd.read_csv(MOVIES_FILE_PATH, sep='::', names=movie_cols, engine='python', encoding='latin-1')
    users = pd.read_csv(USERS_FILE_PATH, sep='::', names=user_cols, engine='python', encoding='latin-1')

    return ratings, movies, users

# ----------------------------------------------------------------------
def preprocess_user_profiles(users_df):
    """
    Converts categorical attributes (Gender, Age, Occupation) into numeric features.
    Returns a DataFrame indexed by UserID with profile feature vectors.
    """
    df = users_df.set_index('UserID').copy()

    # Gender: M / F → binary
    df['Gender_feat'] = (df['Gender'] == 'M').astype(int)

    # Age: treat as numeric directly, or encode via one-hot
    # Here we just use the numeric "Age" code as is.
    df['Age_feat'] = df['Age'].astype(float)

    # Occupation: one-hot encoding
    occ_onehot = pd.get_dummies(df['Occupation'], prefix='Occ')
    df = pd.concat([df, occ_onehot], axis=1)

    # Drop original columns
    df = df.drop(columns=['Gender', 'Age', 'Occupation', 'ZipCode'], errors='ignore')

    return df

# ----------------------------------------------------------------------
def compute_rating_similarity(user_item_matrix):
    """
    Compute Pearson correlation-based user-user similarity from the rating matrix.
    """
    # Use DataFrame.corr which computes pairwise Pearson correlation
    # Users with no overlapping ratings will give NaN → fill with 0
    sim = user_item_matrix.T.corr(method='pearson')
    sim = sim.fillna(0)
    return sim

# ----------------------------------------------------------------------
def compute_profile_similarity(profile_df):
    """
    Compute similarity between users based on profile vectors (e.g. cosine or correlation).
    Here we use Pearson correlation on the profile features.
    """
    sim = profile_df.T.corr(method='pearson')
    sim = sim.fillna(0)
    return sim

# ----------------------------------------------------------------------
def combine_similarities(rating_sim, profile_sim, w_rating=0.7, w_profile=0.3):
    """
    Weighted sum of two similarity matrices (same index/columns).
    """
    return w_rating * rating_sim + w_profile * profile_sim

# ----------------------------------------------------------------------
def generate_user_based_recommendations(ratings_train, user_item_matrix, user_similarity):
    """
    Use user_similarity to recommend items. Weighted sum of neighbor ratings.
    """
    all_recs = []
    for user in user_item_matrix.index:
        # neighbors sorted descending
        sim_scores = user_similarity[user].drop(labels=[user], errors='ignore')
        top_neighbors = sim_scores.sort_values(ascending=False).head(NUM_NEIGHBORS).index

        # movies this user hasn’t rated
        unrated = user_item_matrix.columns[user_item_matrix.loc[user].isna()]

        for m in unrated:
            neigh_ratings = user_item_matrix.loc[top_neighbors, m].dropna()
            if neigh_ratings.empty:
                continue
            sim_sub = user_similarity.loc[user, neigh_ratings.index]
            weights = sim_sub.values
            ratings = neigh_ratings.values
            if np.sum(np.abs(weights)) == 0:
                continue
            pred = np.dot(weights, ratings) / np.sum(np.abs(weights))
            all_recs.append({'UserID': user, 'MovieID': m, 'Predicted_Rating': pred})

    recs_df = pd.DataFrame(all_recs)
    return recs_df

# ----------------------------------------------------------------------
def main():
    start_time = time.time()
    # 1. Load data
    ratings, movies, users = load_data()

    # 2. Split into train / test
    train_ratings, test_ratings = train_test_split(ratings, test_size=TEST_SIZE,
                                                    stratify=ratings['UserID'], random_state=42)

    # 3. Build user-item matrix from train
    user_item = train_ratings.pivot(index='UserID', columns='MovieID', values='Rating')

    # 4. Process user profiles
    user_profiles = preprocess_user_profiles(users)

    # 5. Compute similarities
    rating_sim = compute_rating_similarity(user_item)
    profile_sim = compute_profile_similarity(user_profiles)

    print("Sample of rating-based similarity matrix:")
    print(rating_sim.iloc[:5, :5])
    print("Sample of profile-based similarity matrix:")
    print(profile_sim.iloc[:5, :5])

    # 6. Combine similarities
    hybrid_user_sim = combine_similarities(rating_sim, profile_sim,
                                           w_rating=WEIGHT_RATING, w_profile=WEIGHT_PROFILE)

    # 7. Generate recommendations
    recs = generate_user_based_recommendations(train_ratings, user_item, hybrid_user_sim)

    # 8. Evaluate
    print("\n--- Ranking Quality Evaluation ---")
    metrics = evaluate_ranking_metrics(recs, test_ratings, RELEVANCE_THRESHOLD, TOP_N_RECOMMENDATIONS)
    print(f"Relevance Threshold: Rating >= {RELEVANCE_THRESHOLD}")
    print(f"Top-N (K): {TOP_N_RECOMMENDATIONS}")
    print(f"Users Evaluated: {metrics['Evaluated Users']}")
    print(f"Precision@{TOP_N_RECOMMENDATIONS}: {metrics[f'Precision@{TOP_N_RECOMMENDATIONS}']:.4f}")
    print(f"Recall@{TOP_N_RECOMMENDATIONS}:   {metrics[f'Recall@{TOP_N_RECOMMENDATIONS}']:.4f}")
    print(f"F1@{TOP_N_RECOMMENDATIONS}:        {metrics[f'F1-Score@{TOP_N_RECOMMENDATIONS}']:.4f}")

    # 9. Sample recommendation for one user
    sample_user = random.choice(user_item.index.tolist())
    sample_recs = recs[recs['UserID'] == sample_user].sort_values(by='Predicted_Rating', ascending=False).head(TOP_N_RECOMMENDATIONS)
    sample_recs = sample_recs.merge(movies, on='MovieID', how='left')
    print("\nSample recommendations for user", sample_user)
    print(sample_recs[['Title', 'Genres', 'Predicted_Rating']].to_string(index=False))

    end_time = time.time()
    print("\n=============================================")
    print(f"**Total Execution Time:     {end_time - start_time:.2f}**")
    print("=============================================")

if __name__ == "__main__":
    main()

# =============================== END ===============================


