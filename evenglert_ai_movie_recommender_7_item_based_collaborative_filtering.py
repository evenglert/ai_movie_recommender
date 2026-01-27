# # AI-POWERED MOVIE RECOMMENDATION ENGINE
# ## ITEM-BASED COLLABORATIVE FILTERING
# ## Author: Evgeniya Englert
# ## Last update: 2026-01-27
# 
# ## Summary: Item-Based Collaborative Filtering using Pearson Similarity
# 
# This notebook implements an **Item-Based Collaborative Filtering** recommendation system for movies. The approach leverages **Pearson Correlation Similarity** to predict ratings for unseen movies and generate **Top-N personalized recommendations** for each user.
# 
# ***
# 
# ### Data Description
# 
# The analysis uses the **MovieLens 1M dataset**, which contains ~1 million ratings from 6,040 users across ~3,900 movies.
# 
# | Dataset | File | Attributes | Role in Analysis |
# | :--- | :--- | :--- | :--- |
# | **Ratings** | `ratings.dat` | UserID, MovieID, Rating, Timestamp (dropped) | Construct the User-Item matrix and generate predictions. |
# | **Movies** | `movies.dat` | MovieID, Title, Genres | Map movie IDs to titles and genres for presenting recommendations. |
# 
# ***
# 
# ### Purpose
# 
# The main **PURPOSE** of this notebook is to implement **Item-Based Collaborative Filtering**:
# 
# 1. **Compute item-item similarity:** Using Pearson correlation on the User-Item rating matrix.
# 2. **Predict ratings for unrated items:** Based on similar items the user has rated.
# 3. **Generate personalized Top-N recommendations** for each user.
# 4. **Evaluate recommendation quality** using Precision@K, Recall@K, and F1@K.
# 
# ***
# 
# ### Key Steps and Methodology
# 
# 1. **Data Loading:**
#    * Load `ratings.dat` and `movies.dat`.
#    * Drop the `Timestamp` column from ratings.
# 
# 2. **Train-Test Split:**
#    * Split ratings into train and test sets (default 80/20 split) using stratified sampling per user.
# 
# 3. **Build Item-User Matrix:**
#    * Rows = users, Columns = movies, Values = ratings.
#    * NaNs indicate unrated items.
# 
# 4. **Item Similarity Computation:**
#    * Compute **Pearson correlation** between movies using the item-user matrix.
# 
# 5. **Generate Recommendations:**
#    * For each user, predict ratings for unrated movies based on top-N most similar items they have rated.
#    * Filter top predictions to produce **Top-N recommendations**.
# 
# 6. **Evaluation:**
#    * Metrics used: **Precision@K, Recall@K, F1-Score@K**.
#    * A movie is considered relevant if `Rating >= 4.0`.
#    * Evaluates recommendation quality per user.
# 
# 7. **Sample Recommendations:**
#    * Display the top-N recommendations for a randomly selected user, including movie **Title**, **Genres**, and predicted rating.
# 
# ***
# 
# ### Evaluation Summary
# 
# **Sample Output Metrics (for reference):**
# 
# * Relevance Threshold: Rating >= 4.0
# * Top-N (K): 10
# * Users Evaluated: 6013
# * Precision@10: 0.0018
# * Recall@10:   0.0175
# * F1@10:        0.0033
# * Execution Time: ~4.4 hrs
# 
# ---
# 
# **High-level workflow:**
# 
#                 MovieLens Datasets
#               (ratings.dat, movies.dat)
#                            │
#                            ▼
#                     Load Data
#         ─────────────────────────────────
#         - Read ratings and movies
#         - Drop rating timestamp column
#                            │
#                            ▼
#               Stratified Train/Test Split
#         ─────────────────────────────────
#         - Split ratings by UserID
#         - Hold out TEST_SIZE for evaluation
#                            │
#                            ▼
#             Build Item–User Rating Matrix
#         ─────────────────────────────────
#         - Use training data only
#         - Rows: Users
#         - Columns: Movies
#         - Values: Ratings (NaN = unrated)
#                            │
#                            ▼
#            Compute Item–Item Similarity
#         ─────────────────────────────────
#         - Use Pearson correlation on item-user matrix
#         - Fill missing correlations with 0
#                            │
#                            ▼
#      Generate Item-Based Recommendations
#         ─────────────────────────────────
#         For each user:
#           1. Find unrated movies
#           2. Identify top-N similar items rated by the user
#           3. Predict rating using weighted average of neighbor ratings
#           4. Store predicted scores
#                            │
#                            ▼
#            Top-N Recommendation Selection
#         ─────────────────────────────────
#         - Rank predicted movies per user
#         - Select Top-N recommendations
#                            │
#                            ▼
#                 Ranking Evaluation
#         ─────────────────────────────────
#         - Compare Top-N lists with test set
#         - Relevant items: Rating ≥ 4.0
#         - Compute:
#             • Precision@N
#             • Recall@N
#             • F1-Score@N
#                            │
#                            ▼
#       Output: Personalized Top-N Movie Recommendations
# 

# =============================== BEGIN ===============================

# Item-Based Collaborative Filtering for Movie Recommendation System Using Pearson Correlation Similarity
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# --- Configuration ---
BASE_DIR = Path.cwd() 
RATING_FILE_PATH = BASE_DIR / "ratings.dat"
MOVIES_FILE_PATH = BASE_DIR / "movies.dat"
TOP_N_RECOMMENDATIONS = 10
TEST_SIZE = 0.2
RELEVANCE_THRESHOLD = 4.0
NUM_NEIGHBORS = 10  # Number of similar items to consider

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
    rating_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    movie_cols = ['MovieID', 'Title', 'Genres']

    ratings = pd.read_csv(RATING_FILE_PATH, sep='::', names=rating_cols, engine='python', encoding='latin-1') \
        .drop(columns='Timestamp')
    movies = pd.read_csv(MOVIES_FILE_PATH, sep='::', names=movie_cols, engine='python', encoding='latin-1')

    return ratings, movies

# ----------------------------------------------------------------------
def compute_item_similarity(item_user_matrix):
    """
    Compute Pearson correlation similarity between items based on user ratings.
    """
    sim = item_user_matrix.corr(method='pearson')
    sim = sim.fillna(0)
    return sim

# ----------------------------------------------------------------------
def generate_item_based_recommendations(ratings_train, item_user_matrix, item_similarity):
    """
    Predict ratings for each user-item pair by looking at items similar to the target item.
    """
    all_recs = []
    users = item_user_matrix.index
    items = item_user_matrix.columns

    for user in users:
        user_ratings = item_user_matrix.loc[user]
        unrated_items = user_ratings[user_ratings.isna()].index

        for item in unrated_items:
            # Find top similar items that the user has rated
            similar_items = item_similarity[item].drop(labels=[item])
            similar_items = similar_items[similar_items > 0]  # Keep only positive correlations
            top_similar_items = similar_items.sort_values(ascending=False).head(NUM_NEIGHBORS).index

            # Ratings of the user on those similar items
            neighbor_ratings = user_ratings[top_similar_items].dropna()
            if neighbor_ratings.empty:
                continue

            # Similarity scores for those items
            sim_scores = item_similarity.loc[item, neighbor_ratings.index]

            # Weighted average prediction
            if np.sum(np.abs(sim_scores)) == 0:
                continue
            pred_rating = np.dot(sim_scores.values, neighbor_ratings.values) / np.sum(np.abs(sim_scores))

            all_recs.append({'UserID': user, 'MovieID': item, 'Predicted_Rating': pred_rating})

    recs_df = pd.DataFrame(all_recs)
    return recs_df

# ----------------------------------------------------------------------
def main():
    # 1. Load data
    ratings, movies = load_data()

    # 2. Train-test split
    train_ratings, test_ratings = train_test_split(ratings, test_size=TEST_SIZE,
                                                   stratify=ratings['UserID'], random_state=42)

    # 3. Build item-user matrix from train set (rows=users, columns=items)
    item_user = train_ratings.pivot(index='UserID', columns='MovieID', values='Rating')

    # 4. Compute item-item similarity matrix
    item_sim = compute_item_similarity(item_user)

    print("Sample of item-item similarity matrix:")
    print(item_sim.iloc[:5, :5])

    # 5. Generate recommendations
    recs = generate_item_based_recommendations(train_ratings, item_user, item_sim)

    # 6. Evaluate recommendations
    print("\n--- Ranking Quality Evaluation ---")
    metrics = evaluate_ranking_metrics(recs, test_ratings, RELEVANCE_THRESHOLD, TOP_N_RECOMMENDATIONS)
    print(f"Relevance Threshold: Rating >= {RELEVANCE_THRESHOLD}")
    print(f"Top-N (K): {TOP_N_RECOMMENDATIONS}")
    print(f"Users Evaluated: {metrics['Evaluated Users']}")
    print(f"Precision@{TOP_N_RECOMMENDATIONS}: {metrics[f'Precision@{TOP_N_RECOMMENDATIONS}']:.4f}")
    print(f"Recall@{TOP_N_RECOMMENDATIONS}:   {metrics[f'Recall@{TOP_N_RECOMMENDATIONS}']:.4f}")
    print(f"F1@{TOP_N_RECOMMENDATIONS}:        {metrics[f'F1-Score@{TOP_N_RECOMMENDATIONS}']:.4f}")

    # 7. Sample recommendations for a random user
    sample_user = random.choice(item_user.index.tolist())
    sample_recs = recs[recs['UserID'] == sample_user].sort_values(by='Predicted_Rating', ascending=False).head(TOP_N_RECOMMENDATIONS)
    sample_recs = sample_recs.merge(movies, on='MovieID', how='left')
    print("\nSample recommendations for user", sample_user)
    print(sample_recs[['Title', 'Genres', 'Predicted_Rating']].to_string(index=False))

if __name__ == "__main__":
    main()

# =============================== END ===============================


