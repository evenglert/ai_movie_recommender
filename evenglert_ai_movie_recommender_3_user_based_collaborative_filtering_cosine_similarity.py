# # AI-POWERED MOVIE RECOMMENDATION ENGINE  
# ## USER-BASED COLLABORATIVE FILTERING WITH COSINE SIMILARITY AND STRATIFIED 5-FOLD CV  
# ### Author: Evgeniya Englert  
# ### Last update: 2026-01-21  
# 
# ---
# 
# ## Summary  
# 
# This notebook implements a **User-Based Collaborative Filtering (UBCF)** pipeline using **cosine similarity** to find similar users (neighbors) and generate personalized movie recommendations.  
# It introduces **stratified 5-fold cross-validation** to evaluate recommendation quality robustly across users.  
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
# | **Ratings** | `ratings.dat` | UserID, MovieID, Rating, Timestamp (dropped) | Used to construct User-Item Matrix and for stratified 5-fold CV. |
# | **Movies** | `movies.dat` | MovieID, Title, Genres | Used to enrich recommendations with titles and genres. |
# 
# ---
# 
# ### Purpose  
# 
# The system aims to implement **user-based collaborative filtering with cosine similarity** and evaluate it with **stratified cross-validation**:
# 
# 1. **Compute user-user similarity** using cosine similarity.
# 2. **Identify top-N similar neighbors** for each user.
# 3. **Generate Top-N recommendations** based on neighbors’ ratings of unrated movies.
# 4. **Assess recommendation quality** using Precision, Recall, and F1-Score at top-K across 5 stratified folds.
# 
# ---
# 
# ### Key Steps and Methodology  
# 
# 1. **Data Loading and Preparation**
#    - Load ratings (`ratings.dat`) and movies (`movies.dat`) datasets.
#    - Drop timestamps.
#    - Prepare User-Item matrix for each training fold.
# 
# 2. **User Similarity Computation**
#    - Fill missing ratings with `0`.
#    - Compute **cosine similarity** between all users.
#    - Identify **top-K nearest neighbors** for each user.
# 
# 3. **Recommendation Generation**
#    - Predict ratings for unrated movies using **weighted averages of neighbors’ ratings**.
#    - Select **Top-N movies** with the highest predicted ratings.
# 
# 4. **Stratified 5-Fold Cross-Validation**
#    - Split ratings data into 5 folds **stratified by user** to ensure each fold contains all users proportionally.
#    - Generate recommendations and compute ranking metrics for each fold.
#    - Aggregate results across folds.
# 
# 5. **Evaluation**
#    - Metrics: **Precision@K**, **Recall@K**, **F1@K**, number of evaluated users, and fold execution time.
#    - Report **mean and standard deviation** across all folds.
# 
# ---
# 
# ### Ranking Evaluation  
# 
# - **Relevant movies:** Rating ≥ 4.0 (`RELEVANCE_THRESHOLD`)  
# - **Top-K cutoff:** `TOP_N` = 10  
# 
# Metrics computed:
# 
# | Metric | Description |
# | ------ | ----------- |
# | Precision@K | Proportion of top-K recommended movies that are relevant. |
# | Recall@K | Proportion of relevant movies retrieved in top-K recommendations. |
# | F1-Score@K | Harmonic mean of Precision@K and Recall@K. |
# 
# ---
# 
# ### Example Cross-Validation Results  
# 
# **User-Based Collaborative Filtering with Cosine Similarity (5-Fold CV)**
# 
# - Top-N Cutoff (K): 10  
# - Number of Folds: 5  
# - Mean Precision@10: ~0.0002  
# - Precision@10: mean=0.2325, std=0.0016
# - Recall@10:    mean=0.8358, std=0.0035
# - F1-Score@10:  mean=0.3413, std=0.0017
# - Evaluated Users:   6011.6
# - Fold Execution Time (s): mean=34.60, std=12.28 ⏱️
# - Total Execution Time: 178.84 seconds ⏱️
# 
# ---
# 
# ### High-Level Architecture: Stratified 5-Fold UBCF Workflow
# 
#               MovieLens Ratings Data
#                      │
#                      ▼
#              Load & Prepare Data
#      ─────────────────────────────
#      - Drop timestamp
#      - Prepare User-Item matrix
#                      │
#                      ▼
#         Stratified 5-Fold Cross-Validation
#      ─────────────────────────────────────
#      For each fold:
#        1. Train on 80% / Test on 20%
#        2. Generate User-Item Matrix
#        3. Compute cosine similarity
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

# =============================== BEGIN ===============================
# ====================================================================================
# User-Based Collaborative Filtering with Cosine Similarity and Stratified 5-Fold CV
# ====================================================================================

# ====================================================================================
# User-Based Collaborative Filtering with Cosine Similarity and Stratified 5-Fold CV
# ====================================================================================

import pandas as pd
import numpy as np
import random
import time

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RATING_FILE_PATH = "ratings.dat"   # Path to ratings dataset
MOVIES_FILE_PATH = "movies.dat"    # Path to movies dataset

TOP_N = 10                         # Number of top recommendations per user
RELEVANCE_THRESHOLD = 4.0          # Minimum rating to consider an item relevant
NUM_NEIGHBORS = 10                 # Number of similar users to consider
N_FOLDS = 5                         # Number of folds for cross-validation
RANDOM_STATE = 42                   # Random seed for reproducibility

# ------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------
def evaluate_ranking_metrics(predictions_df, test_df, relevance_threshold, k):
    """
    Compute Precision@K, Recall@K, F1@K averaged over all users.
    
    predictions_df: DataFrame containing predicted recommendations
    test_df: DataFrame with actual user ratings
    relevance_threshold: Minimum rating to consider a movie relevant
    k: Top-K items to evaluate
    """
    # Build dictionary of relevant items per user from the test set
    relevant_items = (
        test_df[test_df["Rating"] >= relevance_threshold]
        .groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    # Build dictionary of recommended items per user
    recommended_items = (
        predictions_df.groupby("UserID")["MovieID"]
        .apply(list)
        .to_dict()
    )

    precisions, recalls, f1s = [], [], []
    evaluated_users = 0

    # Evaluate metrics for each user
    for user_id, true_items in relevant_items.items():
        preds = recommended_items.get(user_id, [])[:k]  # Take top-K predictions

        if not preds:
            continue  # Skip users with no recommendations

        # Binary relevance for predicted items
        y_true = [1 if item in true_items else 0 for item in preds]
        y_pred = [1] * len(preds)  # All predicted items are treated as relevant

        # Compute precision, recall, and F1 for this user
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        evaluated_users += 1

    # Return average metrics across all users
    return {
        "Precision": np.mean(precisions) if precisions else 0,
        "Recall": np.mean(recalls) if recalls else 0,
        "F1": np.mean(f1s) if f1s else 0,
        "Evaluated Users": evaluated_users
    }

# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
def load_data():
    """
    Load ratings and movies datasets into pandas DataFrames.
    """
    rating_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    movie_cols = ["MovieID", "Title", "Genres"]

    # Load ratings, remove timestamp column
    ratings_df = pd.read_csv(
        RATING_FILE_PATH,
        sep="::",
        engine="python",
        names=rating_cols,
        encoding="latin-1"
    ).drop(columns="Timestamp")

    # Load movie metadata
    movies_df = pd.read_csv(
        MOVIES_FILE_PATH,
        sep="::",
        engine="python",
        names=movie_cols,
        encoding="latin-1"
    )

    return ratings_df, movies_df

# ------------------------------------------------------------
# User-Based Collaborative Filtering
# ------------------------------------------------------------
def generate_user_based_recommendations(train_df):
    """
    Generate recommendations for each user based on user-based collaborative filtering.
    """
    # Create user-item matrix: rows=users, columns=movies, values=ratings (fill missing with 0)
    user_item = train_df.pivot(
        index="UserID",
        columns="MovieID",
        values="Rating"
    ).fillna(0).astype(np.float32)

    user_ids = user_item.index
    movie_ids = user_item.columns

    # Compute cosine similarity between all users
    similarity = cosine_similarity(user_item.values)

    # Identify top-N most similar neighbors for each user (excluding self)
    top_neighbors = np.argsort(-similarity, axis=1)[:, 1:NUM_NEIGHBORS + 1]

    recommendations = []

    # Generate recommendations for each user
    for i, user_id in enumerate(user_ids):
        user_ratings = user_item.iloc[i].values
        unrated_mask = user_ratings == 0  # Identify movies not yet rated

        if not np.any(unrated_mask):
            continue  # Skip if user has rated all movies

        neighbor_idxs = top_neighbors[i]          # Get indices of top neighbors
        sim_scores = similarity[i, neighbor_idxs] # Similarity scores with neighbors
        sim_sum = sim_scores.sum()                # Sum of similarities for weighting

        if sim_sum == 0:
            continue  # Skip if all neighbor similarities are zero

        # Weighted sum of neighbors' ratings
        neighbor_ratings = user_item.iloc[neighbor_idxs].values
        weighted_scores = np.dot(sim_scores, neighbor_ratings) / sim_sum

        # Select unrated items and pick top-N by predicted score
        candidate_idxs = np.where(unrated_mask)[0]
        top_items = candidate_idxs[
            np.argsort(-weighted_scores[candidate_idxs])[:TOP_N]
        ]

        # Store recommendations
        for idx in top_items:
            recommendations.append({
                "UserID": user_id,
                "MovieID": movie_ids[idx],
                "Predicted_Rating": weighted_scores[idx]
            })

    return pd.DataFrame(recommendations)

# ------------------------------------------------------------
# Stratified Cross-Validation
# ------------------------------------------------------------
def run_stratified_cv(ratings_df):
    """
    Perform stratified K-Fold cross-validation on ratings data
    to evaluate recommendation performance.
    """
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(ratings_df, ratings_df["UserID"]), start=1
    ):
        print(f"\n--- Fold {fold}/{N_FOLDS} ---")
        fold_start = time.time()

        # Split data into train and test
        train_df = ratings_df.iloc[train_idx]
        test_df = ratings_df.iloc[test_idx]

        # Generate recommendations using training data
        recs_df = generate_user_based_recommendations(train_df)

        # Evaluate recommendations against test data
        metrics = evaluate_ranking_metrics(
            recs_df,
            test_df,
            RELEVANCE_THRESHOLD,
            TOP_N
        )

        fold_time = time.time() - fold_start

        print(
            f"Precision@{TOP_N}: {metrics['Precision']:.4f}, "
            f"Recall@{TOP_N}: {metrics['Recall']:.4f}, "
            f"F1@{TOP_N}: {metrics['F1']:.4f}, "
            f"Evaluated Users: {metrics['Evaluated Users']}, "
            f"Time: {fold_time:.2f}s"
        )

        metrics["Fold_Time"] = fold_time
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    total_start = time.time()

    # Load data
    ratings_df, movies_df = load_data()
    print(f"Total Ratings: {len(ratings_df)}")

    # Run stratified cross-validation and collect metrics
    cv_results = run_stratified_cv(ratings_df)

    total_time = time.time() - total_start

    # --------------------------------------------------------
    # Cross-Validation Summary
    # --------------------------------------------------------
    print("\n=================================================")
    print("Cross-Validation Summary: Execution & Evaluation results")
    print("=================================================")

    # Print average and standard deviation of metrics across folds
    print(
        f"* Precision@{TOP_N}: mean={cv_results['Precision'].mean():.4f}, "
        f"std={cv_results['Precision'].std():.4f}"
    )
    print(
        f"* Recall@{TOP_N}:    mean={cv_results['Recall'].mean():.4f}, "
        f"std={cv_results['Recall'].std():.4f}"
    )
    print(
        f"* F1-Score@{TOP_N}:  mean={cv_results['F1'].mean():.4f}, "
        f"std={cv_results['F1'].std():.4f}"
    )
    print(
        f"* Evaluated Users:   {cv_results['Evaluated Users'].mean():.1f}"
    )
    print(
        f"* Fold Execution Time (s): mean={cv_results['Fold_Time'].mean():.2f}, "
        f"std={cv_results['Fold_Time'].std():.2f} ⏱️"
    )
    print(f"* Total Execution Time: {total_time:.2f} seconds ⏱️")
    print("=================================================\n")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()


# Cross-Validation Summary: Execution & Evaluation results
# - Precision@10: mean=0.2325, std=0.0016
# - Recall@10:    mean=0.8358, std=0.0035
# - F1-Score@10:  mean=0.3413, std=0.0017
# - Evaluated Users:   6011.6
# - Fold Execution Time (s): mean=27.45, std=0.53 ⏱️
# - Total Execution Time: 143.33 seconds ⏱️

# =============================== END ===============================