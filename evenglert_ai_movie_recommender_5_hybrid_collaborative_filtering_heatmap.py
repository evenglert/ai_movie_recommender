# %% [markdown]
# # AI-Powered Movie Recommendation Engine
# ## User-Based & Hybrid Collaborative Filtering with Cosine Similarity
# ### Author: Evgeniya Englert
# ### Last update: 2026-01-27
#
# ---
#
# ## Summary
#
# This code implements a **movie recommendation system** using
# **collaborative filtering** techniques on the **MovieLens 1M dataset**.
#
# Two recommendation models are implemented:
#
# 1. **User-Based Collaborative Filtering (UBCF)**
#    - Uses **Cosine similarity** to measure similarity between users
#    - Predicts ratings using weighted averages of neighbors’ ratings
#
# 2. **Hybrid Collaborative Filtering**
#    - Combines **user-based CF** and **item-based CF**
#    - Uses Cosine similarity for both users and items
#    - Blends both scores using a configurable weight
#
# The system is evaluated using a **single stratified train/test split**
# (not cross-validation) and ranking-based metrics:
#
# - Precision@K
# - Recall@K
# - F1-Score@K
#
# Additional analysis includes:
# - User clustering using K-Means
# - User–user similarity heatmap visualization
# - Sample personalized recommendations
#
# ---
#
# ## Data Description
#
# The analysis uses the **MovieLens 1M dataset**, consisting of:
#
# | Dataset | File | Description |
# |------|------|------------|
# | Ratings | ratings.dat | User ratings for movies |
# | Movies | movies.dat | Movie titles and genres |
# | Users | users.dat | User demographic information |
#
# ---
#
# ## Methodology Overview
#
# ### 1. Data Loading
# - Load ratings, movies, and users datasets
# - Drop rating timestamps
#
# ### 2. Train/Test Split
# - Perform a **single stratified split by user**
# - Construct a **User–Item rating matrix** from training data
#
# ### 3. Similarity Computation
# - Compute **user–user Cosine similarity**
# - Missing ratings are filled with zeros
#
# ### 4. Recommendation Models
# - User-Based Collaborative Filtering
# - Hybrid User + Item Collaborative Filtering
#
# ### 5. Evaluation
# - Top-K ranking evaluation
# - Relevant items defined as rating ≥ 4.0
#
# ---
#
# ⚠️ **Notes**
# - This code does **not** use k-fold cross-validation
# - Similarity matrices are computed **once** and reused
#
# ---


# %% =============================== BEGIN ===============================

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from pathlib import Path
import time
import random
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RATING_FILE_PATH = BASE_DIR / "ratings.dat"
MOVIES_FILE_PATH = BASE_DIR / "movies.dat"
USERS_FILE_PATH = BASE_DIR / "users.dat"

TOP_N_RECOMMENDATIONS = 10
TEST_SIZE = 0.2
RELEVANCE_THRESHOLD = 4.0
NUM_NEIGHBORS = 10
NUM_CLUSTERS = 5

# ------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------
def load_data():
    """
    Load ratings, movies, and users datasets into pandas DataFrames.
    """
    rating_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    movie_cols = ["MovieID", "Title", "Genres"]
    users_cols = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]

    # Load ratings and drop timestamp
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

    # Load user demographics
    users_df = pd.read_csv(
        USERS_FILE_PATH,
        sep="::",
        engine="python",
        names=users_cols,
        encoding="latin-1"
    )

    return ratings_df, movies_df, users_df

# ------------------------------------------------------------
# Evaluation Metrics
# ------------------------------------------------------------
def evaluate_ranking_metrics(predictions_df, test_df, relevance_threshold, K):
    """
    Compute Precision@K, Recall@K, and F1@K for Top-K recommendations.
    """

    # Relevant test items per user
    relevant_items = (
        test_df[test_df["Rating"] >= relevance_threshold]
        .groupby("UserID")["MovieID"]
        .apply(set)
        .to_dict()
    )

    # Rank predictions per user
    predictions_df["rank"] = predictions_df.groupby("UserID")["Predicted_Rating"] \
        .rank(method="first", ascending=False)

    top_k = predictions_df[predictions_df["rank"] <= K]

    precisions, recalls, f1s = [], [], []
    evaluated_users = 0

    for user_id, group in top_k.groupby("UserID"):
        if user_id not in relevant_items:
            continue

        recommended = group["MovieID"].tolist()
        relevant = relevant_items[user_id]

        y_true = [1 if m in relevant else 0 for m in recommended]
        y_pred = [1] * len(y_true)

        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        evaluated_users += 1

    return {
        f"Precision@{K}": np.mean(precisions) if precisions else 0.0,
        f"Recall@{K}": np.mean(recalls) if recalls else 0.0,
        f"F1-Score@{K}": np.mean(f1s) if f1s else 0.0,
        "Evaluated Users": evaluated_users
    }

# ------------------------------------------------------------
# Cosine Similarity Computation
# ------------------------------------------------------------
def cosine_user_similarity(user_item_matrix):
    """
    Compute user–user Cosine similarity matrix.

    Missing ratings are filled with zeros before similarity computation.
    """

    similarity = cosine_similarity(user_item_matrix.fillna(0))
    return pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

# ------------------------------------------------------------
# User-Based Collaborative Filtering
# ------------------------------------------------------------
def generate_user_based_recommendations(user_item_matrix, similarity_df):
    """
    Generate Top-N recommendations using User-Based CF with Cosine similarity.
    """

    recommendations = []

    for user_id in user_item_matrix.index:
        # Identify most similar users (excluding self)
        similar_users = similarity_df[user_id].drop(user_id).dropna()
        neighbors = similar_users.sort_values(ascending=False).head(NUM_NEIGHBORS).index

        # Movies not yet rated by the user
        unrated_movies = user_item_matrix.columns[user_item_matrix.loc[user_id].isna()]

        for movie in unrated_movies:
            neighbor_ratings = user_item_matrix.loc[neighbors, movie].dropna()
            if neighbor_ratings.empty:
                continue

            sim_scores = similarity_df.loc[user_id, neighbor_ratings.index]
            score = np.dot(sim_scores, neighbor_ratings) / sim_scores.sum() \
                if sim_scores.sum() != 0 else 0

            recommendations.append({
                "UserID": user_id,
                "MovieID": movie,
                "Predicted_Rating": score
            })

    return pd.DataFrame(recommendations)

# ------------------------------------------------------------
# Hybrid Collaborative Filtering
# ------------------------------------------------------------
def generate_hybrid_recommendations(user_item_matrix, user_similarity, item_weight=0.5):
    """
    Generate recommendations using a hybrid approach that combines:
    - User-based CF (Cosine similarity)
    - Item-based CF (Cosine similarity)
    """

    # Item–Item Cosine similarity
    item_similarity = pd.DataFrame(
        cosine_similarity(user_item_matrix.fillna(0).T),
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    hybrid_recs = []

    for user_id in user_item_matrix.index:
        user_ratings = user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index

        for movie in unrated_movies:
            # User-based component
            sim_users = user_similarity[user_id].drop(user_id)
            user_scores = user_item_matrix.loc[sim_users.index, movie].dropna()
            sim_user_scores = sim_users[user_scores.index]
            user_score = np.dot(sim_user_scores, user_scores) / sim_user_scores.sum() \
                if not user_scores.empty else 0

            # Item-based component
            sim_items = item_similarity[movie].drop(movie)
            rated_items = user_ratings.dropna()
            sim_item_scores = sim_items[rated_items.index]
            item_score = np.dot(sim_item_scores, rated_items) / sim_item_scores.sum() \
                if not rated_items.empty else 0

            hybrid_score = (1 - item_weight) * user_score + item_weight * item_score

            hybrid_recs.append({
                "UserID": user_id,
                "MovieID": movie,
                "Predicted_Rating": hybrid_score
            })

    return pd.DataFrame(hybrid_recs)

# ------------------------------------------------------------
# Clustering & Visualization
# ------------------------------------------------------------
def cluster_users(similarity_df):
    """
    Cluster users using K-Means on Cosine similarity-derived distance matrix.
    """
    distance_matrix = 1 - similarity_df
    model = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    labels = model.fit_predict(distance_matrix)

    return pd.DataFrame({
        "UserID": similarity_df.index,
        "Cluster": labels
    })


def visualize_similarity_heatmap(similarity_df):
    """
    Display a heatmap of the user–user Cosine similarity matrix (subset).
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        similarity_df.iloc[:30, :30],
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=False
    )
    plt.title("User–User Cosine Similarity")
    plt.show()

# ------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------
def main():
    start_time = time.time()

    print("Loading data...")
    ratings_df, movies_df, users_df = load_data()

    # Train/test split (stratified by user)
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=TEST_SIZE,
        stratify=ratings_df["UserID"],
        random_state=42
    )

    # Build User–Item matrix
    user_item_matrix = train_df.pivot(
        index="UserID", columns="MovieID", values="Rating"
    )

    # Compute Cosine similarity
    user_similarity = cosine_user_similarity(user_item_matrix)
    visualize_similarity_heatmap(user_similarity)

    # Generate recommendations
    print("Generating User-Based recommendations...")
    user_cf = generate_user_based_recommendations(user_item_matrix, user_similarity)

    print("Generating Hybrid recommendations...")
    hybrid_cf = generate_hybrid_recommendations(user_item_matrix, user_similarity)

    # Evaluate models
    print("\nUser-Based CF Evaluation:")
    print(evaluate_ranking_metrics(
        user_cf, test_df, RELEVANCE_THRESHOLD, TOP_N_RECOMMENDATIONS
    ))

    print("\nHybrid CF Evaluation:")
    print(evaluate_ranking_metrics(
        hybrid_cf, test_df, RELEVANCE_THRESHOLD, TOP_N_RECOMMENDATIONS
    ))

    # Cluster users
    clusters = cluster_users(user_similarity)
    print("\nUser cluster distribution:")
    print(clusters["Cluster"].value_counts())

    # Sample recommendations
    random_user = random.choice(user_item_matrix.index.tolist())
    sample_recs = (
        hybrid_cf[hybrid_cf["UserID"] == random_user]
        .sort_values("Predicted_Rating", ascending=False)
        .head(10)
        .merge(movies_df, on="MovieID")
    )

    print(f"\nSample recommendations for User {random_user}:")
    print(sample_recs[["Title", "Genres", "Predicted_Rating"]].to_string(index=False))

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

# Entry point
if __name__ == "__main__":
    main()

# %% =============================== END ===============================
