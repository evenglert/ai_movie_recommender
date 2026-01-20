# # AI-POWERED MOVIE RECOMMENDATION ENGINE
# ## COLLABORATIVE FILTERING VIA USER CLUSTERING
# ## Author: Evgeniya Englert
# ## Last update: 2026-01-20
# 
# ## Summary: Cluster-based Recommendation System using Collaborative filtering, PCA and K-Means
# 
# This code executes the full **Collaborative Filtering** pipeline, using **Dimensionality Reduction (PCA)** 
# and **Unsupervised Clustering (K-Means)** to create user segments. 
# The final step is a **Cluster-Based Recommendation** generation, 
# where a target user's cluster determines the top predicted movies.
# 
# ***
# 
# ### Data Description
# 
# The analysis utilizes the **MovieLens 1M dataset**, which includes approximately 1 million ratings from 6,040 distinct users across approx 3,900 movies.
# 
# | Dataset | File | Attributes | Role in Analysis |
# | :--- | :--- | :--- | :--- |
# | **Ratings** | `ratings.dat` | UserID, MovieID, Rating, Timestamp (dropped) | Used to construct the User-Item Matrix and train the PCA/K-Means model. |
# | **Movies** | `movies.dat` | MovieID, Title, Genres | Used for interpreting and presenting the final movie recommendations. |
# 
# ***
# 
# ### Purpose
# 
# The primary **PURPOSE** of this code is to implement a **cluster-based collaborative filtering** system. This involves:
# 
# 1.  **Reducing the high-dimensional User-Item Matrix** using PCA to handle sparsity and improve computational efficiency.
# 
# 2.  **Identifying latent groups of users** (clusters) who share similar movie tastes using K-Means.
# 
# 3.  **Generating personalized recommendations** for a target user based on the average movie ratings of their assigned cluster.
# 
# ***
# 
# ### Key Steps and Methodology
# 
# The code follows a complete machine learning pipeline for generating recommendations, as implemented in the Python code:
# 
# 1.  **Data Loading and Matrix Creation:**
#     * Load the `ratings.dat` and `movies.dat` files.
#     * Create and preprocess the **User-Item Matrix** (users as rows, movies as columns, ratings as values), filling unrated movies with $0$ for dimensionality reduction.
# 
# 2.  **Dimensionality Reduction (PCA):**
#     * Apply **Principal Component Analysis (PCA)** to the User-Item Matrix to reduce the feature space from $\approx 3,900$ movies to a managed set of $100$ components (`N_COMPONENTS`).
# 
# 3.  **User Clustering (K-Means):**
#     * Apply the **K-Means** algorithm (using K=5, defined by OPTIMAL_K) on the reduced feature set to assign every user to a preference cluster.
# 
# 4.  **Recommendation Generation:**
#     * Identify the target user's cluster (`target_cluster`).
#     * Calculate the **average rating** for all movies within that cluster (the cluster's "taste vector").
#     * Filter this list to only include movies the target user **has not yet rated**.
#     * Recommend the **Top 10** movies (`TOP_N_RECOMMENDATIONS`) with the highest average predicted ratings.
#     * Note: Each user will be assigned to a cluster. So each movie can be present in many clusters.
# 
# ***
# 
# Original MovieLens datasets used: https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

# **SUMMARY - Evaluation**
# 
# The code produces a ranked list of Top-K movie recommendations per user.
# 
# A movie is considered relevant if: Rating ≥ relevance_threshold (default = 4.0)
# 
#   Precision@10: → Measures recommendation quality: Of the top-K recommended movies, what proportion were relevant.
# 
#   Recall@10:    → Measures coverage of user preferences: Of all relevant movies for a user, how many were retrieved in the top-K.
# 
#   F1-Score@10:  → Harmonic mean of Precision@K and Recall@K: How cluster preferences align with user taste.
# 
# **Execution results**
# **Clusetr-Based Collaborative Filtering**
# * Precision@10: mean=0.0002, std=0.0001
# * Recall@10: mean=0.0018, std=0.0006
# * F1-Score@10: mean=0.0003, std=0.0001
# * Evaluated Users: 6015
# * Total Execution Time: 252 seconds ⏱️
# 
# **High-level architecture**
# Class RecommenderEvaluatorClusterBasedCF
# │
# ├── fit_predict()          → Model training & prediction (PCA + KMeans + predictions)
# ├── user_stratified_kfold()→ User-aware cross-validation
# ├── evaluate_fold()        → Fold evaluation (ranking metrics for one fold)
# ├── cross_validate()       → User-aware cross-validation runner
# └── summarize_results()    → Metrics aggregation (mean / std metrics)

# =============================== BEGIN ===============================

# ===============================
# Import libraries
# ===============================

from pathlib import Path

# Standard library
import time
import random

# Third-party libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ProcessPoolExecutor, as_completed

# Warnings (optional cleanup)
import warnings
warnings.filterwarnings("ignore")

# ===============================
# File and column definitions
# ===============================
BASE_DIR = Path(__file__).resolve().parent

RATING_FILE_PATH = BASE_DIR / 'ratings.dat'
MOVIES_FILE_PATH = BASE_DIR / 'movies.dat'

RATING_COLUMN_NAMES = ['UserID', 'MovieID', 'Rating', 'Timestamp']
MOVIES_COLUMN_NAMES = ['MovieID', 'Title', 'Genres']

# ==============================
# Define parameters for main run
# ==============================

# Define parameters for cross-validation and model
n_splits = 5      # 5-fold CV
n_components = 10  # PCA components
n_clusters = 5     # KMeans clusters

# Define parameters for evaluator
relevance_threshold = 4.0  # Ratings >= 4 are considered relevant
top_n = 10                 # Number of top recommendations to evaluate

# ====================================
# Define functions needed in the code
# ====================================

# ===============================
# Data Loading
# ===============================

def load_data():
    # Load ratings
    ratings = pd.read_csv(
        RATING_FILE_PATH,
        sep='::',
        engine='python',
        names=RATING_COLUMN_NAMES,
        encoding='latin-1'
    )

    # Load movies
    movies = pd.read_csv(
        MOVIES_FILE_PATH,
        sep='::',
        engine='python',
        names=MOVIES_COLUMN_NAMES,
        encoding='latin-1'
    )

    return ratings, movies

# =========================================================================
# Define function to evaluate ranking metrics: Precision, Recall, F1-Score
# =========================================================================

def evaluate_ranking_metrics(predictions_df, test_ratings_df, relevance_threshold, K):
    relevant_test_items = test_ratings_df[test_ratings_df['Rating'] >= relevance_threshold] \
        .groupby('UserID')['MovieID'].apply(set).to_dict()

    # Only rank once, outside the loop
    predictions_df = predictions_df.copy()
    predictions_df['rank'] = predictions_df.groupby('UserID')['Predicted_Rating'] \
        .rank(method='first', ascending=False)

    top_k_predictions = predictions_df[predictions_df['rank'] <= K]

    precisions, recalls, f1s = [], [], []

    for user_id, group in top_k_predictions.groupby('UserID'):
        relevant = relevant_test_items.get(user_id)
        if not relevant:
            continue

        recommended = group['MovieID'].tolist()
        y_true = [1 if movie_id in relevant else 0 for movie_id in recommended]
        y_pred = [1] * len(y_true)

        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return {
        f'Precision@{K}': np.mean(precisions) if precisions else 0.0,
        f'Recall@{K}': np.mean(recalls) if recalls else 0.0,
        f'F1-Score@{K}': np.mean(f1s) if f1s else 0.0,
        'Evaluated Users': len(precisions)
    }
    
# ========================================================================
# Define RecommenderEvaluatorClusterBasedCF class including functions:
# fit_predict(): Model training & prediction (PCA + KMeans + predictions)
# user_stratified_kfold(): User-aware cross-validation
# evaluate_fold(): Fold evaluation (ranking metrics for one fold)
# cross_validate(): User-aware cross-validation runner
# summarize_results(): Metrics aggregation (mean / std metrics)
# ========================================================================

class RecommenderEvaluatorClusterBasedCF:
    # ================================================================
    # Function __init__()
    # Purpose: Initializes the evaluator with key parameters.
    # Parameters:
    # relevance_threshold: Minimum rating to consider an item relevant.
    # top_n: Number of top recommendations to evaluate.
    # random_state: Seed for reproducibility.
    # verbose: Whether to print progress messages.
    # ================================================================
    
    def __init__(
        self,
        relevance_threshold,
        top_n,
        random_state=42,
        verbose=True
    ):
        self.relevance_threshold = relevance_threshold
        self.top_n = top_n
        self.random_state = random_state
        self.verbose = verbose

    def _print(self, msg):
        if self.verbose:
            print(msg)

    # ==================================================================
    # Function fit_predict(): Model training & prediction
    # Purpose: Fits the cluster-based collaborative filtering model 
    # and predicts ratings for unknown user-item pairs.
    # Steps:
    # 1. Pivot ratings into a user-item matrix.
    # 2. Fill missing ratings with 0.
    # 3. Scale features and apply PCA for dimensionality reduction.
    # 4. Cluster users using KMeans.
    # 5. Compute cluster-level average ratings.
    # 6. Assign cluster ratings to each user and return only predictions 
    #    for items the user hasn’t rated.
    # Returns: Long-format DataFrame of predicted ratings 
    # (UserID, MovieID, Predicted_Rating).
    # ==================================================================
    def fit_predict(
        self,
        train_ratings_df,
        n_components,
        n_clusters
    ):
        train_user_movie_matrix = train_ratings_df.pivot(
            index='UserID',
            columns='MovieID',
            values='Rating'
        )

        user_features = train_user_movie_matrix.fillna(0)

        scaler = StandardScaler()
        user_features_scaled = scaler.fit_transform(user_features)

        pca = PCA(
            n_components=n_components,
            random_state=self.random_state
        )
        reduced_features = pca.fit_transform(user_features_scaled)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(reduced_features)

        user_features = user_features.copy()
        user_features['Cluster'] = clusters

        cluster_profiles = (
            train_ratings_df
            .merge(
                user_features[['Cluster']],
                left_on='UserID',
                right_index=True
            )
            .groupby(['Cluster', 'MovieID'])['Rating']
            .mean()
            .unstack(fill_value=0)
        )

        cluster_ratings = cluster_profiles.reindex(
            user_features['Cluster'].values
        )
        cluster_ratings.index = user_features.index

        preds = cluster_ratings.where(
            train_user_movie_matrix.isna()
        )

        preds_long = (
            preds
            .stack()
            .reset_index()
            .rename(columns={0: 'Predicted_Rating'})
        )

        return preds_long

    # ======================================================================
    # Function user_stratified_kfold(): User-aware cross-validation
    # Purpose: Generates user-stratified K-Fold splits for cross-validation.
    # Steps:
    # 1. Groups data by UserID.
    # 2. Shuffles and splits each user’s ratings across folds.
    # 3. Ensures every fold has each user represented.
    # Returns: Yields (train_df, test_df) pairs for each fold.
    # ======================================================================
    def user_stratified_kfold(self, ratings_df, n_splits):
        rng = np.random.RandomState(self.random_state)
        user_groups = ratings_df.groupby('UserID')

        folds = [([], []) for _ in range(n_splits)]

        for _, group in user_groups:
            idx = group.index.to_numpy()
            rng.shuffle(idx)
            splits = np.array_split(idx, n_splits)

            for i in range(n_splits):
                folds[i][0].extend(
                    np.concatenate(
                        [splits[j] for j in range(n_splits) if j != i]
                    )
                )
                folds[i][1].extend(splits[i])

        for train_idx, test_idx in folds:
            yield (
                ratings_df.loc[train_idx],
                ratings_df.loc[test_idx]
            )

    # ===========================================================
    # Function evaluate_fold(): Fold evaluation
    # Purpose: Evaluates model performance on a single CV fold.
    # Steps:
    # 1. Calls fit_predict on training data.
    # 2. Computes ranking metrics (via evaluate_ranking_metrics) 
    #    comparing predicted vs. test ratings.
    # 3. Prints fold metrics.
    # Returns: Dictionary of metrics for the fold.
    # ===========================================================
    def evaluate_fold(
        self,
        train_ratings_df,
        test_ratings_df,
        n_components,
        n_clusters,
        fold_id=None
    ):
        fold_start = time.perf_counter()
    
        if fold_id is not None:
            self._print(f"\n▶ Fold {fold_id}")
    
        preds = self.fit_predict(
            train_ratings_df,
            n_components,
            n_clusters
        )
    
        metrics = evaluate_ranking_metrics(
            preds,
            test_ratings_df,
            self.relevance_threshold,
            self.top_n
        )
    
        fold_time = time.perf_counter() - fold_start
        metrics['Fold Execution Time (s)'] = fold_time
    
        for k, v in metrics.items():
            if isinstance(v, float):
                self._print(f"  {k}: {v:.4f}")
            else:
                self._print(f"  {k}: {v}")
    
        self._print(f"  ⏱ Fold Time: {fold_time:.2f} seconds")
    
        return metrics
    

    # =====================================================
    # Function cross_validate():Cross-validation runner
    # Purpose: Runs cross-validation across multiple folds.
    # Steps:
    # 1. Uses user_stratified_kfold to generate folds.
    # 2. Calls evaluate_fold for each fold.
    # 3. Aggregates fold metrics using summarize_results.
    # 4. Prints fold-wise and summary metrics.
    # Returns: List of metrics dictionaries for each fold.
    # =====================================================
    def cross_validate(
        self,
        ratings_df,
        n_splits,
        n_components,
        n_clusters
    ):
        cv_start = time.perf_counter()
    
        self._print(
            f"\n=============================="
            f"\nCV: PCA={n_components}, KMeans={n_clusters}"
            f"\n=============================="
        )
    
        fold_metrics = []
    
        for fold, (train_df, test_df) in enumerate(
            self.user_stratified_kfold(ratings_df, n_splits),
            start=1
        ):
            metrics = self.evaluate_fold(
                train_df,
                test_df,
                n_components,
                n_clusters,
                fold_id=fold
            )
            fold_metrics.append(metrics)
    
        cv_time = time.perf_counter() - cv_start
    
        summary = self.summarize_results(fold_metrics)
    
        # Fold timing stats
        fold_times = [m['Fold Execution Time (s)'] for m in fold_metrics]
        summary['Avg Fold Time (s)'] = {
            'mean': np.mean(fold_times),
            'std': np.std(fold_times)
        }
        summary['Total CV Time (s)'] = cv_time
    
        self._print("\n▶ CV Summary:")
        for k, v in summary.items():
            if isinstance(v, dict):
                self._print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
            else:
                self._print(f"  {k}: {v:.2f}")
    
        self._print(f"\n⏱ Total CV Time: {cv_time:.2f} seconds")
    
        return fold_metrics


    # ========================================================
    # Function summarize_results()(staticmethod): 
    #   Metrics aggregation
    # Purpose: Aggregates metrics across all folds.
    # Steps:
    # 1. Computes mean and standard deviation for each metric.
    # 2. Computes average number of evaluated users.
    # Returns: Dictionary with aggregated metrics.
    # ========================================================
    @staticmethod
    def summarize_results(fold_metrics):
        summary = {}
        keys = [k for k in fold_metrics[0] if k != 'Evaluated Users']

        for key in keys:
            values = [m[key] for m in fold_metrics]
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        summary['Evaluated Users'] = int(
            np.mean([m['Evaluated Users'] for m in fold_metrics])
        )

        return summary

# ===============
# Main execution
# ===============

def main():
    overall_start = time.perf_counter()

    ratings_df, movies_df = load_data()
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies.")

    evaluator = RecommenderEvaluatorClusterBasedCF(
        relevance_threshold=relevance_threshold,
        top_n=top_n,
        random_state=42,
        verbose=True
    )

    print("\nStarting cross-validation and predictions...")
    fold_metrics = evaluator.cross_validate(
        ratings_df,
        n_splits=n_splits,
        n_components=n_components,
        n_clusters=n_clusters
    )

    summary = evaluator.summarize_results(fold_metrics)
    print("\n=== Cross-validation Summary ===")
    for metric, value in summary.items():
        if isinstance(value, dict):
            print(f"{metric}: mean={value['mean']:.4f}, std={value['std']:.4f}")
        else:
            print(f"{metric}: {value}")

    total_time = time.perf_counter() - overall_start
    print(f"\n⏱️ Total Execution Time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
    
# =============================== END ===============================