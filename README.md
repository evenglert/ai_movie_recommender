# README: AI-Powered Movie Recommendation System 🎬✨

This repository contains the documentation and architectural overview for the **AI-Powered Movie Recommendation System**, a final project by **Evgeniya Englert**. The system addresses user retention challenges on streaming platforms by merging traditional collaborative filtering with a **Large Language Model (LLM)** for a human-centered experience.

---

## 📈 Business Problem

Modern streaming platforms face the challenge of keeping users engaged amidst overwhelming content. This project solves this by:

* **Increasing User Satisfaction**: Delivering suggestions that match specific preferences.
* **Boosting Engagement**: Driving longer sessions and higher viewing frequency.
* **Strategic Differentiation**: Transforming algorithmic outputs into "human-like" communication to make users feel personally valued.

---

## 💾 Datasets

The project utilizes the **MovieLens 1M dataset**:

* **Users**: 6,040 
* **Movies**: ~3,900 
* **Ratings**: ~1,000,000 

| Dataset | Attributes | Role in Analysis |
| --- | --- | --- |
| **Users** | UserID, Gender, Age, Occupation, Zip-code | Used to enhance personalization.|
| **Ratings** | UserID, MovieID, Rating, Timestamp (dropped) | Used to construct the User-Item Matrix for training.|
| **Movies** | MovieID, Title, Genres | Used for interpreting results and presenting recommendations.|

---

## 🧪 Recommendation Methods & Evaluation

Multiple collaborative filtering (CF) approaches were tested to identify the best balance of accuracy and efficiency.

### Performance Comparison

| CF Approach | Precision@10 | Recall@10 | F1-Score@10 | Execution Time | Outcome |
| --- | --- | --- | --- | --- | --- |
| **User-Based (Cosine)** | **0.2325** | **0.8358** | **0.3414** | **143 sec** | <br>**Selected Model** |
| User-Based (Pearson) | 0.0334 | 0.2498 | 0.0575 | ~128 min | Rejected |
| Cluster-Based | 0.0002 | 0.0023 | 0.0004 | 252 sec | Fast but poor quality |
| Item-Based | 0.0018 | 0.0175 | 0.0033 | ~4.4 hrs | Rejected |
---
## 🧠 Final Architecture

### A. Core Recommendation Engine

The system uses **User-Based Collaborative Filtering with Cosine Similarity**.

* **Similarity Metric**: Cosine Similarity on the User–Item Matrix.

* **Neighbor Selection**: Users with the most similar rating vectors are chosen.

* **Logic**: Recommends highly-rated movies from similar neighbors, excluding those the user has already seen.

### B. Personalization Layer (LLM Integration)

A pre-trained LLM is integrated via the **Ollama API** to enhance the user experience.

* **Purpose**: Generate friendly, exciting, and natural-sounding messages to introduce the recommendations.

* **Prompt Engineering**: The LLM receives the user's name, job, and age alongside movie titles to create a customized pitch (e.g., *"You are a friendly and enthusiastic movie recommender for 'WatchIt'..."*).

* **Goal**: Transform data into human-centered communication to increase trust and perceived value.

---
## 🚀 End-to-End Workflow
1. **Data Ingestion**: Loading raw MovieLens data.
2. **Preprocessing**: Cleaning and constructing the User–Item Matrix.
3. **Model Execution**: Running the CF model and evaluating performance.
4. **Recommendation**: Generating the top-N movie list.
5. **Personalization**: Using the LLM to create a tailored introduction.
6. **Delivery**: Sending the final recommendations to the user via API.
---
## Modules
Modules
* **evenglert_ai_movie_recommender_1_user_data_exploration_clustering:** Exploration of a user dataset for use in a recommendation engine
* **evenglert_ai_movie_recommender_2_cluster_based_collaborative_filtering:** Cluster-based collaborative filtering using PCA, K-Means
* **evenglert_ai_movie_recommender_3_user_based_collaborative_filtering_cosine_similarity:** User-Based Collaborative Filtering (UBCF) pipeline using cosine similarity to find similar users (neighbors)
* **evenglert_ai_movie_recommender_4_user_based_collaborative_filtering_pearson_correlation:** Optimized User-Based Collaborative Filtering (UBCF) pipeline using Pearson correlation
* **evenglert_ai_movie_recommender_5_hybrid_collaborative_filtering_heatmap:** Hybrid User + Item Collaborative Filtering with Cosine Similarity

## Resources

* **MovieLens 1M Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset).