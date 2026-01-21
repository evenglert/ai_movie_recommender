# ai_movie_recommender
# AI-Powered Movie Recommendations
## Author: Evgeniya Englert

An AI-powered movie recommendation engine has been developed using a collaborative filtering approach. The project leverages data from signed-in users to identify similar profiles based on provided movie ratings and recommends titles based on the viewing habits of those similar users. A Large Language Model (LLM) personalizes the recommendations with custom messages, enhancing the user experience.

---

## Business Description 📊
This project addressed a key business challenge: user engagement and retention on a movie streaming platform. 

By providing highly relevant and personalized movie recommendations, the platform increases user satisfaction, encourages more frequent use, and ultimately drives higher revenue. 

The recommendation engine serves as a strategic tool to improve the overall customer experience and differentiate the platform from competitors. 

The integration of an LLM to generate personalized messages adds a layer of sophistication, making the recommendations feel like a personal suggestion rather than a generic list.

---

## Data Science Description 🔬
This project was built as a user-centric collaborative filtering recommendation system. 
The core methodology involved using unsupervised machine learning, specifically k-means clustering, to group users with similar movie profiles. 
The project processed the MovieLens dataset, containing user data, ratings, and movie information. 

The final system takes a user's information, assigns them to a specific movie cluster, and provides recommendations based on the most popular movies within that cluster. 
The use of an LLM for personalized messages introduced a natural language generation (NLG) component, adding a novel layer to the recommendation process.

---

## Project Components 💻

### Data Acquisition and Preparation
* **Raw Data Acquisition:** The project utilized the public MovieLens dataset from Kaggle, which includes user-movie interactions.
* **Dataset Scale:** The system processed 1,000,209 anonymous ratings of approximately 3,900 movies.
* **User Base:** Data was derived from 6,040 MovieLens users.
* **Key Features:** The processed data includes `userId`, `movieId`, `rating` (1-5 stars), and `timestamp`.

### Collaborative Filtering Implementation
* **Logic:** The system identifies users with similar tastes (those who rated the same movies similarly) to answer: "What did people similar to me like?"

### Recommendation Generation
The project utilizes standard Collaborative Filtering techniques to generate results:
* **User-Based Collaborative Filtering:** Finds a user's nearest neighbors based on rating patterns

### Modules
* **evenglert_ai_movie_recommender_1_user_data_exploration_clustering     :** Exploration of a user dataset for use in a recommendation engine
* **evenglert_ai_movie_recommender_2_cluster_based_collaborative_filtering:** Cluster-based collaborative filtering using PCA, K-Means
* **evenglert_ai_movie_recommender_3_user_based_collaborative_filtering_cosine_similarity:** User-Based Collaborative Filtering (UBCF) pipeline using cosine similarity to find similar users (neighbors)
* **evenglert_ai_movie_recommender_4_user_based_collaborative_filtering_pearson_correlation:** Optimized User-Based Collaborative Filtering (UBCF) pipeline using Pearson correlation

### DatasetResources:
* **MovieLens dataset:**  https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset
