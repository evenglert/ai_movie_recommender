"# ai_movie_recommender" 
# AI-Powered Movie Recommendations 
## Author: Evgeniya Englert 

An AI-powered movie recommendation engine will be developed using a collaborative filtering approach. The project will leverage data of signed-in users to identify similar users based on their provided movie ratings and recommend movies based on the viewing habits of those similar users. A Large Language Model (LLM) will personalize the recommendations with custom messages, enhancing the user experience.
## Business Description 📊
This project addresses a key business challenge: user engagement and retention on a movie streaming platform. By providing highly relevant and personalized movie recommendations, the platform can increase user satisfaction, encourage more frequent use, and ultimately drive higher revenue. The recommendation engine serves as a strategic tool to improve the overall customer experience and differentiate the platform from competitors. The use of an LLM to generate personalized messages adds a layer of sophistication, making the recommendations feel more like a personal suggestion.
Data Science Description 🔬
This project aims to build a user-centric collaborative filtering recommendation system. The core methodology involves using unsupervised machine learning, specifically k-means clustering, to group users with similar movie profiles. The project will process the MovieLens dataset, which contains user data, ratings, and movie information. The final output is a system that, given a new user's information, assigns them to a movie cluster and provides recommendations based on the most popular movies within that cluster. The use of an LLM for personalized messages introduces a natural language generation (NLG) component, adding a novel layer to the recommendation process.
Project Steps 💻
## Data Acquisition and Preparation
* Acquire Raw Data: The project will start by acquiring the public Kaggle dataset: MovieLens dataset, which includes user-movie interactions.
* These files contain 1,000,209 anonymous ratings of approximately 3,900 movies 
* made by 6,040 MovieLens users who joined MovieLens in 2000.
* It includes:
** userId 
** movieId
** rating (e.g., 1-5 stars)
** timestamp

## Collaborative Filtering
* Collaborative Filtering works by finding users with similar tastes (e.g., users who rated the same movies similarly). It answers the question: "What did people similar to me like?"
* For cold-start users (those without explicit user-movie interactions), demographic filtering will be implemented using user clusters derived from available demographic data.

## Recommendation Generation
This project will pivot to one of these standard techniques of Collaborative Filtering to generate recommendations for users:
* User-Based Collaborative Filtering: Find a user's nearest neighbors based on rating patterns (using metrics like cosine similarity or Pearson correlation) and recommend movies highly rated by those neighbors.
* Item-Based Collaborative Filtering: Recommend movies that are similar to the ones a user has already rated highly.
* Hybrid Collaborative Filtering.

## Personalized Messaging with LLM
* Input for LLM: The user's name (or a generic greeting), their demographic information, and the list of recommended movies will be provided to a pre-trained LLM.
* NLG Generation: The LLM will generate a personalized, natural-sounding message introducing the recommendations, making the user feel more valued and understood.
* We plan to use a pre-trained LLM model via an API (like Ollama, the Gemini API, or OpenAI's GPT) with prompt engineering. This involves creating a well-crafted prompt.
* Example Prompt:
** "You are a friendly and enthusiastic movie recommender for our platform, 'WatchIt'. A user named [User's Name] is a [User's Job] and is [User's Age]. Based on their tastes, we think they'll love these movies: [Movie 1], [Movie 2], and [Movie 3]. Write a short, exciting, and personalized message (2-3 sentences) introducing these recommendations to them."

## Summary of the Workflow
The project workflow can be visualized as a pipeline where data flows through different stages, each adding value. Starting with raw data, it is cleaned and processed, then fed into the collaborative filtering model. The outputs will be used to generate recommendations, which are then passed to the pre-trained LLM for personalization before being presented to the user.

## Resources:
MovieLens dataset: https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

