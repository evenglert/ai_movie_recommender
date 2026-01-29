# # AI-POWERED MOVIE RECOMMENDATION ENGINE
# ## LLM GENERATED PERSONALIZED MESSAGES WITH MOVIE RECOMMENDATIONS
# ### Author: Evgeniya Englert
# ### Last update: 2026-01-29
# ## Summary
# * FastAPI service for LLM-enhanced personalized movie recommendations
# * This code generates user-centric message with movie recommendations using pre-trained LLM Ollama based on provided user information and pre-prepared in the previous step movie recommendations

# FastAPI
# * start bash (cmd)
# * cd "your path"
# * python -m uvicorn evenglert_ai_movie_recommender_9_llm_fastapi:app --reload
# * You should see: INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
# * Open a browser and go to: http://127.0.0.1:8000/docs
# * You’ll see your Swagger UI with all endpoints.
# * The /recommendations POST endpoint will appear.
# * Provide a JSON body like:
# {
#   "user_id": 15
# }
# * The API will return the top 3 personalized recommendations with decoded demographics and LLM message.
# * Examples: user_id: 1 - 6040
# 
# FastAPI service that loads precomputed movie recommendations and user metadata, then uses an Ollama LLM to generate personalized, human-like movie recommendation messages.

# =============================== BEGIN ===============================

import ollama
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ====================================================
# Configuration
# ====================================================
OLLAMA_MODEL = "mistral"

RECS_FILE = "top3_recommendations_per_user_cf_cosine.csv"
USERS_FILE = "users.dat"

USER_COLUMNS = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
TOP_K_LLM_RECS = 3

app = FastAPI(
    title="WatchIt Recommendation API",
    description="Generate personalized movie recommendations using collaborative filtering + LLM",
    version="1.0.0"
)

# ====================================================
# Decode Dictionaries
# ====================================================
GENDER_MAP = {"M": "Male", "F": "Female"}

AGE_MAP = {
    1: "Under 18",
    18: "18–24",
    25: "25–34",
    35: "35–44",
    45: "45–49",
    50: "50–55",
    56: "56+"
}

OCCUPATION_MAP = {
    0: "Other / Not specified",
    1: "Academic / Educator",
    2: "Artist",
    3: "Clerical / Admin",
    4: "College / Grad Student",
    5: "Customer Service",
    6: "Doctor / Health Care",
    7: "Executive / Managerial",
    8: "Farmer",
    9: "Homemaker",
    10: "K-12 Student",
    11: "Lawyer",
    12: "Programmer",
    13: "Retired",
    14: "Sales / Marketing",
    15: "Scientist",
    16: "Self-employed",
    17: "Technician / Engineer",
    18: "Tradesman / Craftsman",
    19: "Unemployed",
    20: "Writer"
}

# ====================================================
# Load Data (once at startup)
# ====================================================
recs_df = pd.read_csv(RECS_FILE)

users_df = pd.read_csv(
    USERS_FILE,
    sep="::",
    engine="python",
    names=USER_COLUMNS,
    encoding="latin-1"
)

users_df["Gender_Text"] = users_df["Gender"].map(GENDER_MAP)
users_df["Age_Text"] = users_df["Age"].map(AGE_MAP)
users_df["Occupation_Text"] = users_df["Occupation"].map(OCCUPATION_MAP)


# ====================================================
# Request Model (Select Window Input)
# ====================================================
class UserRequest(BaseModel):
    user_id: int


# ====================================================
# LLM Message Generator
# ====================================================
def generate_personalized_message(user_row, user_recs):

    rec_list = "\n".join(
        f"- {row.Title} ({row.Genres})"
        for _, row in user_recs.iterrows()
    )

    system_prompt = (
        "You are a warm, friendly, and emotionally intelligent movie recommender for the platform WatchIt. "
        "Adapt tone and vocabulary to the user’s age. "
        "Naturally connect the user’s lifestyle or occupation to the mood or themes of the recommended genres. "
        "Write exactly 2–3 sentences, each on a new line. "
        "Use bullet points ONLY for the top 3 movie recommendations. "
        "Do not mention zip code."
    )

    user_prompt = f"""
    USER PROFILE:
    Age Group: {user_row['Age_Text']}
    Gender: {user_row['Gender_Text']}
    Occupation: {user_row['Occupation_Text']}

    RECOMMENDED MOVIES:
    {rec_list}

    MESSAGE:
    """

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.8, "num_ctx": 4096}
    )

    return response["message"]["content"].strip()


# ====================================================
# API Endpoint
# ====================================================
@app.post("/recommendations")
def get_personalized_recommendations(request: UserRequest):

    user_row = users_df[users_df["UserID"] == request.user_id]

    if user_row.empty:
        raise HTTPException(status_code=404, detail="UserID not found")

    user_row = user_row.iloc[0]

    user_recs = (
        recs_df[recs_df["UserID"] == request.user_id]
        .sort_values("Predicted_Rating", ascending=False)
        .head(TOP_K_LLM_RECS)
    )

    if user_recs.empty:
        raise HTTPException(status_code=404, detail="No recommendations for this user")

    message = generate_personalized_message(user_row, user_recs)

    return {
        "user": {
            "UserID": int(user_row["UserID"]),
            "Age": user_row["Age_Text"],
            "Gender": user_row["Gender_Text"],
            "Occupation": user_row["Occupation_Text"]
        },
        "top_recommendations": [
            {
                "title": row.Title,
                "genres": row.Genres,
                "predicted_rating": round(row.Predicted_Rating, 2)
            }
            for _, row in user_recs.iterrows()
        ],
        "personalized_message": message
    }

# =============================== END ===============================