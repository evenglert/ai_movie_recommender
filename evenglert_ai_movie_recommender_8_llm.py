# # AI-POWERED MOVIE RECOMMENDATION ENGINE
# ## LLM GENERATED PERSONALIZED MESSAGES WITH MOVIE RECOMMENDATIONS
# ### Author: Evgeniya Englert
# ### Last update: 2026-01-29
# ## Summary
# This code generates user-centric message with movie recommendations using pre-trained LLM Ollama based on provided user information and pre-prepared in the previous step movie recommendations

# =============================== BEGIN ===============================

import ollama
import pandas as pd
from pathlib import Path

# ====================================================
# Configuration
# ====================================================
OLLAMA_MODEL = "mistral"

BASE_DIR = Path(__file__).resolve().parent
RECS_FILE = BASE_DIR / "top3_recommendations_per_user_cf_cosine.csv"
USERS_FILE = BASE_DIR / "users.dat"

USER_COLUMNS = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
TOP_K_LLM_RECS = 3


# ====================================================
# Decode Dictionaries
# ====================================================
GENDER_MAP = {
    "M": "Male",
    "F": "Female"
}

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
# Load & Decode Data
# ====================================================
def load_data():
    """Load recommendations and decode user profile attributes."""

    # Load recommendation data
    recs_df = pd.read_csv(RECS_FILE)

    # Load user data
    users_df = pd.read_csv(
        USERS_FILE,
        sep='::',
        engine='python',
        names=USER_COLUMNS,
        encoding='latin-1'
    )

    # Decode encrypted fields
    users_df['Gender_Text'] = users_df['Gender'].map(GENDER_MAP)
    users_df['Age_Text'] = users_df['Age'].map(AGE_MAP)
    users_df['Occupation_Text'] = users_df['Occupation'].map(OCCUPATION_MAP)

    return recs_df, users_df


# ====================================================
# LLM Message Generator
# ====================================================
def generate_personalized_message(user_row, user_recs):
    """Generate a personalized movie recommendation message using Ollama."""

    # Format recommendations for LLM
    rec_list = "\n".join(
        f"- {row.Title} ({row.Genres})"
        for _, row in user_recs.iterrows()
    )

    # System prompt (behavior + constraints)
    system_prompt = (
        "You are a warm, friendly, and emotionally intelligent movie recommender for the platform WatchIt. "
        "Your goal is to make the user feel understood, excited, and personally connected to their movie recommendations. "
        "Carefully adapt your language, tone, and vocabulary to match the user’s age, keeping the message natural and easy to understand. "
        "Gently acknowledge the user’s lifestyle, daily routine, or occupation, and naturally connect it to the mood or themes of the recommended genres. "
        "Output requirements: write a short, personalized message of exactly 2–3 sentences. "
        "Each sentence must appear on its own line. "
        "DO NOT PRINT ZIP CODE IN THE OUTPUT RECOMMENDATION MESSAGE."
        "USE bullet points for TOP 3 recommendations!"
    )

    # User prompt with decoded attributes
    user_prompt = f"""
    Write a short personalized movie recommendation message using the information below.

    --- USER PROFILE ---
    Age Group: {user_row['Age_Text']}
    Gender: {user_row['Gender_Text']}
    Occupation: {user_row['Occupation_Text']}
    Zip Code: {user_row['Zip-code']}

    --- RECOMMENDED MOVIES ---
    {rec_list}

    --- MESSAGE ---
    """

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.8, "num_ctx": 4096}
        )

        return response["message"]["content"].strip()

    except Exception as e:
        return f"LLM Error: {e}"


# ====================================================
# Generate Message for a Random User
# ====================================================
def generate_for_random_user(recs_df, users_df):
    """Select a random user and generate a personalized message."""

    user_row = users_df.sample(1).iloc[0]
    user_id = user_row['UserID']

    user_recs = (
        recs_df[recs_df['UserID'] == user_id]
        .sort_values('Predicted_Rating', ascending=False)
        .head(TOP_K_LLM_RECS)
    )

    if user_recs.empty:
        return None, None, "No recommendations found for this user."

    message = generate_personalized_message(user_row, user_recs)

    return user_row, user_recs, message


# ====================================================
# Example Usage
# ====================================================
if __name__ == "__main__":

    try:
        recs_df, users_df = load_data()

        user_row, user_recs, message = generate_for_random_user(recs_df, users_df)

        print("\n" + "=" * 85)
        print(
            f"PERSONALIZED MESSAGE FOR USER {user_row['UserID']} | "
            f"age={user_row['Age_Text']}, "
            f"gender={user_row['Gender_Text']}, "
            f"occupation={user_row['Occupation_Text']}, "
            f"zip-code={user_row['Zip-code']}"
        )
        print("=" * 85)

        print("\nMovies used for personalization:")
        for _, row in user_recs.iterrows():
            print(f"  • {row.Title} ({row.Genres}) — predicted {row.Predicted_Rating:.2f}")

        print("\nLLM Personalized Message:\n")
        print(message)

        print("\n" + "=" * 85)

    except FileNotFoundError as e:
        print(f"❌ Missing file: {e.filename}")
    except ImportError:
        print("❌ The 'ollama' package is not installed. Run: pip install ollama")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# =============================== END ===============================