# # AI-POWERED MOVIE RECOMMENDATION ENGINE
# ## INPUT DATA PREPARATION - USERS
# ### Author: Evgeniya Englert
# ### Last update: 2026-01-12
# ## Summary
# This notebook prepares user dataset for use in a recommendation engine. 
# 
# I use dataset MovieLens which consists of movies, ratings and users data.
# 
# These files contain 1,000,209 anonymous ratings of approximately 3,900 movies 
# made by 6,040 MovieLens users who joined MovieLens in 2000.
# 
# * Original input dataset used:
#     * https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset
# 
# The cleaned and prepared user data will be saved as user_data.csv, serving as the foundation for exploring user similarities in a later stage.
# 
# For the further training of the recommendation system, let's assume, all these consumers made a purchase.
# 
# Attributes: User data:
# * UserID::Gender::Age::Occupation (Job)::Zip-code
# 
# - Gender is denoted by a "M" for male and "F" for female
# - Age is chosen from the following ranges:
# 
# 	*  1:  "Under 18"
# 	* 18:  "18-24"
# 	* 25:  "25-34"
# 	* 35:  "35-44"
# 	* 45:  "45-49"
# 	* 50:  "50-55"
# 	* 56:  "56+"
# 
# - Occupation is chosen from the following choices:
# 
# 	*  0:  "other" or not specified
# 	*  1:  "academic/educator"
# 	*  2:  "artist"
# 	*  3:  "clerical/admin"
# 	*  4:  "college/grad student"
# 	*  5:  "customer service"
# 	*  6:  "doctor/health care"
# 	*  7:  "executive/managerial"
# 	*  8:  "farmer"
# 	*  9:  "homemaker"
# 	* 10:  "K-12 student"
# 	* 11:  "lawyer"
# 	* 12:  "programmer"
# 	* 13:  "retired"
# 	* 14:  "sales/marketing"
# 	* 15:  "scientist"
# 	* 16:  "self-employed"
# 	* 17:  "technician/engineer"
# 	* 18:  "tradesman/craftsman"
# 	* 19:  "unemployed"
# 	* 20:  "writer"
# 
# Further information on the GroupLens Research project, including research 
# publications, can be found at the following web site:
# * http://www.grouplens.org/
# 
# GroupLens Research currently operates a movie recommender based on 
# collaborative filtering:
# * http://www.movielens.org/
# 
# Author: Evgeniya Englert
# 
# Last Update: 2025-09-04

# %%
# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# %% [markdown]
# Define functions that we'll need in the code

# %%
# Create a function to check a data frame for missings and print a message
def fct_print_missings_check(df, df_name="DataFrame"):
    """
    Checks a DataFrame for any missing values and prints a status message.
    It prints only the columns that contain missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        df_name (str): The name of the DataFrame for the output message.
    """
    # Sum the missing values for each column
    missing_counts = df.isnull().sum()
    
    # Filter to get only the columns with more than 0 missing values
    columns_with_missings = missing_counts[missing_counts > 0]
    
    # Get the total number of missing values across the entire DataFrame
    total_missing_count = columns_with_missings.sum()
    
    if total_missing_count > 0:
        print(f"❗ Check failed for '{df_name}': The DataFrame has {total_missing_count} missing value(s).")
        print(f" Columns with missings: \n{columns_with_missings}")
    else:
        print(f"✅ Check passed for '{df_name}': The DataFrame has no missing values.")

# %% [markdown]
# Import and explore user data

# %%
# Read CSV
column_names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
users_df = pd.read_csv('users.dat', sep='::', engine='python', names=column_names)

# %%
users_df.head()

# %%
# Expolre data set
print(users_df.dtypes)

# %%
# Expolre adjusted data set: No missings
users_df.info()

# %%
# Are there any nulls in the data set?
fct_print_missings_check(users_df)

# %% [markdown]
# Check features

# %%
# Age: Check the "age" feature in a DataFrame and visualize it with a histogram using Seaborn
print("\n------------------------------------------")

# Use .describe() to get descriptive statistics of the 'age' column.
# This provides valuable insights like mean, standard deviation, min, max, and quartiles.
print("\nDescriptive Statistics for 'age' feature:")
print(users_df['Age'].describe())

# --- Step 2: Create a histogram using Seaborn ---
# Set the style for the plot to make it visually appealing.
sns.set_style("whitegrid")

# Create a figure and an axes object for the plot.
plt.figure(figsize=(8, 6))

# Generate the histogram for the 'Age' column.
# 'binwidth' controls the width of the bars.
sns.histplot(data=users_df, x='Age', kde=True, binwidth=5, color='skyblue')

# Add a title and labels to the plot for better readability.
plt.title('Distribution of Age', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Display the plot.
plt.show()

# %%
# Gender: Checking the unique values of the 'Gender' categorical feature in a DataFrame
# Check the unique values in the 'Gender' column.
# This is useful to see all the different categories present in the data.
unique_gender_types = users_df['Gender'].unique()
print("Unique values in the 'Gender' column:")
print(unique_gender_types)
print("-" * 30)

# Check the frequency of each unique value.
# This gives you a count of how many times each category appears.
gender_counts = users_df['Gender'].value_counts()
print("Counts of each unique Gender type:")
print(gender_counts)

# %% [markdown]
# Change dtype for 'UserID', 'Gender', 'Occupation', 'Zip-code' to 'category'

# %%
users_df['UserID'] = users_df['UserID'].astype('category')
users_df['Occupation'] = users_df['Occupation'].astype('category')
users_df['Gender'] = users_df['Gender'].astype('category')
users_df['Zip-code'] = users_df['Zip-code'].astype('category')

# %%
users_df.dtypes

# %%
# Number of unique users
print(f"Number of unique users: {users_df['UserID'].nunique()}")

# %% [markdown]
# Save transformed user data as a csv-file

# %%
# Print the DataFrame to show what we are saving
print("Original DataFrame:")
print(users_df)
print("-" * 30)

# 2. Save the DataFrame to a CSV file named 'consumer_data.csv'
# The .to_csv() method is the standard way to do this.
# The 'index=False' argument is a best practice to prevent saving
# the DataFrame's row index as an extra column in the CSV file.
try:
    users_df.to_csv('user_data.csv', index=False)
    print("DataFrame successfully saved to 'user_data.csv'")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")

# Note: The file will be created in the same directory where this script is run.

# %% [markdown]
# Clustering with K-Means

# %%
# --------------------
# Define Features (X)
# --------------------
# In unsupervised learning, there is no target variable. We only use features.
# Features used: 'age', 'job', 'marital', 'education', 'housing'
X = users_df[['Gender', 'Age', 'Occupation', 'Zip-code']]

# --------------------
# Preprocessing: Define column types for transformation
# --------------------
# K-Means is a distance-based algorithm, so it's crucial to prepare the data.
# We'll scale numeric features and one-hot encode categorical features.

# Automatically identify numeric and categorical features based on data types
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='category').columns.tolist()

print(f"Automatically identified numeric features: {numeric_features}")
print(f"Automatically identified categorical features: {categorical_features}")

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --------------------
# Create a Machine Learning Pipeline
# --------------------
# The pipeline uses KMeans for classification.
# We've set the number of clusters to 5 for this example.
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('clusterer', KMeans(n_clusters=5, random_state=42, n_init=10))])

# --------------------
# Fit the Model 
# --------------------
# We fit the model on the entire dataset to find the clusters.
print("Fitting the K-Means model...")
users_df['cluster_assignment'] = model.fit_predict(X)
print("Clustering complete.")

# --------------------
# View the Results
# --------------------
# Display the DataFrame with the new cluster assignments.
print("\nDataFrame with Cluster Assignments:")
print(users_df)

# Get the centroids of the clusters in the scaled feature space
scaled_data = preprocessor.fit_transform(X)
centroids = model.named_steps['clusterer'].cluster_centers_

# %% [markdown]
# Check cluster distribution

# %%
# User clusters: Checking the values of 'cluster_assignment'
unique_clusters = users_df['cluster_assignment'].unique()
sorted_unique_clusters = sorted(unique_clusters)
print("Unique values in the 'cluster_assignment' column:")
print(sorted_unique_clusters)
print("-" * 30)

# Check the frequency of each unique value.
# This gives you a count of how many times each category appears.
cluster_counts = users_df['cluster_assignment'].value_counts()
print("Counts of each unique cluster_assignment:")
print(cluster_counts)

# %%
# --------------------
# Visualize the Clusters
# --------------------
# To visualize the clusters in 2D, we first need to reduce the dimensionality
# of the data using Principal Component Analysis (PCA).
print("\nVisualizing the clusters using PCA:")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['cluster_assignment'] = users_df['cluster_assignment']

# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='cluster_assignment', data=pca_df,
                palette='viridis', style='cluster_assignment', s=100)
plt.title('Consumer Clusters (Visualized with PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# %%
# Example of a new users's cluster prediction
# 'Gender', 'Age', 'Occupation', 'Zip-code'
new_user = pd.DataFrame({
    'Gender': ['F'],
    'Age': [20],
    'Occupation': ['20'],
    'Zip-code': ['55455']
})

cluster_prediction = model.predict(new_user)
print(f"\nCluster prediction for a new consumer: Cluster {cluster_prediction[0]}")

# %%
# Save the clusters to a CSV file
users_df.to_csv('user_clusters.csv')
print("\nDataFrame generated successfully.")


