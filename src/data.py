import pandas as pd

# Load the data
ratings = pd.read_csv("/home/iliana/Documents/ml-latest-small/ratings.csv")
movies  = pd.read_csv("/home/iliana/Documents/ml-latest-small/movies.csv")

# print(ratings.head())
# print(movies.head())

# print("Number of ratings:", ratings.shape[0])
# print("Number of users:", ratings["userId"].nunique())
# print("Number of movies:", ratings["movieId"].nunique())

# Unique ID lists
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

# Create mappings
user_to_index = {u: i for i, u in enumerate(user_ids)}
movie_to_index = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"] = ratings["userId"].map(user_to_index)
ratings["movie_idx"] = ratings["movieId"].map(movie_to_index)

print(ratings.head())

density = ratings.shape[0] / (ratings["userId"].nunique() * ratings["movieId"].nunique())
print("Matrix density:", density)

# Set of observed entries
Omega = list(zip(ratings["user_idx"], ratings["movie_idx"]))