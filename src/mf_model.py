# # #
# Matrix Factorization using SGD
# Comparison for different latent dimensions K
# # #

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Data loading
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Map userId and movieId to 0-based indices
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()

user_to_index = {u: i for i, u in enumerate(user_ids)}
movie_to_index = {m: i for i, m in enumerate(movie_ids)}

ratings["user_idx"] = ratings["userId"].map(user_to_index)
ratings["movie_idx"] = ratings["movieId"].map(movie_to_index)

n_users = len(user_ids)
n_movies = len(movie_ids)

# Train/test split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Hyperparameters
K_values = [10, 20, 50]
alpha = 0.005
lambda_reg = 0.02
epochs = 25

# Global mean (same for all runs)
mu = train_data["rating"].mean()

baseline_rmse = np.sqrt(np.mean((test_data["rating"] - mu) ** 2))
print(f"Baseline RMSE (global mean): {baseline_rmse:.4f}\n")

# RMSE storage
rmse_train_all = {}
rmse_test_all = {}

# Helper function
def compute_rmse(data, P, Q, b_u, b_i):
    errors = []
    for _, row in data.iterrows():
        u = int(row["user_idx"])
        i = int(row["movie_idx"])
        r = row["rating"]
        pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
        errors.append((r - pred) ** 2)
    return np.sqrt(np.mean(errors))

# Training loop for each K
for K in K_values:
    print(f"Training model with K = {K}")

    # Initialize parameters
    P = np.random.normal(scale=1.0 / K, size=(n_users, K))
    Q = np.random.normal(scale=1.0 / K, size=(n_movies, K))
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_movies)

    rmse_train_history = []
    rmse_test_history = []

    for epoch in range(epochs):
        squared_error = 0.0

        for _, row in train_data.iterrows():
            u = int(row["user_idx"])
            i = int(row["movie_idx"])
            r = row["rating"]

            pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
            err = r - pred
            squared_error += err ** 2

            # Update biases
            b_u[u] += alpha * (err - lambda_reg * b_u[u])
            b_i[i] += alpha * (err - lambda_reg * b_i[i])

            # Update latent factors
            P[u] += alpha * (err * Q[i] - lambda_reg * P[u])
            Q[i] += alpha * (err * P[u] - lambda_reg * Q[i])

        rmse_train = np.sqrt(squared_error / len(train_data))
        rmse_test = compute_rmse(test_data, P, Q, b_u, b_i)

        rmse_train_history.append(rmse_train)
        rmse_test_history.append(rmse_test)

        print(
            f"  Epoch {epoch+1}/{epochs} | "
            f"Train RMSE: {rmse_train:.4f} | "
            f"Test RMSE: {rmse_test:.4f}"
        )

    # Store results
    rmse_train_all[K] = rmse_train_history
    rmse_test_all[K] = rmse_test_history

    # Save per-K results
    np.save(f"rmse_train_K{K}.npy", rmse_train_history)
    np.save(f"rmse_test_K{K}.npy", rmse_test_history)

    print(f"Finished training for K = {K}\n")

# Final comparison
print("Final Test RMSE per K:")
for K in K_values:
    print(f"K = {K}: {rmse_test_all[K][-1]:.4f}")
