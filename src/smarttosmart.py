import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
# from mf_code import *
# from mf_model import *
# ==========================================
# 1. SHARED DATA PIPELINE (Apples-to-Apples)
# ==========================================

def load_and_split_data():
    print("--- Loading and Splitting Data ---")
    # Load Data (Assuming ml-latest-small is available as per your scripts)
    # Using the pandas method from your SGD script as it's cleaner for splitting
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    
    # Map to 0-based indices
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {m: i for i, m in enumerate(movie_ids)}
    
    ratings['user_idx'] = ratings['userId'].map(user_map)
    ratings['movie_idx'] = ratings['movieId'].map(item_map)
    
    n_users = len(user_ids)
    n_items = len(movie_ids)
    
    # EXACT SAME SPLIT FOR BOTH
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # Create Sparse-like Matrix for ALS (using the train_df only!)
    M_train = np.zeros((n_users, n_items))
    mask_train = np.zeros((n_users, n_items))
    
    for row in train_df.itertuples():
        M_train[row.user_idx, row.movie_idx] = row.rating
        mask_train[row.user_idx, row.movie_idx] = 1
        
    print(f"Data Prepared: {n_users} Users, {n_items} Items")
    print(f"Train Size: {len(train_df)}, Test Size: {len(test_df)}")
    
    return train_df, test_df, M_train, mask_train, n_users, n_items

def evaluate_model(model_name, true_ratings, predictions, duration):
    # CRITICAL: Both models must be Clipped [0.5, 5.0]
    pred_clipped = np.clip(predictions, 0.5, 5.0)
    
    mse = np.mean((true_ratings - pred_clipped) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"[{model_name}] RMSE: {rmse:.4f} | Time: {duration:.2f}s")
    return rmse

# ==========================================
# 2. ALS IMPLEMENTATION (From your code)
# ==========================================

def compute_biases_damped(M, mask, lambda_reg=25.0):
    # --- FIX START ---
    # Ensure mask is boolean (True/False) instead of float (1.0/0.0)
    # This fixes the "IndexError: arrays used as indices..."
    mask = mask.astype(bool)
    # --- FIX END ---

    global_mean = np.sum(M[mask]) / np.sum(mask)
    n_users, n_items = M.shape
    
    b_i = np.zeros(n_items)
    for i in range(n_items):
        # Now we can just use the boolean mask directly
        idx = mask[:, i] 
        if idx.sum() > 0:
            b_i[i] = np.sum(M[idx, i] - global_mean) / (idx.sum() + lambda_reg)
            
    b_u = np.zeros(n_users)
    for u in range(n_users):
        idx = mask[u, :] 
        if idx.sum() > 0:
            b_u[u] = np.sum(M[u, idx] - global_mean - b_i[idx]) / (idx.sum() + lambda_reg)
            
    return global_mean, b_u, b_i
def run_als(M_train, mask_train, test_df, n_users, n_items):
    start = time.time()
    
    # Hyperparams (from your successful run)
    rank = 70
    lambda_als = 10.0
    lambda_bias = 10.0
    iterations = 15
    
    # 1. Compute Biases
    mu, b_u, b_i = compute_biases_damped(M_train, mask_train, lambda_reg=lambda_bias)
    
    # 2. Compute Residuals
    M_resid = np.zeros_like(M_train)
    rows, cols = np.where(mask_train)
    for r, c in zip(rows, cols):
        M_resid[r, c] = M_train[r, c] - (mu + b_u[r] + b_i[c])
        
    # 3. Train Factors
    U = np.random.normal(scale=1./rank, size=(n_users, rank))
    V = np.random.normal(scale=1./rank, size=(n_items, rank))
    lambda_I = lambda_als * np.eye(rank)
    rmses = []
    for _ in range(iterations):
        # Update U
        VtV = np.dot(V.T, V)
        for i in range(n_users):
            idx = mask_train[i, :] == 1
            if idx.sum() > 0:
                V_sub = V[idx, :]
                A = np.dot(V_sub.T, V_sub) + lambda_I
                b = np.dot(V_sub.T, M_resid[i, idx])
                U[i, :] = np.linalg.solve(A, b)
        # Update V
        for j in range(n_items):
            idx = mask_train[:, j] == 1
            if idx.sum() > 0:
                U_sub = U[idx, :]
                A = np.dot(U_sub.T, U_sub) + lambda_I
                b = np.dot(U_sub.T, M_resid[idx, j])
                V[j, :] = np.linalg.solve(A, b)

    # 4. Predict on Test Set
    test_users = test_df['user_idx'].values
    test_items = test_df['movie_idx'].values
    true_ratings = test_df['rating'].values
    
    # Vectorized prediction
    # Pred = mu + bu + bi + u.v
    # (Using simple loop to avoid memory issues with massive broadcasting)
    preds = []
    for u, i in zip(test_users, test_items):
        pred = mu + b_u[u] + b_i[i] + np.dot(U[u], V[i])
        preds.append(pred)
        
    return evaluate_model("ALS (Damped)", true_ratings, np.array(preds), time.time() - start)

def run_sgd_improved(train_df, test_df, n_users, n_items):
    start = time.time()
    
    # IMPROVEMENT 1: Hyperparameters
    K = 70
    alpha = 0.02          # Higher start rate
    decay = 0.02          # Decay factor
    lambda_reg = 0.05      # Stronger regularization
    epochs = 30           # More epochs (SGD needs time)
    
    # Init
    mu = train_df['rating'].mean()
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)
    P = np.random.normal(scale=1./K, size=(n_users, K))
    Q = np.random.normal(scale=1./K, size=(n_items, K))
    
    users = train_df['user_idx'].values
    items = train_df['movie_idx'].values
    ratings = train_df['rating'].values
    n_samples = len(users)
    indices = np.arange(n_samples)
    
    print(f"Training SGD (Alpha={alpha}, Decay={decay})...")
    
    for epoch in range(epochs):
        # IMPROVEMENT 2: Learning Rate Decay
        current_alpha = alpha / (1 + decay * epoch)
        np.random.shuffle(indices)
        
        for idx in indices:
            u, i, r = users[idx], items[idx], ratings[idx]
            
            dot = np.dot(P[u], Q[i])
            pred = mu + b_u[u] + b_i[i] + dot
            err = r - pred
            
            # IMPROVEMENT 3: Update Biases WITH Regularization
            b_u[u] += current_alpha * (err - lambda_reg * b_u[u])
            b_i[i] += current_alpha * (err - lambda_reg * b_i[i])
            
            # Update Factors
            p_old = P[u].copy()
            P[u] += current_alpha * (err * Q[i] - lambda_reg * P[u])
            Q[i] += current_alpha * (err * p_old - lambda_reg * Q[i])
    # Predict
    test_users = test_df['user_idx'].values
    test_items = test_df['movie_idx'].values
    true_ratings = test_df['rating'].values
    
    preds = []
    for u, i in zip(test_users, test_items):
        pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
        preds.append(pred)
        
    return evaluate_model("SGD (Improved)", true_ratings, np.array(preds), time.time() - start)


# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Prep
    train_df, test_df, M_train, mask_train, n_users, n_items = load_and_split_data()
    
    # 2. Run Apples-to-Apples
    print("\n--- Starting Comparison ---")
    rmse_als = run_als(M_train, mask_train, test_df, n_users, n_items)
    rmse_sgd = run_sgd_improved(train_df, test_df, n_users, n_items)
    
    print("\n--- Verdict ---")
    diff = rmse_sgd - rmse_als
    print(f"Gap: {diff:.4f}")
    print(f"RMSE ALS: {rmse_als:.4f} | RMSE SGD: {rmse_sgd:.4f}")
    if abs(diff) < 0.01:
        print("Result: EFFECTIVELY TIED (Differences are negligible)")
    elif diff > 0:
        print("Result: ALS WINS (Likely due to Damped Bias pre-calculation)")
    else:
        print("Result: SGD WINS")