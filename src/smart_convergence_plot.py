import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. SHARED DATA PIPELINE
# ==========================================
def load_and_prep_data():
    print("--- Loading Data ---")
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    
    # Map IDs
    user_map = {u: i for i, u in enumerate(ratings['userId'].unique())}
    item_map = {m: i for i, m in enumerate(ratings['movieId'].unique())}
    ratings['user_idx'] = ratings['userId'].map(user_map)
    ratings['movie_idx'] = ratings['movieId'].map(item_map)
    
    n_users, n_items = len(user_map), len(item_map)
    
    # Shared Split
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # Matrix for ALS
    M_train = np.zeros((n_users, n_items))
    mask_train = np.zeros((n_users, n_items), dtype=bool)
    for row in train_df.itertuples():
        M_train[row.user_idx, row.movie_idx] = row.rating
        mask_train[row.user_idx, row.movie_idx] = True
        
    return train_df, test_df, M_train, mask_train, n_users, n_items

def get_rmse_vectorized(true_ratings, preds):
    # Vectorized RMSE calculation for speed
    preds_clipped = np.clip(preds, 0.5, 5.0)
    return np.sqrt(np.mean((true_ratings - preds_clipped)**2))

# ==========================================
# 2. SMART ALS (History Tracking)
# ==========================================
def run_smart_als_history(M, mask, test_df, n_users, n_items, rank=12, reg=10.0, bias_reg=25.0, iters=15):
    history = []
    
    # A. Pre-calculate Damped Biases
    global_mean = M[mask].mean()
    b_i = np.zeros(n_items)
    b_u = np.zeros(n_users)
    
    # Vectorized Bias Calc
    # (Doing this inside loop for clarity, but logic is "smart")
    for i in range(n_items):
        idx = mask[:, i]
        if idx.sum() > 0:
            b_i[i] = np.sum(M[idx, i] - global_mean) / (idx.sum() + bias_reg)
    for u in range(n_users):
        idx = mask[u, :]
        if idx.sum() > 0:
            b_u[u] = np.sum(M[u, idx] - global_mean - b_i[idx]) / (idx.sum() + bias_reg)

    # B. Calculate Residuals
    M_resid = np.zeros_like(M)
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        M_resid[r, c] = M[r, c] - (global_mean + b_u[r] + b_i[c])

    # C. ALS Loop
    U = np.random.normal(scale=1./rank, size=(n_users, rank))
    V = np.random.normal(scale=1./rank, size=(n_items, rank))
    lambda_I = reg * np.eye(rank)
    
    # Pre-extract Test Arrays for Fast Scoring
    test_u_idx = test_df['user_idx'].values
    test_i_idx = test_df['movie_idx'].values
    test_y_true = test_df['rating'].values
    
    print("Training Smart ALS...")
    for it in range(iters):
        # Update U
        VtV = np.dot(V.T, V)
        for i in range(n_users):
            idx = mask[i, :]
            if idx.sum() > 0:
                V_sub = V[idx, :]
                A = np.dot(V_sub.T, V_sub) + lambda_I
                b = np.dot(V_sub.T, M_resid[i, idx])
                U[i, :] = np.linalg.solve(A, b)
        
        # Update V
        for j in range(n_items):
            idx = mask[:, j]
            if idx.sum() > 0:
                U_sub = U[idx, :]
                A = np.dot(U_sub.T, U_sub) + lambda_I
                b = np.dot(U_sub.T, M_resid[idx, j])
                V[j, :] = np.linalg.solve(A, b)
        
        # Fast Score at end of iteration
        # Prediction = mu + bu + bi + (U . V)
        interaction = np.sum(U[test_u_idx] * V[test_i_idx], axis=1)
        preds = global_mean + b_u[test_u_idx] + b_i[test_i_idx] + interaction
        rmse = get_rmse_vectorized(test_y_true, preds)
        history.append(rmse)
        print(f"  ALS Iter {it+1}: {rmse:.4f}")
        
    return history

# ==========================================
# 3. SMART SGD (History Tracking)
# ==========================================
def run_smart_sgd_history(train_df, test_df, n_users, n_items, rank=12, lr=0.02, reg=0.1, decay=0.02, epochs=30):
    history = []
    
    # Init
    mu = train_df['rating'].mean()
    b_u = np.zeros(n_users)
    b_i = np.zeros(n_items)
    P = np.random.normal(scale=1./rank, size=(n_users, rank))
    Q = np.random.normal(scale=1./rank, size=(n_items, rank))
    
    # Pre-extract Arrays
    u_vec = train_df['user_idx'].values
    i_vec = train_df['movie_idx'].values
    r_vec = train_df['rating'].values
    indices = np.arange(len(u_vec))
    
    test_u_idx = test_df['user_idx'].values
    test_i_idx = test_df['movie_idx'].values
    test_y_true = test_df['rating'].values
    
    print("Training Smart SGD...")
    for epoch in range(epochs):
        # Decay Learning Rate
        current_lr = lr / (1 + decay * epoch)
        np.random.shuffle(indices)
        
        # Fast Loop (Optimized somewhat)
        for idx in indices:
            u, i, r = u_vec[idx], i_vec[idx], r_vec[idx]
            
            dot = np.dot(P[u], Q[i])
            pred = mu + b_u[u] + b_i[i] + dot
            err = r - pred
            
            # Update Biases (Regularized)
            b_u[u] += current_lr * (err - reg * b_u[u])
            b_i[i] += current_lr * (err - reg * b_i[i])
            
            # Update Factors (Regularized)
            p_old = P[u].copy()
            P[u] += current_lr * (err * Q[i] - reg * P[u])
            Q[i] += current_lr * (err * p_old - reg * Q[i])
            
        # Fast Score at end of epoch
        interaction = np.sum(P[test_u_idx] * Q[test_i_idx], axis=1)
        preds = mu + b_u[test_u_idx] + b_i[test_i_idx] + interaction
        rmse = get_rmse_vectorized(test_y_true, preds)
        history.append(rmse)
        print(f"  SGD Epoch {epoch+1}: {rmse:.4f}")
        
    return history

# ==========================================
# 4. PLOTTING
# ==========================================
if __name__ == "__main__":
    # 1. Prep
    train_df, test_df, M, mask, n_users, n_items = load_and_prep_data()
    iterations = 30
    # 2. Run Smart Models
    # Note: SGD needs more epochs (30) to catch up to ALS (15)
    als_hist = run_smart_als_history(M, mask, test_df, n_users, n_items, 
                                     rank=50, reg=10.0, bias_reg=10.0, iters=iterations)
    
    sgd_hist = run_smart_sgd_history(train_df, test_df, n_users, n_items, 
                                     rank=50, lr=0.02, reg=0.05, decay=0.02, epochs=iterations)
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    
    # Plot ALS
    plt.plot(range(1, len(als_hist)+1), als_hist, 
             label='Smart ALS (Damped Bias)', color='blue', linewidth=2, marker='o')
    
    # Plot SGD
    plt.plot(range(1, len(sgd_hist)+1), sgd_hist, 
             label='Smart SGD (Decay + Reg)', color='orange', linewidth=2, linestyle='--')
    plt.axhline(0.8762, color='green', linestyle='-', linewidth=1.5, label = 'Column Mean')
    plt.title("Convergence Showdown: Smart ALS vs Smart SGD")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Test RMSE (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig("smartplot.png")