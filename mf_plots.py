import numpy as np
import matplotlib.pyplot as plt

K_values = [10, 20, 50]

# Train vs Test RMSE AND Gap (per K)

plt.figure(figsize=(12, 5))

# Training vs Test RMSE (per K)
plt.subplot(1, 2, 1)

for K in K_values:
    rmse_train = np.load(f"rmse_train_K{K}.npy")
    rmse_test = np.load(f"rmse_test_K{K}.npy")
    epochs = range(1, len(rmse_train) + 1)

    plt.plot(epochs, rmse_train, label=f"Train (K={K})")
    plt.plot(epochs, rmse_test, linestyle="--", label=f"Test (K={K})")

plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training vs Test RMSE (per K)")
plt.ylim(0.5, 1.5)   # zoomed out y-axis
plt.legend()
plt.grid(True)

# Generalization Gap (per K)
plt.subplot(1, 2, 2)

for K in K_values:
    rmse_train = np.load(f"rmse_train_K{K}.npy")
    rmse_test = np.load(f"rmse_test_K{K}.npy")
    epochs = range(1, len(rmse_train) + 1)
    gap = rmse_test - rmse_train

    plt.plot(epochs, gap, label=f"K={K}")

plt.xlabel("Epoch")
plt.ylabel("RMSE Gap")
plt.title("Generalization Gap (per K)")
plt.ylim(-0.1, 0.6)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# WINDOW 2: Comparison across K
plt.figure(figsize=(12, 5))

# Training RMSE comparison
plt.subplot(1, 2, 1)

for K in K_values:
    rmse_train = np.load(f"rmse_train_K{K}.npy")
    epochs = range(1, len(rmse_train) + 1)
    plt.plot(epochs, rmse_train, label=f"K={K}")

plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training RMSE Comparison Across K")
plt.ylim(0.5, 1.5)
plt.legend()
plt.grid(True)

# Generalization Gap comparison
plt.subplot(1, 2, 2)

for K in K_values:
    rmse_train = np.load(f"rmse_train_K{K}.npy")
    rmse_test = np.load(f"rmse_test_K{K}.npy")
    epochs = range(1, len(rmse_train) + 1)
    gap = rmse_test - rmse_train
    plt.plot(epochs, gap, label=f"K={K}")

plt.xlabel("Epoch")
plt.ylabel("RMSE Gap")
plt.title("Generalization Gap Comparison Across K")
plt.ylim(-0.1, 0.6)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # Latent dimensions used
# K_values = [10, 20, 50]

# for K in K_values:
#     rmse_train = np.load(f"rmse_train_K{K}.npy")
#     rmse_test = np.load(f"rmse_test_K{K}.npy")

#     epochs = range(1, len(rmse_train) + 1)

#     # -------------------------
#     # Plot 1: Training vs Test RMSE
#     # -------------------------
#     plt.figure()
#     plt.plot(epochs, rmse_train, label="Training RMSE")
#     plt.plot(epochs, rmse_test, linestyle="--", label="Test RMSE")
#     plt.xlabel("Epoch")
#     plt.ylabel("RMSE")
#     plt.title(f"Training vs Test RMSE (K = {K})")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # -------------------------
#     # Plot 2: Generalization Gap
#     # -------------------------
#     gap = rmse_test - rmse_train

#     plt.figure()
#     plt.plot(epochs, gap)
#     plt.xlabel("Epoch")
#     plt.ylabel("RMSE Gap (Test - Train)")
#     plt.title(f"Generalization Gap Over Epochs (K = {K})")
#     plt.grid(True)
#     plt.show()
