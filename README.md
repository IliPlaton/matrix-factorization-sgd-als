# Optimization of Regularized Matrix Factorization for Movie Recommendation Systems

## Project Overview
User-item rating matrices in recommender systems are typically extremely sparse. This project addresses the task of predicting missing ratings as a matrix completion and optimization problem to deliver personalized recommendations. The goal is to learn a low-rank approximation that minimizes prediction error on unseen ratings using the MovieLens Latest Small dataset (~100K ratings, 600 users, 9K movies).

The objective function minimizes the squared error on observed ratings alongside $L_2$ regularization terms to control model complexity:

$$\min_{U,V}\sum_{(i,j)\in\Omega}(R_{ij}-u_{i}v_{j}^{\top})^{2}+\lambda(||u_{i}||^{2}+||v_{j}||^{2})$$

The implementation includes global, user, and item bias terms, comparing Stochastic Gradient Descent (SGD) and Alternating Least Squares (ALS) optimization solvers.

---

## Methodology and Solvers
* **Stochastic Gradient Descent (SGD):** Offers flexibility and fine-grained optimization control, converging steadily across different hyperparameters.
* **Alternating Least Squares (ALS):** Provides a strong baseline for comparison but scales less efficiently for dense hyperparameter tuning.
* **Evaluation Metric:** Models are evaluated using Root Mean Squared Error (RMSE) on an 80-20 train-test split against global mean and column-based baseline predictors.

---

## Hyperparameter Tuning and Evaluation

To analyze the performance of the SGD solver, experimentation was conducted across multiple latent dimensions ($K=10$, $K=20$, and $K=50$) over 25 training epochs.

### Key Insights

* **Steady Convergence:** The SGD solver yields smooth and consistent error reduction across all evaluated values of $K$.
* **Bias-Variance Trade-off:** Increasing the latent dimension $K$ allows the model to capture more complex user-item interactions, which minimizes training error. However, a larger capacity inherently risks overfitting.
* **Generalization Gap:** The divergence between training and testing performance grows over successive epochs. This gap underlines the necessity of tuning the regularization parameter ($\lambda$) to maintain performance on unseen data.

For a comprehensive review of the findings, implementation specifics, and the baseline comparisons, refer to the full project poster: `CS573_poster_final.pdf`.

