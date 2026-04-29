# collaborative-filtering-nmf

A machine learning project that predicts missing Netflix movie ratings using two collaborative filtering approaches: **Singular Value Decomposition (SVD)** and **Non-Negative Matrix Factorization (NMF)**.

## Overview

This project addresses the sparse data problem in recommendation systems by completing an incomplete user-item rating matrix. With 92% sparsity (14,328 users, 150 movies), the goal is to fill missing entries and generate personalized movie recommendations.

## Objectives

-   Fill missing entries in the user-item rating matrix using collaborative filtering
-   Compare prediction accuracy between SVD and NMF methods
-   Generate personalized movie recommendations for users
-   Output complete rating matrices for downstream analysis

## Methodology

### Data Preparation

-   **Dataset**: 14,328 users × 150 movies rating matrix
-   **Sparsity**: 92.05% (only 170,814 known ratings out of 2,149,200 total entries)
-   **Imputation**: Missing values filled with column means as a baseline

### Collaborative Filtering Approaches

#### 1. **SVD (Singular Value Decomposition)**

The SVD method decomposes the centered rating matrix into: - **U**: User-feature matrix (m × k) - **Σ**: Singular values (k × k diagonal) - **V**: Item-feature matrix (k × n)

The low-rank approximation uses **k=15 latent features** to capture the most important patterns in the data.

```         
R ≈ U_k @ Σ_k @ V_k^T + column_means
```

**RMSE on known ratings: 0.712498**

#### 2. **NMF (Non-Negative Matrix Factorization)**

NMF decomposes the rating matrix as a product of two non-negative matrices: - **P**: User feature matrix (m × k) - **Q**: Item feature matrix (k × n)

**Matrix Decomposition:**

```         
R ≈ P @ Q
```

##### Mathematical Formulation

**Loss Function:**

Minimize the Frobenius norm of the error on known ratings:

$$J = \frac{1}{2} \|R_{\text{known}} - (PQ)_{\text{known}}\|_F^2 = \frac{1}{2} \sum_{(i,j) \in \Omega} (R_{ij} - (PQ)_{ij})^2$$

where Ω is the set of known ratings and $\|\cdot\|_F$ denotes the Frobenius norm.

**Gradient Derivations:**

Define the error matrix: $$Y = R - PQ$$

Then the partial derivatives of J with respect to P and Q are:

$$\frac{\partial J}{\partial P_{ik}} = -\sum_j Y_{ij} Q_{kj} = -(YQ^T)_{ik}$$

$$\frac{\partial J}{\partial Q_{kj}} = -\sum_i P_{ik} Y_{ij} = -(P^T Y)_{kj}$$

**Gradient Expressions:**

$$\nabla_P J = -YQ^T = -(R - PQ)Q^T = PQQ^T - RQ^T$$

$$\nabla_Q J = -P^T Y = -P^T(R - PQ) = P^T PQ - P^T R$$

**Multiplicative Update Rules:**

To maintain non-negativity (crucial for the rating interpretation), we use multiplicative update rules rather than standard gradient descent:

$$P \leftarrow P \odot \frac{RQ^T}{PQQ^T + \epsilon}$$

$$Q \leftarrow Q \odot \frac{P^T R}{P^T PQ + \epsilon}$$

where $\odot$ denotes element-wise multiplication and $\epsilon$ is a small regularization constant ($10^{-12}$) to prevent division by zero.

**Derivation of Update Rules:**

The multiplicative updates emerge from ensuring that each step reduces the objective while maintaining non-negativity:

$$P_{ik}^{(t+1)} = P_{ik}^{(t)} \cdot \frac{(RQ^T)_{ik}}{(PQQ^T)_{ik} + \epsilon}$$

$$Q_{kj}^{(t+1)} = Q_{kj}^{(t)} \cdot \frac{(P^T R)_{kj}}{(P^T PQ)_{kj} + \epsilon}$$

These updates are designed so that: 1. Numerators contain terms that drive factorization accuracy (actual data) 2. Denominators contain predicted reconstruction terms (regularization) 3. Element-wise multiplication naturally preserves non-negativity

**Algorithm Summary:**

```         
Initialize P, Q with positive random values
For t = 1 to max_iterations:
    # Update user features
    P ← P ⊙ (R·Q^T) / (P·Q·Q^T + ε)
    
    # Update item features  
    Q ← Q ⊙ (P^T·R) / (P^T·P·Q + ε)
    
    # Compute error on known ratings
    RMSE ← sqrt(mean((R - P·Q)²)[known mask])
    
    if RMSE < threshold:
        break
```

**Convergence:** - Converged at iteration 278 - **RMSE on known ratings: 0.397192** - Final convergence criterion: RMSE \< 0.4

### Results Comparison

| Metric               | SVD      | NMF               |
|----------------------|----------|-------------------|
| RMSE (known ratings) | 0.712498 | 0.397192          |
| Improvement          | —        | **44.25% better** |

**NMF significantly outperforms SVD** by capturing non-linear patterns and maintaining non-negativity constraints that align with rating semantics.

#### Why NMF Outperforms SVD

1.  **Non-negativity Constraint**: NMF factors P and Q are inherently non-negative, which:

    -   Aligns with the 1-5 rating scale interpretation
    -   Reduces out-of-range predictions (SVD often produces negative or \>5 values)
    -   Provides interpretability: factors represent "strength of preference"

2.  **Loss Function Design**: NMF optimizes only on known ratings via the masked Frobenius norm:

    -   $$J = \frac{1}{2} \|M \odot (R - PQ)\|_F^2$$ where M is the mask
    -   Directly targets accuracy where it matters
    -   SVD reconstructs the entire matrix, less flexible for sparse data

3.  **Gradient Structure**: Multiplicative updates maintain non-negativity:

    -   Standard gradient descent on unrestricted factors would violate constraints
    -   Multiplicative updates: $X_{ij} \leftarrow X_{ij} \cdot \frac{\text{numerator}}{\text{denominator}}$
    -   This preserves positivity: if $X_{ij} > 0$, it stays positive

4.  **Sparsity-Aware Optimization**: NMF uses masked computations:

    ``` python
    R_masked = R * mask  # Zero out unknowns before factorization
    ```

    -   Prevents overfitting to imputed values
    -   Focuses model capacity on observed user preferences

5.  **Feature Learning**: With k=15 latent features:

    -   SVD: Forced orthogonality limits expressive power for non-linear structure
    -   NMF: Free to learn correlated factors that better capture preference patterns
    -   Results show more natural clustering of users and items

## Files Generated

-   `netflix_ratings_svd_filled.csv` - Complete ratings using SVD predictions
-   `netflix_ratings_nmf_filled.csv` - Complete ratings using NMF predictions (recommended)

## Usage

### Prerequisites

``` python
import numpy as np
import pandas as pd
```

### Input Files Required

-   `netflix_rating.csv` - User-item rating matrix (User_ID as index, Movie_IDs as columns)
-   `netflix_movie.csv` - Movie metadata with columns: Movie_ID, Year, Name

### Running the Notebook

1.  Load data and inspect sparsity
2.  Perform SVD-based collaborative filtering (k=15 features)
3.  Perform NMF-based collaborative filtering with gradient descent
4.  Compare RMSE metrics across methods
5.  Generate personalized recommendations for sample users
6.  Export completed rating matrices to CSV

### Example Output

```         
Movie Recommendations for User 6:
  1. Viva La Bam: Season 1 (Predicted Rating: 5.00/5.0)
  2. Husbands and Wives (Predicted Rating: 5.00/5.0)

Movie Recommendations for User 7:
  1. Kill Bill: Vol. 2 (Predicted Rating: 5.00/5.0)
  2. Herbie Rides Again (Predicted Rating: 5.00/5.0)
```

## Technical Details

### SVD Implementation

-   Center the rating matrix by column means
-   Apply full SVD decomposition
-   Truncate to k=15 singular values
-   Reconstruct and add back column means

### NMF Implementation

#### Initialization

Initialize P and Q with small random positive values to break symmetry:

``` python
P = np.random.rand(m, k) + 1e-3   # m users × k features
Q = np.random.rand(k, n) + 1e-3   # k features × n items
```

The small offset (1e-3) prevents zero initialization which would lead to division by zero in update rules.

#### Multiplicative Update Step

**Update P (User Feature Matrix):**

``` python
numerator_P = R_masked @ Q.T              # (m×k) = (m×n) @ (n×k)
denominator_P = ((P @ Q) * mask) @ Q.T + ε  # (m×k)
P *= (numerator_P / denominator_P)        # Element-wise multiplication
```

Breaking down the denominater: - `P @ Q` produces the reconstruction (m×n matrix) - `(P @ Q) * mask` zeros out unknown entries - Multiplying by Q\^T gives (m×k) shape

**Update Q (Item Feature Matrix):**

``` python
numerator_Q = P.T @ R_masked              # (k×n) = (k×m) @ (m×n)
denominator_Q = P.T @ ((P @ Q) * mask) + ε  # (k×n)
Q *= (numerator_Q / denominator_Q)        # Element-wise multiplication
```

### Evaluation Metric

**Root Mean Squared Error (RMSE)** on known ratings:

$$\text{RMSE} = \sqrt{\frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (R_{ij} - \hat{R}_{ij})^2}$$

where: - Ω is the set of known (observed) ratings - $R_{ij}$ is the true rating - $\hat{R}_{ij}$ is the predicted rating - $|\Omega| = 170,814$ known ratings

**Implementation:**

``` python
def compute_nmf_rmse(R, P, Q, mask):
    R_pred = P @ Q
    error = np.sqrt(np.mean((R_pred[mask] - R[mask]) ** 2))
    return error
```

This metric directly measures prediction accuracy on observed data.

