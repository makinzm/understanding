# Reference

- [足立　修一, 丸田　一郎, 「カルマンフィルタの基礎」 - 東京電機大学出版局 2012 ]( https://www.tdupress.jp/book/b349390.html )
- [ Some System Identification Challenges and Approaches - ScienceDirect ]( https://www.sciencedirect.com/science/article/pii/S1474667016386153 )

---

# Preface

Model Based Development was becoming popular. Kalman filter is one of the best known algorithms in system control and estimation.

This method can be applied to various fields, such as robotics, navigation, economics and agriculture.

---

An old filtering problem is defined as follows: we find an algorithm to find noise from the signal.

- Wiener filter is typical example of filter.

A current filtering problem is defined as follows: we estimate the state from the signal.

- Kalman filter is typical example of filter.

---

There are three types of state estimation problems of time $k$:

1. Prediction: Used data range is $[0, k-n]$
2. Filtering: Used data range is $[0, k]$
3. Smoothing: Used data range is $[0, k+n]$

# Modeling

## Time-series modeling

- Model: AR / MA / ARMA / ARIMA models
- Problem: Can this noise be represented as a filtered white noise?
    - → Spectral factorization problem: Find a causal, stable filter $H(z)$ such that $S(\omega) = H(e^{j\omega})H^*(e^{j\omega})$ where $S(\omega)$ is the power spectral density of the noise.
    - This is solved by Spectral Decomposition Theorem: [ Spectral theorem - Wikipedia ]( https://en.wikipedia.org/wiki/Spectral_theorem )
- How to realize model: Least squares method, Levinson-Durbin algorithm, etc.

## System modeling

Control Problem is separated into three steps:

1. Modeling: Find a system from input and output.
2. Analysis: Analyze the output from the system and input.
3. Design: Find an input from the system and output.

- Transformation: Fourier transform, Laplace transform, Z-transform, etc.
- Function: Correlation function, Spectral density function, Whiteness, Normality, etc.
- Sampling: Quantization etc.

There is two type of system modeling:
1. Model Based Control: Modern Control, Robust Control, Model Predictive Control, etc.
2. Model Free Control: Fuzzy Control, Neuro Control, etc.

(Classical Control is PID control.)

There is three types of creating a mathematical model of a system:
1. First principle modeling: Create a mathematical model based on physical laws and principles.
2. System identification: Create a mathematical model based on input-output data.
3. Grey box modeling: Create a mathematical model based on both physical laws and input-output data.

# Kalman filter

## Linear Kalman filter

Based on orthogonal projection theorem, we can find the optimal estimator of the state from the signal.

There is a stationaly linear Kalman filter, which is a special case of the linear Kalman filter. In this case, the system is time-invariant and the noise is white Gaussian noise:

1. Initialization: 
    - Let $x_0$ be generated from a Gaussian distribution with mean $\mu_0$ and covariance $\Sigma_0 \in\mathbb{R}^{n\times n}$.
    - Let $P_{0|0} = \Sigma_0$ and $\hat{x}_{0|0} = \mu_0$ mean of the state estimation at time $0$.
    - Let $y_k \in\mathbb{R}^p$ be the observation at time $k$.
    - Let system noise and observation noise be generated from Gaussian distributions with mean $0$ and covariance $Q\in\mathbb{R}^{r\times r}$ and $R\in\mathbb{R}^{p\times p}$, respectively.
    - Let $A\in\mathbb{R}^{n\times n}$, $B\in\mathbb{R}^{n\times r}$, and $C\in\mathbb{R}^{p\times n}$ be the state transition matrix, control input matrix, and observation matrix, respectively.
    - $n$ means the dimension of the state, $r$ means the dimension of the control input, and $p$ means the dimension of the observation.
2. Update for $k\in{1, 2, \ldots}$:
    - Prediction step: 
        - $\hat{x}_{k|k-1} = A\hat{x}_{k-1|k-1}$
        - $P_{k|k-1} = AP_{k-1|k-1}A^T + BQ B^T$
    - Update step:
        - $K_k = P_{k|k-1}C^T(CP_{k|k-1}C^T + R)^{-1}$ is the Kalman gain.
        - $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k(y_k - C\hat{x}_{k|k-1})$
        - $P_{k|k} = (I - K_kC)P_{k|k-1}$

If $(A, B)$ is controllable and $(A, C)$ is observable, then the estimation error covariance $P_{k|k}$ converges to a unique positive semi-definite matrix $P$ as $k\to\infty$, which is the solution of the algebraic Riccati equation:

- Controllability: A pair of matrices $(A, B)$ is controllable if the controllability matrix $[B, AB, A^2B, \ldots, A^{n-1}B]$ has full rank $n$.
- Observability: A pair of matrices $(A, C)$ is observable if the observability matrix $\begin{bmatrix} C \\ CA \\ CA^2 \\ \vdots \\ CA^{n-1} \end{bmatrix}$ has full rank $n$.

---

There is a non-stationary linear Kalman filter, which is a general case of the linear Kalman filter. In this case, the system is time-varying and the noise is white Gaussian noise:

1. Update for $k\in{1, 2, \ldots}$:
    - Prediction step: 
        - $\hat{x}_{k|k-1} = A_k\hat{x}_{k-1|k-1}$
        - $P_{k|k-1} = A_kP_{k-1|k-1}A_k^T + B_kQ_k B_k^T$
    - Update step:
        - $K_k = P_{k|k-1}C_k^T(C_kP_{k|k-1}C_k^T + R_k)^{-1}$ is the Kalman gain.
        - $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k(y_k - C_k\hat{x}_{k|k-1})$
        - $P_{k|k} = (I - K_kC_k)P_{k|k-1}$

## Non-linear Kalman filter

1. [ Extended Kalman filter - Wikipedia ]( https://en.wikipedia.org/wiki/Extended_Kalman_filter )
2. UKF: Unscented Kalman filter - [ Kalman filter # Unscented_Kalman_filter - Wikipedia ]( https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter )

# Application

- Integrated Inertial Navigation System
- Litium-ion battery state estimation
