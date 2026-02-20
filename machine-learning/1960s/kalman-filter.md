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

# Kalman filter

## Linear Kalman filter

## Non-linear Kalman filter

# Application
