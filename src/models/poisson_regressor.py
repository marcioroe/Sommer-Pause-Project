import numpy as np


class PoissonRegressor:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        self._learning_rate = learning_rate
        self._n_iterations = n_iterations
        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to data via batch gradient descent.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target counts of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []

        for _ in range(self._n_iterations):
            predicted = np.exp(X @ self.coef_)
            loss = np.mean(predicted - y * np.log(predicted))
            self.loss_history_.append(loss)

            gradient = X.T @ (predicted - y) / n_samples
            self.coef_ -= self._learning_rate * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expected counts for the given features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted rates (expected goals) of shape (n_samples,).

        Raises:
            RuntimeError: If called before fit().
        """
        if self.coef_ is None:
            raise RuntimeError("PoissonRegressor must be fit before calling predict().")
        return np.exp(X @ self.coef_)
