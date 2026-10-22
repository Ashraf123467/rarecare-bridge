"""
Softmax Multi-Class Classifier for Federated Learning.
Vectorized, deterministic, and supports parameter extraction and aggregation.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class FederatedSoftmaxClassifier:
    """
    Multinomial Logistic Regression / Softmax Neural Classifier.
    Maps 8 clinical features to 4 rare disease categories.
    """

    def __init__(
        self,
        num_features: int = 8,
        num_classes: int = 4,
        seed: int = 42,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.seed = seed
        
        # Initialize weights and bias with small Gaussian noise
        rng = np.random.RandomState(seed)
        self.W = (rng.randn(num_classes, num_features) * 0.05).astype(np.float32)
        self.b = np.zeros(num_classes, dtype=np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        # Shift logits by max for numerical stability
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for input feature matrix X."""
        X = np.asarray(X, dtype=np.float32)
        logits = np.dot(X, self.W.T) + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def compute_loss(self, X: np.ndarray, y: np.ndarray, l2_reg: float = 0.001) -> float:
        """Compute multiclass cross-entropy loss with L2 weight regularization."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        if n == 0:
            return 0.0

        probs = self.predict_proba(X)
        # Avoid log(0) with clipping
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        
        # Negative log likelihood of true classes
        log_likelihood = -np.log(probs_clipped[np.arange(n), y])
        data_loss = float(np.mean(log_likelihood))
        reg_loss = 0.5 * l2_reg * float(np.sum(self.W ** 2))
        return data_loss + reg_loss

    def fit_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.05,
        l2_reg: float = 0.001,
        batch_size: Optional[int] = None,
        epochs: int = 5,
        seed: Optional[int] = None,
    ) -> float:
        """
        Train the model locally for a specified number of epochs.
        Returns the final epoch training loss.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        if n == 0:
            return 0.0

        rng = np.random.RandomState(seed or self.seed)
        final_loss = 0.0

        for _ in range(epochs):
            indices = np.arange(n)
            rng.shuffle(indices)

            bs = batch_size if (batch_size and 0 < batch_size < n) else n

            for start_idx in range(0, n, bs):
                batch_idx = indices[start_idx : start_idx + bs]
                X_b = X[batch_idx]
                y_b = y[batch_idx]
                b_size = len(y_b)

                # Forward pass
                probs = self.predict_proba(X_b)

                # One-hot encoding of targets
                Y_one_hot = np.zeros((b_size, self.num_classes), dtype=np.float32)
                Y_one_hot[np.arange(b_size), y_b] = 1.0

                # Compute analytical gradient
                # error shape: (b_size, num_classes)
                error = probs - Y_one_hot

                # grad_W shape: (num_classes, num_features)
                grad_W = (np.dot(error.T, X_b) / b_size) + (l2_reg * self.W)
                # grad_b shape: (num_classes,)
                grad_b = np.mean(error, axis=0)

                # Parameter update step
                self.W -= lr * grad_W
                self.b -= lr * grad_b

            final_loss = self.compute_loss(X, y, l2_reg=l2_reg)

        return final_loss

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Export model weights and biases for federated aggregation.
        No raw data is included in this dictionary.
        """
        return {
            "W": self.W.copy(),
            "b": self.b.copy(),
        }

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Update model parameters from central aggregated weights."""
        if "W" in params and "b" in params:
            self.W = params["W"].astype(np.float32).copy()
            self.b = params["b"].astype(np.float32).copy()
        else:
            raise KeyError("Parameters dictionary must contain 'W' and 'b' keys.")

    def clone(self) -> "FederatedSoftmaxClassifier":
        """Create a deep copy of the classifier."""
        new_model = FederatedSoftmaxClassifier(
            num_features=self.num_features,
            num_classes=self.num_classes,
            seed=self.seed,
        )
        new_model.set_parameters(self.get_parameters())
        return new_model
