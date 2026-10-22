"""
Hospital Client Node for Federated Healthcare Network.
Strictly encapsulates local hospital data; only model weights are exported.
"""

from typing import Dict, Tuple, Any
import numpy as np

from rarecare.data.synthetic_data import HospitalDataset
from rarecare.federated.model import FederatedSoftmaxClassifier
from rarecare.federated.metrics import calculate_metrics, EvaluationMetrics


class HospitalClient:
    """
    Simulated Hospital Edge Client Node.
    
    PRIVACY PRESERVATION CONTRACT:
    - Raw patient data (self._local_dataset) NEVER leaves the client boundary.
    - Server interactions are strictly limited to receiving global weights and 
      returning updated weights + sample count.
    """

    def __init__(
        self,
        hospital_key: str,
        dataset: HospitalDataset,
        num_features: int = 8,
        num_classes: int = 4,
        seed: int = 42,
    ):
        self.hospital_key = hospital_key
        self.hospital_name = dataset.hospital_name
        # Private local dataset - NOT shared across network
        self._local_dataset = dataset
        
        # Local model instance
        self.model = FederatedSoftmaxClassifier(
            num_features=num_features,
            num_classes=num_classes,
            seed=seed,
        )

    @property
    def sample_count(self) -> int:
        """Return the number of local patient records."""
        return self._local_dataset.sample_count

    @property
    def class_distribution(self) -> Dict[int, int]:
        """Return the local class distribution for local analytics."""
        return self._local_dataset.class_counts

    def get_public_metadata(self) -> Dict[str, Any]:
        """Return non-sensitive hospital configuration metadata."""
        return {
            "hospital_key": self.hospital_key,
            "hospital_name": self.hospital_name,
            "total_records": self.sample_count,
            "class_counts": self.class_distribution,
        }

    def train_local_round(
        self,
        global_params: Dict[str, np.ndarray],
        epochs: int = 3,
        lr: float = 0.05,
        l2_reg: float = 0.001,
        batch_size: int = 32,
    ) -> Tuple[Dict[str, np.ndarray], int, float]:
        """
        Federated Client Execution:
        1. Receive global model parameters from coordinator.
        2. Update local model with global parameters.
        3. Train locally on private patient dataset.
        4. Return updated weights, sample count, and training loss.
        
        CRITICAL: Raw rows X and labels y are NOT transmitted.
        """
        # 1. Synchronize local model with global weights
        self.model.set_parameters(global_params)

        # 2. Train exclusively on local private data
        loss = self.model.fit_step(
            X=self._local_dataset.X,
            y=self._local_dataset.y,
            lr=lr,
            l2_reg=l2_reg,
            batch_size=batch_size,
            epochs=epochs,
        )

        # 3. Export ONLY model parameters (W, b) and scalar sample count
        updated_params = self.model.get_parameters()
        return updated_params, self.sample_count, float(loss)

    def train_isolated_local_model(
        self,
        epochs: int = 30,
        lr: float = 0.05,
        l2_reg: float = 0.001,
    ) -> FederatedSoftmaxClassifier:
        """
        Train a standalone model solely on this hospital's local data (No federation).
        Used for benchmarking against the federated model to demonstrate the clinical
        value of cross-institutional collaboration.
        """
        local_standalone = FederatedSoftmaxClassifier(
            num_features=self.model.num_features,
            num_classes=self.model.num_classes,
            seed=42,
        )
        local_standalone.fit_step(
            X=self._local_dataset.X,
            y=self._local_dataset.y,
            lr=lr,
            l2_reg=l2_reg,
            epochs=epochs,
        )
        return local_standalone

    def evaluate_local_data(self, model: FederatedSoftmaxClassifier = None) -> EvaluationMetrics:
        """
        Evaluate a model on this hospital's local dataset.
        If no model is supplied, evaluates self.model.
        """
        eval_model = model if model is not None else self.model
        y_pred = eval_model.predict(self._local_dataset.X)
        return calculate_metrics(
            y_true=self._local_dataset.y,
            y_pred=y_pred,
            num_classes=eval_model.num_classes,
        )
