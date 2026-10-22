"""
Central Federated Server and FedAvg Aggregator.
Coordinates federation rounds and evaluates global convergence on holdout data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from rarecare.federated.model import FederatedSoftmaxClassifier
from rarecare.federated.client import HospitalClient
from rarecare.federated.metrics import calculate_metrics, EvaluationMetrics


@dataclass
class RoundMetricRecord:
    """Telemetry data captured during a single federated communication round."""
    round_num: int
    test_loss: float
    test_accuracy: float
    test_macro_f1: float
    test_macro_precision: float
    test_macro_recall: float
    client_losses: Dict[str, float]
    aggregated_weight_norm: float


@dataclass
class FederationHistory:
    """Historical progression of metrics across all federated rounds."""
    records: List[RoundMetricRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[Any]]:
        """Export history as dictionary for charting."""
        return {
            "round": [r.round_num for r in self.records],
            "test_accuracy": [r.test_accuracy for r in self.records],
            "test_f1": [r.test_macro_f1 for r in self.records],
            "test_precision": [r.test_macro_precision for r in self.records],
            "test_recall": [r.test_macro_recall for r in self.records],
            "test_loss": [r.test_loss for r in self.records],
        }


class CentralFederatedServer:
    """
    Central Federated Learning Aggregation Coordinator.
    
    Implements standard McMahan et al. FedAvg (Federated Averaging).
    Maintains zero access to raw client patient records.
    """

    def __init__(
        self,
        clients: List[HospitalClient],
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_features: int = 8,
        num_classes: int = 4,
        seed: int = 42,
    ):
        self.clients = clients
        # Global validation / test holdout benchmark
        self.X_test = X_test
        self.y_test = y_test
        
        # Central global model
        self.global_model = FederatedSoftmaxClassifier(
            num_features=num_features,
            num_classes=num_classes,
            seed=seed,
        )
        
        self.history = FederationHistory()
        self.current_round = 0

    def aggregate_fedavg(
        self,
        client_updates: List[Tuple[Dict[str, np.ndarray], int]],
    ) -> Dict[str, np.ndarray]:
        """
        FedAvg Aggregation Formula:
          W_{t+1} = sum_{k=1}^K (n_k / N_total) * W_k
          b_{t+1} = sum_{k=1}^K (n_k / N_total) * b_k
        """
        total_samples = sum(n_k for _, n_k in client_updates)
        if total_samples == 0:
            return self.global_model.get_parameters()

        first_params, _ = client_updates[0]
        agg_W = np.zeros_like(first_params["W"])
        agg_b = np.zeros_like(first_params["b"])

        for params, n_k in client_updates:
            weight = float(n_k) / float(total_samples)
            agg_W += weight * params["W"]
            agg_b += weight * params["b"]

        return {
            "W": agg_W,
            "b": agg_b,
        }

    def run_round(
        self,
        local_epochs: int = 3,
        lr: float = 0.05,
        l2_reg: float = 0.001,
        batch_size: int = 32,
    ) -> RoundMetricRecord:
        """
        Execute a single communication round of Federated Learning:
        1. Broadcast current global parameters to all hospital clients.
        2. Hospital clients train locally on private data.
        3. Clients send back updated parameters and sample weights.
        4. Central server performs FedAvg aggregation.
        5. Evaluate updated global model on the unbiased global test set.
        """
        self.current_round += 1
        current_global_params = self.global_model.get_parameters()

        client_updates: List[Tuple[Dict[str, np.ndarray], int]] = []
        client_losses: Dict[str, float] = {}

        # 1-3: Client local training step
        for client in self.clients:
            params, sample_count, loss = client.train_local_round(
                global_params=current_global_params,
                epochs=local_epochs,
                lr=lr,
                l2_reg=l2_reg,
                batch_size=batch_size,
            )
            client_updates.append((params, sample_count))
            client_losses[client.hospital_key] = loss

        # 4: Central FedAvg aggregation
        new_global_params = self.aggregate_fedavg(client_updates)
        self.global_model.set_parameters(new_global_params)

        # 5: Benchmark against global test set
        test_metrics = self.evaluate_on_test_set()
        test_loss = self.global_model.compute_loss(self.X_test, self.y_test, l2_reg=l2_reg)

        weight_norm = float(np.linalg.norm(new_global_params["W"]))

        record = RoundMetricRecord(
            round_num=self.current_round,
            test_loss=test_loss,
            test_accuracy=test_metrics.accuracy,
            test_macro_f1=test_metrics.macro_f1,
            test_macro_precision=test_metrics.macro_precision,
            test_macro_recall=test_metrics.macro_recall,
            client_losses=client_losses,
            aggregated_weight_norm=weight_norm,
        )
        self.history.records.append(record)
        return record

    def run_simulation(
        self,
        total_rounds: int = 8,
        local_epochs: int = 3,
        lr: float = 0.05,
        l2_reg: float = 0.001,
    ) -> FederationHistory:
        """Run full federation across multiple communication rounds."""
        for _ in range(total_rounds):
            self.run_round(local_epochs=local_epochs, lr=lr, l2_reg=l2_reg)
        return self.history

    def evaluate_on_test_set(self) -> EvaluationMetrics:
        """Evaluate the central global model on the holdout test set."""
        y_pred = self.global_model.predict(self.X_test)
        return calculate_metrics(
            y_true=self.y_test,
            y_pred=y_pred,
            num_classes=self.global_model.num_classes,
        )
