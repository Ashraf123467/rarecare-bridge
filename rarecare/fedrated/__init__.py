"""
Federated Learning Engine for RareCare Bridge
"""

from rarecare.federated.model import FederatedSoftmaxClassifier
from rarecare.federated.client import HospitalClient
from rarecare.federated.server import CentralFederatedServer, FederationHistory
from rarecare.federated.metrics import calculate_metrics, EvaluationMetrics

__all__ = [
    "FederatedSoftmaxClassifier",
    "HospitalClient",
    "CentralFederatedServer",
    "FederationHistory",
    "calculate_metrics",
    "EvaluationMetrics",
]
