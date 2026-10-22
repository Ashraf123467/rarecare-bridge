"""
Tests for Privacy-Preserving Federated Learning Engine and Metrics.
"""

import numpy as np
import pytest

from rarecare.data.synthetic_data import (
    generate_all_hospitals,
    generate_global_test_dataset,
)
from rarecare.federated.model import FederatedSoftmaxClassifier
from rarecare.federated.client import HospitalClient
from rarecare.federated.server import CentralFederatedServer
from rarecare.federated.metrics import calculate_metrics


def test_softmax_classifier_training_and_shapes():
    """Verify classifier forward pass, loss decrease, and parameter export/import."""
    X_test, y_test, _ = generate_global_test_dataset(n_samples=100, seed=12)
    model = FederatedSoftmaxClassifier(num_features=8, num_classes=4, seed=42)

    initial_loss = model.compute_loss(X_test, y_test)
    probs = model.predict_proba(X_test)
    assert probs.shape == (100, 4)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    # Train for 10 epochs
    final_loss = model.fit_step(X_test, y_test, lr=0.08, epochs=10)
    assert final_loss < initial_loss

    # Test parameters cloning
    params = model.get_parameters()
    assert "W" in params and "b" in params
    assert params["W"].shape == (4, 8)
    assert params["b"].shape == (4,)

    cloned = model.clone()
    np.testing.assert_array_equal(cloned.W, model.W)


def test_privacy_preservation_boundary():
    """
    CRITICAL PRIVACY TEST:
    Verify that HospitalClient strictly encapsulates raw patient data.
    When training rounds execute, ONLY model weights and sample counts are transmitted.
    """
    hospitals = generate_all_hospitals(seed_offset=1)
    client_a = HospitalClient("Hospital_A", hospitals["Hospital_A"])

    global_model = FederatedSoftmaxClassifier(num_features=8, num_classes=4, seed=10)
    global_params = global_model.get_parameters()

    # Execute federated training step
    returned_params, returned_samples, returned_loss = client_a.train_local_round(
        global_params=global_params,
        epochs=2,
    )

    # 1. Verify returned parameters strictly contain numerical tensors W and b
    assert isinstance(returned_params, dict)
    assert set(returned_params.keys()) == {"W", "b"}
    assert isinstance(returned_params["W"], np.ndarray)
    assert isinstance(returned_params["b"], np.ndarray)

    # 2. Assert NO raw dataset or patient identifiers are in the returned payload
    assert "X" not in returned_params
    assert "y" not in returned_params
    assert "data" not in returned_params
    assert "patient_id" not in returned_params
    assert "records" not in returned_params

    # 3. Verify sample count is a positive integer
    assert isinstance(returned_samples, int)
    assert returned_samples == client_a.sample_count


def test_fedavg_mathematical_aggregation():
    """Verify exact weighted averaging of model weights in CentralFederatedServer."""
    X_test, y_test, _ = generate_global_test_dataset(n_samples=50, seed=1)
    server = CentralFederatedServer(clients=[], X_test=X_test, y_test=y_test)

    # Create synthetic client weights
    W1 = np.ones((4, 8), dtype=np.float32) * 2.0
    b1 = np.ones(4, dtype=np.float32) * 1.0

    W2 = np.ones((4, 8), dtype=np.float32) * 4.0
    b2 = np.ones(4, dtype=np.float32) * 3.0

    # Case 1: Equal weighting (n1=100, n2=100) -> Mean should be 3.0 and 2.0
    updates_equal = [
        ({"W": W1, "b": b1}, 100),
        ({"W": W2, "b": b2}, 100),
    ]
    agg_equal = server.aggregate_fedavg(updates_equal)
    np.testing.assert_allclose(agg_equal["W"], 3.0, atol=1e-6)
    np.testing.assert_allclose(agg_equal["b"], 2.0, atol=1e-6)

    # Case 2: Unequal weighting (n1=300, n2=100) -> 0.75 * 2.0 + 0.25 * 4.0 = 2.5
    updates_unequal = [
        ({"W": W1, "b": b1}, 300),
        ({"W": W2, "b": b2}, 100),
    ]
    agg_unequal = server.aggregate_fedavg(updates_unequal)
    np.testing.assert_allclose(agg_unequal["W"], 2.5, atol=1e-6)
    np.testing.assert_allclose(agg_unequal["b"], 1.5, atol=1e-6)


def test_federated_simulation_and_metrics_calculation():
    """Verify that multi-round federated training runs and improves test metrics."""
    hospitals = generate_all_hospitals(seed_offset=50)
    clients = [HospitalClient(k, ds) for k, ds in hospitals.items()]
    X_test, y_test, _ = generate_global_test_dataset(n_samples=200, seed=77)

    server = CentralFederatedServer(
        clients=clients,
        X_test=X_test,
        y_test=y_test,
        seed=42,
    )

    initial_metrics = server.evaluate_on_test_set()

    # Run 6 federated rounds
    history = server.run_simulation(total_rounds=6, local_epochs=3, lr=0.08)

    final_metrics = server.evaluate_on_test_set()

    # The federated model should significantly outperform random initialization
    assert final_metrics.accuracy > initial_metrics.accuracy
    assert final_metrics.macro_f1 > initial_metrics.macro_f1
    assert final_metrics.accuracy >= 0.80  # Should achieve >= 80% accuracy on synthetic benchmark

    # Verify metrics structure
    assert 0.0 <= final_metrics.macro_precision <= 1.0
    assert 0.0 <= final_metrics.macro_recall <= 1.0
    assert final_metrics.confusion_matrix.shape == (4, 4)
    assert len(history.records) == 6
