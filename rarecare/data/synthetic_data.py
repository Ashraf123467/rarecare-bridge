"""
Synthetic Clinical Data Generation for Federated Healthcare Simulation.
Generates imbalanced, Non-IID cohorts across simulated hospital nodes.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

from rarecare.config import (
    HOSPITAL_PROFILES,
    FEATURE_NAMES,
    DISEASE_CLASSES,
    DISEASE_SHORT_NAMES,
)


@dataclass
class HospitalDataset:
    """Encapsulates local synthetic hospital dataset."""
    hospital_key: str
    hospital_name: str
    X: np.ndarray             # Shape: (N, num_features)
    y: np.ndarray             # Shape: (N,)
    df: pd.DataFrame          # Tabular representation with column names
    class_counts: Dict[int, int]
    feature_names: List[str]

    @property
    def sample_count(self) -> int:
        return len(self.y)


def _sample_feature_vector(
    class_label: int,
    rng: np.random.RandomState,
    hospital_shift: np.ndarray,
) -> np.ndarray:
    """
    Generate an 8-dimensional clinical feature vector conditional on disease class,
    with added hospital-specific measurement shift (simulating real non-IID laboratory drift).
    
    Features:
      0: age_normalized
      1: symptom_duration_months
      2: metabolic_gb3_marker
      3: pulmonary_art_pressure
      4: neurological_chorea_score
      5: cardiac_nt_probnp
      6: renal_egfr_deficit
      7: genetic_variant_score
    """
    if class_label == 0:
        # Control: Normal/baseline variations across all markers
        vec = np.array([
            rng.uniform(0.18, 0.75),            # Age 18-75 normalized
            rng.uniform(0.05, 0.30),            # Short benign symptom history
            rng.normal(1.0, 0.20),              # Normal Gb3
            rng.normal(1.0, 0.20),              # Normal pulmonary pressure
            rng.normal(0.5, 0.15),              # No chorea
            rng.normal(0.9, 0.25),              # Normal NT-proBNP
            rng.normal(0.8, 0.20),              # Normal renal function
            rng.normal(0.2, 0.10),              # Low pathogenicity variant
        ])
    elif class_label == 1:
        # Fabry Disease: Elevated Gb3, renal deficit, moderate cardiac strain
        vec = np.array([
            rng.uniform(0.20, 0.65),            # Early-to-middle adulthood
            rng.uniform(0.40, 0.85),            # Chronic diagnostic odyssey
            rng.normal(3.8, 0.50),              # Markedly elevated Gb3
            rng.normal(1.1, 0.25),              # Mild/normal pulmonary
            rng.normal(0.6, 0.18),              # Low neurological chorea
            rng.normal(2.5, 0.45),              # Elevated cardiac strain
            rng.normal(3.4, 0.55),              # Significant renal deficit
            rng.normal(3.2, 0.40),              # Pathogenic GLA variant likelihood
        ])
    elif class_label == 2:
        # Pulmonary Arterial Hypertension (PAH): Extreme pulmonary pressure & NT-proBNP
        vec = np.array([
            rng.uniform(0.30, 0.70),            # Mid-adulthood
            rng.uniform(0.30, 0.75),            # Progressive shortness of breath
            rng.normal(1.0, 0.22),              # Normal Gb3
            rng.normal(4.2, 0.55),              # Extremely high pulmonary arterial pressure
            rng.normal(0.5, 0.15),              # Low chorea
            rng.normal(4.0, 0.60),              # Severe NT-proBNP elevation
            rng.normal(1.5, 0.30),              # Mild secondary renal impact
            rng.normal(2.9, 0.45),              # BMPR2 variant marker
        ])
    elif class_label == 3:
        # Huntington's Disease: Extreme neurological chorea score & genetic CAG marker
        vec = np.array([
            rng.uniform(0.25, 0.60),            # Typical 30s-50s onset
            rng.uniform(0.35, 0.80),            # Motor progression
            rng.normal(1.0, 0.20),              # Normal Gb3
            rng.normal(1.0, 0.20),              # Normal pulmonary
            rng.normal(4.1, 0.50),              # High chorea motor exam
            rng.normal(1.1, 0.25),              # Normal cardiac
            rng.normal(0.9, 0.20),              # Normal renal
            rng.normal(4.5, 0.40),              # High HTT CAG expansion marker
        ])
    else:
        raise ValueError(f"Unknown class label {class_label}")

    # Add hospital-specific calibration/population drift (Non-IID feature shift)
    vec = vec + hospital_shift
    # Ensure all biological markers remain non-negative
    return np.clip(vec, 0.0, 10.0)


def generate_hospital_dataset(
    hospital_key: str,
    seed: int = 42,
) -> HospitalDataset:
    """
    Generate synthetic dataset for a specific hospital node based on its unique profile.
    Maintains Non-IID label skew and slight feature distribution differences.
    """
    if hospital_key not in HOSPITAL_PROFILES:
        raise ValueError(f"Unknown hospital key: {hospital_key}")

    profile = HOSPITAL_PROFILES[hospital_key]
    n_samples = profile["sample_size"]
    class_dist = profile["class_distribution"]
    
    rng = np.random.RandomState(seed)

    # Unique hospital hardware / calibration bias vector
    if hospital_key == "Hospital_A":
        hospital_shift = np.array([0.02, 0.05, 0.15, -0.05, 0.0, 0.10, 0.12, 0.05])
    elif hospital_key == "Hospital_B":
        hospital_shift = np.array([0.03, 0.02, -0.05, 0.20, -0.02, 0.18, 0.05, 0.02])
    else:  # Hospital_C
        hospital_shift = np.array([-0.02, 0.04, 0.0, -0.04, 0.15, -0.05, -0.02, 0.16])

    # Determine counts per class
    classes = sorted(class_dist.keys())
    probs = [class_dist[c] for c in classes]
    sampled_classes = rng.choice(classes, size=n_samples, p=probs)

    feature_matrix = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=np.float32)
    for i, c in enumerate(sampled_classes):
        feature_matrix[i] = _sample_feature_vector(c, rng, hospital_shift)

    # Construct DataFrame for easy display and tabular analysis
    df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    df["disease_class"] = sampled_classes
    df["disease_name"] = df["disease_class"].map(DISEASE_CLASSES)
    df["hospital"] = profile["name"]

    # Calculate actual class counts
    class_counts = {int(c): int(np.sum(sampled_classes == c)) for c in classes}

    return HospitalDataset(
        hospital_key=hospital_key,
        hospital_name=profile["name"],
        X=feature_matrix,
        y=sampled_classes.astype(np.int64),
        df=df,
        class_counts=class_counts,
        feature_names=FEATURE_NAMES,
    )


def generate_all_hospitals(seed_offset: int = 100) -> Dict[str, HospitalDataset]:
    """Generate datasets for all 3 simulated hospitals."""
    datasets = {}
    for i, key in enumerate(HOSPITAL_PROFILES.keys()):
        datasets[key] = generate_hospital_dataset(key, seed=seed_offset + i * 37)
    return datasets


def generate_global_test_dataset(
    n_samples: int = 320,
    seed: int = 999,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate an unbiased, balanced global test holdout dataset (80 samples per class).
    Used to benchmark local hospital models vs the federated global model fairly.
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // len(DISEASE_CLASSES)
    remainder = n_samples % len(DISEASE_CLASSES)

    all_features = []
    all_labels = []

    # Unbiased zero shift for true population benchmark
    zero_shift = np.zeros(len(FEATURE_NAMES))

    for c in sorted(DISEASE_CLASSES.keys()):
        count = samples_per_class + (1 if c < remainder else 0)
        for _ in range(count):
            all_features.append(_sample_feature_vector(c, rng, zero_shift))
            all_labels.append(c)

    X_test = np.array(all_features, dtype=np.float32)
    y_test = np.array(all_labels, dtype=np.int64)

    # Shuffle test set
    perm = rng.permutation(len(y_test))
    X_test = X_test[perm]
    y_test = y_test[perm]

    df_test = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    df_test["disease_class"] = y_test
    df_test["disease_name"] = df_test["disease_class"].map(DISEASE_CLASSES)

    return X_test, y_test, df_test
