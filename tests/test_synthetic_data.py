"""
Tests for Synthetic Healthcare Data Generation and Non-IID Hospital Cohorts.
"""

import numpy as np
import pytest

from rarecare.data.synthetic_data import (
    generate_hospital_dataset,
    generate_all_hospitals,
    generate_global_test_dataset,
)
from rarecare.config import FEATURE_NAMES, HOSPITAL_PROFILES


def test_hospital_dataset_structure():
    """Verify that hospital dataset has valid shapes, columns, and properties."""
    hosp_a = generate_hospital_dataset("Hospital_A", seed=42)
    assert hosp_a.sample_count == HOSPITAL_PROFILES["Hospital_A"]["sample_size"]
    assert hosp_a.X.shape == (hosp_a.sample_count, len(FEATURE_NAMES))
    assert hosp_a.y.shape == (hosp_a.sample_count,)
    assert set(hosp_a.df.columns).issuperset(FEATURE_NAMES)
    assert "disease_class" in hosp_a.df.columns
    assert "hospital" in hosp_a.df.columns


def test_non_iid_distribution_skew():
    """
    Verify Non-IID characteristics across the 3 hospitals:
    - Hospital A should have highest proportion of Fabry Disease (Class 1)
    - Hospital B should have highest proportion of PAH (Class 2)
    - Hospital C should have highest proportion of Huntington's (Class 3)
    """
    all_hospitals = generate_all_hospitals(seed_offset=100)

    # Class 1: Fabry Disease
    fabry_proportions = {
        k: ds.class_counts[1] / ds.sample_count for k, ds in all_hospitals.items()
    }
    assert fabry_proportions["Hospital_A"] > fabry_proportions["Hospital_B"]
    assert fabry_proportions["Hospital_A"] > fabry_proportions["Hospital_C"]

    # Class 2: PAH
    pah_proportions = {
        k: ds.class_counts[2] / ds.sample_count for k, ds in all_hospitals.items()
    }
    assert pah_proportions["Hospital_B"] > fabry_proportions["Hospital_A"]
    assert pah_proportions["Hospital_B"] > pah_proportions["Hospital_C"]

    # Class 3: Huntington's Disease
    hd_proportions = {
        k: ds.class_counts[3] / ds.sample_count for k, ds in all_hospitals.items()
    }
    assert hd_proportions["Hospital_C"] > hd_proportions["Hospital_A"]
    assert hd_proportions["Hospital_C"] > hd_proportions["Hospital_B"]


def test_global_test_dataset_balanced():
    """Verify that the global test dataset is balanced across all 4 classes."""
    X_test, y_test, df_test = generate_global_test_dataset(n_samples=320, seed=999)
    assert len(X_test) == 320
    assert len(y_test) == 320
    assert X_test.shape[1] == len(FEATURE_NAMES)

    # Each of the 4 classes should have exactly 80 samples
    unique, counts = np.unique(y_test, return_counts=True)
    assert len(unique) == 4
    for count in counts:
        assert count == 80
