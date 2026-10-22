"""
Synthetic Data Module for RareCare Bridge
"""

from rarecare.data.synthetic_data import (
    generate_hospital_dataset,
    generate_all_hospitals,
    generate_global_test_dataset,
    HospitalDataset,
)
from rarecare.data.preset_notes import PRESET_CLINICAL_NOTES

__all__ = [
    "generate_hospital_dataset",
    "generate_all_hospitals",
    "generate_global_test_dataset",
    "HospitalDataset",
    "PRESET_CLINICAL_NOTES",
]
