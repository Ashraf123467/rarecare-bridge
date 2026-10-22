"""
Tests for Clinical Note Structuring, Input Validation, and Demo Mode Fallback.
"""

import pytest
from rarecare.note_structurer.extractor import extract_structured_note
from rarecare.note_structurer.schemas import StructuredClinicalNote
from rarecare.data.preset_notes import PRESET_CLINICAL_NOTES


def test_validation_empty_note():
    """Verify that an empty clinical note is rejected with a validation error."""
    res = extract_structured_note("")
    assert not res.success
    assert "empty" in res.error_message.lower()


def test_validation_short_note():
    """Verify that an excessively brief note is rejected with a validation error."""
    res = extract_structured_note("Patient is unwell.")
    assert not res.success
    assert "too brief" in res.error_message.lower()


def test_demo_mode_preset_notes():
    """Verify that all preset synthetic notes structure properly in demo mode."""
    for preset in PRESET_CLINICAL_NOTES:
        res = extract_structured_note(
            note_text=preset["content"],
            force_demo=True,
        )
        assert res.success
        assert res.is_demo_mode
        assert isinstance(res.data, StructuredClinicalNote)
        assert res.data.patient_id != ""
        assert len(res.data.primary_symptoms) > 0
        assert len(res.data.icd10_candidates) > 0
        assert len(res.data.recommended_workup) > 0
        assert len(res.data.feature_vector_estimate) == 8
        assert "CONFIRMED_SYNTHETIC" in res.data.synthetic_declaration


def test_demo_mode_custom_note_parsing():
    """Verify that an arbitrary custom synthetic note extracts demographic and clinical entities."""
    custom_text = (
        "CONSULTATION NOTE: 39-year-old female presenting with severe burning pain in fingers, "
        "anhidrosis, angiokeratomas on lower abdomen, and plasma lyso-Gb3 elevation to 38 nmol/L. "
        "Strong suspicion for Fabry Disease."
    )
    res = extract_structured_note(note_text=custom_text, force_demo=True)
    assert res.success
    assert res.data.age == 39
    assert res.data.gender == "Female"
    assert "Fabry" in res.data.suspected_condition
    assert res.data.feature_vector_estimate["metabolic_gb3_marker"] > 3.0
