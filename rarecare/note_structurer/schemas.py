"""
Pydantic Schemas for Structured Healthcare Information Extraction.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class BiomarkerFinding(BaseModel):
    """Specific lab finding or clinical biomarker measurement."""
    name: str = Field(description="Name of the biomarker or laboratory assay (e.g. lyso-Gb3, NT-proBNP)")
    value: str = Field(description="Observed value with units (e.g. 44.8 nmol/L, 2140 pg/mL)")
    reference_range: str = Field(description="Reference or normal population range")
    interpretation: str = Field(description="Clinical status: Normal, Elevated, Deficient, or Borderline")


class StructuredClinicalNote(BaseModel):
    """Comprehensive structured extraction schema for unstructured synthetic medical notes."""
    patient_id: str = Field(default="SYNTH-UNKNOWN", description="Synthetic Patient Identifier")
    age: Optional[int] = Field(default=None, description="Patient age in years")
    gender: Optional[str] = Field(default=None, description="Patient sex/gender")
    chief_complaint: str = Field(description="Primary presenting complaint")
    primary_symptoms: List[str] = Field(default_factory=list, description="Key signs and symptoms extracted")
    suspected_condition: str = Field(description="Primary rare disease or clinical presentation suspected")
    confidence_level: str = Field(default="High", description="Extraction confidence (Low, Moderate, High)")
    icd10_candidates: List[str] = Field(default_factory=list, description="Candidate ICD-10 diagnostic codes")
    biomarkers: List[BiomarkerFinding] = Field(default_factory=list, description="Extracted biomarkers with values")
    recommended_workup: List[str] = Field(default_factory=list, description="Next recommended clinical actions or confirmatory tests")
    urgency_level: str = Field(default="Urgent", description="Clinical urgency rating: Routine, Urgent, or Emergency")
    clinical_summary: str = Field(description="Concise synthesis of the patient case for multi-disciplinary review")
    feature_vector_estimate: Dict[str, float] = Field(
        default_factory=dict,
        description="Estimated values for the 8 standard federated tabular indicators (scaled 0.0 - 5.0)"
    )
    disclaimer: str = Field(
        default="EXPERIMENTAL SYNTHETIC DEMONSTRATION ONLY - NOT CLINICALLY VALIDATED",
        description="Regulatory medical disclaimer"
    )
    synthetic_declaration: str = Field(
        default="CONFIRMED_SYNTHETIC_DATA_ONLY",
        description="Verification that input data is entirely synthetic"
    )


class ExtractionResult(BaseModel):
    """Wrapper encapsulating extraction outcome, demo mode flag, and errors."""
    success: bool
    data: Optional[StructuredClinicalNote] = None
    is_demo_mode: bool = False
    model_used: str = "offline-synthetic-engine"
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
