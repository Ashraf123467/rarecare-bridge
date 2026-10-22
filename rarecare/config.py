"""
RareCare Bridge Configuration and Constants
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()

# Strict Medical Disclaimer (Required by compliance & ethics)
MEDICAL_DISCLAIMER: str = (
    "⚠️ MEDICAL DISCLAIMER: RareCare Bridge is an experimental AI research demonstration tool "
    "designed exclusively for synthetic healthcare data structuring and privacy-preserving federated machine learning. "
    "It has NOT undergone clinical validation and is NOT approved by the U.S. FDA, EMA, or any national or international "
    "health authority. Under no circumstances should this system be utilized for real-world patient diagnosis, clinical triage, "
    "prescription generation, or direct therapeutic decision-making. All clinical notes and records simulated within this application "
    "are purely synthetic and do not correspond to living or deceased individuals."
)

# Disease Classes for Multi-Class Rare Disease Identification
DISEASE_CLASSES: Dict[int, str] = {
    0: "Control (Benign Non-Rare Presentation)",
    1: "Fabry Disease (Lysosomal Storage Disorder)",
    2: "Pulmonary Arterial Hypertension (PAH)",
    3: "Huntington's Disease (Neurogenetic Disorder)",
}

DISEASE_SHORT_NAMES: Dict[int, str] = {
    0: "Control",
    1: "Fabry Disease",
    2: "PAH",
    3: "Huntington's",
}

# 8 Normalized Clinical Indicator Features
FEATURE_NAMES = [
    "age_normalized",               # Normalized patient age [0, 1]
    "symptom_duration_months",      # Normalized symptom timeline
    "metabolic_gb3_marker",         # Plasma globotriaosylceramide proxy
    "pulmonary_art_pressure",       # Mean pulmonary arterial pressure proxy
    "neurological_chorea_score",    # Motor exam score proxy
    "cardiac_nt_probnp",            # Cardiac strain biomarker proxy
    "renal_egfr_deficit",           # Estimated GFR deficit proxy
    "genetic_variant_score",        # Pathogenic variant predictive likelihood
]

# Simulated Hospital Cohort Profiles (Imbalanced, Non-IID distributions)
HOSPITAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "Hospital_A": {
        "name": "St. Jude Rare Metabolic Center",
        "code": "HOSP-METABOLIC",
        "location": "Boston, MA",
        "specialty": "Metabolic & Lysosomal Storage Disorders",
        "color": "#0284C7",  # Sky blue
        "description": "Tertiary referral hub for lysosomal storage diseases. Displays extreme prevalence of Fabry Disease with elevated Gb3 and renal deficit markers.",
        "sample_size": 280,
        # Extreme skew towards Fabry Disease
        "class_distribution": {0: 0.25, 1: 0.55, 2: 0.12, 3: 0.08},
    },
    "Hospital_B": {
        "name": "Metro Cardiopulmonary Institute",
        "code": "HOSP-CARDIO",
        "location": "Chicago, IL",
        "specialty": "Cardiorespiratory & Pulmonary Vascular Medicine",
        "color": "#E11D48",  # Rose
        "description": "Urban cardiovascular research center. High patient density with severe Pulmonary Arterial Hypertension (elevated mean PAP and NT-proBNP).",
        "sample_size": 240,
        # Extreme skew towards PAH
        "class_distribution": {0: 0.20, 1: 0.10, 2: 0.60, 3: 0.10},
    },
    "Hospital_C": {
        "name": "Northwest Neurogenetic Hospital",
        "code": "HOSP-NEURO",
        "location": "Seattle, WA",
        "specialty": "Neuromuscular & Inherited Genetic Disorders",
        "color": "#7C3AED",  # Violet
        "description": "Regional neurogenetics facility. Skewed cohort of early-to-mid stage Huntington's Disease patients exhibiting high chorea and genetic variant scores.",
        "sample_size": 260,
        # Extreme skew towards Huntington's Disease
        "class_distribution": {0: 0.22, 1: 0.08, 2: 0.12, 3: 0.58},
    },
}

# GenAI Settings
DEFAULT_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "ggemini-3.1-flash-lite")

def get_gemini_api_key() -> str:
    """
    Safely retrieve GEMINI_API_KEY from environment variables.
    Never hardcode API keys.
    """
    return os.environ.get("GEMINI_API_KEY", "").strip()
