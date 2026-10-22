"""
Preset Synthetic Unstructured Clinical Notes for RareCare Bridge.
All clinical narratives are strictly synthetic and designed for demonstration.
"""

from typing import List, Dict

PRESET_CLINICAL_NOTES: List[Dict[str, str]] = [
    {
        "id": "NOTE-SYNTH-001",
        "title": "Case 1: 34yo Male with Refractory Neuropathic Pain & Renal Deficit",
        "suspected_condition": "Fabry Disease",
        "content": (
            "PATIENT VISIT SUMMARY - METABOLIC OUTPATIENT CLINIC\n"
            "PATIENT ID: SYNTH-METAB-7821\n"
            "DEMOGRAPHICS: 34-year-old male, non-smoker, BMI 23.4.\n\n"
            "CHIEF COMPLAINT:\n"
            "Severe burning dysesthesias in distal extremities (hands and feet), episodic abdominal cramping, "
            "and marked hypohidrosis (inability to sweat even during vigorous exercise or fever episodes).\n\n"
            "CLINICAL HISTORY & ODYSSEY:\n"
            "Patient has reported acute pain crises provoked by temperature fluctuations, fatigue, and stress since age 12. "
            "Previously misdiagnosed with juvenile rheumatoid arthritis and psychogenic somatization disorder.\n"
            "Notices clusters of reddish-purple punctate macules (angiokeratomas) clustered across the 'bathing trunk' "
            "distribution (umbilicus and upper thighs). Slit-lamp biomicroscopy noted corneal verticillata (whorl-like opacity).\n\n"
            "LABORATORY & BIOMARKER FINDINGS:\n"
            "- Serum creatinine: 1.42 mg/dL (baseline was 0.9 mg/dL two years ago)\n"
            "- Estimated GFR (CKD-EPI): 62 mL/min/1.73m² (mild-to-moderate renal insufficiency)\n"
            "- Urinalysis: Persistent microalbuminuria (220 mg/g creatinine)\n"
            "- Cardiac Echo: Concentric left ventricular hypertrophy (interventricular septum thickness 13.5 mm) without systemic arterial hypertension\n"
            "- Plasma Globotriaosylsphingosine (lyso-Gb3): 44.8 nmol/L (severely elevated, reference < 1.8 nmol/L)\n"
            "- Leukocyte Alpha-Galactosidase A activity: 1.1 nmol/hr/mg protein (markedly deficient, reference > 15.0)\n\n"
            "FAMILY HISTORY:\n"
            "Maternal uncle succumbed to end-stage renal disease and early cryptogenic ischemic stroke at age 43.\n\n"
            "ASSESSMENT & PLAN:\n"
            "Presentation highly congruent with classic X-linked Fabry Disease (alpha-galactosidase A deficiency). "
            "Confirmatory GLA gene sequencing recommended immediately. Initiate multidisciplinary nephrology, cardiology, "
            "and genetics evaluation for targeted enzyme replacement therapy (ERT) or pharmacological chaperone eligibility."
        ),
    },
    {
        "id": "NOTE-SYNTH-002",
        "title": "Case 2: 41yo Female with Progressive Exertional Dyspnea & RV Strain",
        "suspected_condition": "Pulmonary Arterial Hypertension (PAH)",
        "content": (
            "CARDIOPULMONARY SPECIALTY CONSULTATION\n"
            "PATIENT ID: SYNTH-CARDIO-4310\n"
            "DEMOGRAPHICS: 41-year-old female, school educator, sedentary lifestyle.\n\n"
            "CHIEF COMPLAINT:\n"
            "Progressive shortness of breath on exertion (NYHA Functional Class III), profound lethargy, and recurrent "
            "near-syncope when ascending single flight of stairs over past 9 months.\n\n"
            "PHYSICAL EXAMINATION:\n"
            "Blood pressure: 112/74 mmHg. Pulse: 94 bpm regular. SpO2: 91% on room air, desaturates to 84% on 6-minute walk test. "
            "Auscultation reveals prominent, narrowly split second heart sound (loud P2) and holosystolic regurgitant murmur "
            "at the lower left sternal border augmenting with inspiration (Carvallo sign). "
            "Elevated jugular venous pulse (+4 cm above sternal angle). Bilateral 2+ pitting pretibial edema.\n\n"
            "DIAGNOSTIC WORKUP:\n"
            "- Transthoracic Echocardiogram (TTE): Significant right ventricular dilatation and hypokinesia. "
            "Flattening of interventricular septum in systole ('D-shaped' left ventricle). Estimated RV systolic pressure (RVSP): 68 mmHg.\n"
            "- Serum NT-proBNP: 2,140 pg/mL (markedly elevated cardiac ventricular strain).\n"
            "- High-Resolution CT Chest: No evidence of chronic interstitial lung disease or diffuse emphysema.\n"
            "- Ventilation-Perfusion (V/Q) Scan: Low probability for thromboembolic disease (rules out CTEPH).\n"
            "- Pulmonary Function Tests: Normal spirometry and lung volumes, isolated reduction in DLCO (46% predicted).\n"
            "- Genetic screen: Heterozygous pathogenic variant identified in BMPR2 locus.\n\n"
            "ASSESSMENT & PLAN:\n"
            "Suspected Group 1 Pulmonary Arterial Hypertension (PAH - Idiopathic/Heritable). "
            "Urgent right heart catheterization (RHC) ordered to confirm hemodynamics (mean PAP > 20 mmHg, PCWP <= 15 mmHg, "
            "PVR >= 2 Wood Units) and assess acute vasoreactivity. Initiate prompt upfront combination vasodilator therapy."
        ),
    },
    {
        "id": "NOTE-SYNTH-003",
        "title": "Case 3: 47yo Male with Involuntary Chorea & Subtle Cognitive Decline",
        "suspected_condition": "Huntington's Disease",
        "content": (
            "NEUROLOGY & MOVEMENT DISORDERS COMPREHENSIVE INTAKE\n"
            "PATIENT ID: SYNTH-NEURO-9942\n"
            "DEMOGRAPHICS: 47-year-old male, aerospace engineer, accompanied by spouse.\n\n"
            "CHIEF COMPLAINT:\n"
            "Irregular, uncoordinated involuntary twitches in upper limbs, progressive balance instability, and "
            "executive workplace dysfunction over past 18 months.\n\n"
            "HISTORY OF PRESENT ILLNESS:\n"
            "Spouse reports increasing irritability, impulsivity, and difficulty multi-tasking. Patient drops household objects "
            "frequently and displays subtle, continuous restless fidgeting of toes, fingers, and facial grimacing. "
            "Patient initially attributed symptoms to vocational burnout and tension.\n\n"
            "NEUROLOGICAL EXAMINATION:\n"
            "- Unified Huntington's Disease Rating Scale (UHDRS) total motor score: 32.\n"
            "- Motor: Generalized choreiform movements prominent in distal extremities, trunk, and perioral muscles. "
            "Motor impersistence demonstrated on tongue protrusion ('serpentine tongue') and grip test ('milkmaid's sign').\n"
            "- Oculomotor: Marked slowing of saccadic initiation and velocity on horizontal gaze.\n"
            "- Gait: Wide-based, lurching, choreic gait with impaired tandem stance.\n"
            "- Montreal Cognitive Assessment (MoCA): 24/30 (deficits in executive sequencing, trail-making, and verbal fluency).\n\n"
            "NEUROIMAGING & GENETIC TESTING:\n"
            "- Brain MRI 3T Volumetric: Prominent bilateral caudate nucleus head atrophy with compensatory enlargement "
            "of the frontal horns of the lateral ventricles. Caudate-interventricular distance markedly elevated.\n"
            "- Molecular Genetics (Targeted HTT PCR assay): CAG trinucleotide repeat expansion in Exon 1 of HTT gene:\n"
            "  * Allele 1: 18 repeats (normal range <= 26)\n"
            "  * Allele 2: 44 repeats (full penetrance mutant range >= 40)\n\n"
            "ASSESSMENT & PLAN:\n"
            "Definite clinical and molecular confirmation of Adult-Onset Huntington's Disease. "
            "Counseling on chorea symptom management with VMAT2 inhibitor (e.g. deutetrabenazine or valbenazine). "
            "Referral to specialized neurogenetics counseling, physical therapy, and social services support."
        ),
    },
    {
        "id": "NOTE-SYNTH-004",
        "title": "Case 4: 52yo Female with Controlled Primary Hypertension (Control Case)",
        "suspected_condition": "Control (Benign Non-Rare Presentation)",
        "content": (
            "INTERNAL MEDICINE ROUTINE ADULT HEALTH MAINTENANCE\n"
            "PATIENT ID: SYNTH-CTRL-1055\n"
            "DEMOGRAPHICS: 52-year-old female, accountant, non-smoker.\n\n"
            "CHIEF COMPLAINT:\n"
            "Routine 6-month checkup for mild primary essential hypertension and occasional tension-type headaches.\n\n"
            "HISTORY & REVIEW OF SYSTEMS:\n"
            "Patient reports feeling generally well. Workload has been stressful during tax season. "
            "No exertional dyspnea, no chest discomfort, no extremity burning, no muscle twitches, no syncope, "
            "no night sweats, no significant weight changes. Current medications: Amlodipine 5 mg daily.\n\n"
            "PHYSICAL EXAMINATION:\n"
            "Blood pressure: 124/80 mmHg (well-controlled). Heart rate: 70 bpm regular. BMI: 25.8.\n"
            "Cardiovascular: Regular rate and rhythm, normal S1 and S2, no murmurs, gallops, or rubs. Lungs: Clear to auscultation bilaterally.\n"
            "Extremities: Warm, well-perfused, zero peripheral edema. Skin: Clear, normal perspiration, no rash or angiokeratomas.\n"
            "Neurological: Alert and fully oriented x 4. Cranial nerves II-XII intact. Rapid alternating movements smooth and symmetric. "
            "No choreiform movements, tremor, or ataxia. Gait normal and narrow-based.\n\n"
            "LABORATORY EVALUATION:\n"
            "- Fasting blood glucose: 94 mg/dL\n"
            "- Comprehensive Metabolic Panel: Serum creatinine 0.88 mg/dL, eGFR > 90 mL/min/1.73m², normal liver enzymes.\n"
            "- Serum NT-proBNP: 52 pg/mL (normal < 125 pg/mL)\n"
            "- Plasma Gb3 level: 0.9 nmol/L (normal reference range)\n"
            "- Lipid panel: Total cholesterol 192 mg/dL, LDL 112 mg/dL, HDL 58 mg/dL.\n\n"
            "ASSESSMENT & PLAN:\n"
            "Well-controlled primary essential hypertension without target organ damage. "
            "No red flags for rare genetic, metabolic, or pulmonary vascular pathologies. "
            "Continue lifestyle modifications, maintain amlodipine 5 mg PO daily, follow-up in 12 months."
        ),
    },
]
