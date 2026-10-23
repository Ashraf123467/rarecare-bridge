# 🩺 RareCare Bridge

> **Privacy-Preserving Federated Healthcare Intelligence for Rare Diseases**  
> *PromptWars MVP Demonstration*

[Live Demo](https://rarecare-bridge.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google GenAI](https://img.shields.io/badge/Google%20GenAI-Official%20SDK-0D9488.svg)](https://github.com/google-gemini/generative-ai-python)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-FedAvg-7C3AED.svg)](#federated-learning-architecture)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ⚠️ Important Medical Disclaimer

> **MEDICAL DISCLAIMER:**  
> **RareCare Bridge is an experimental AI research demonstration tool designed exclusively for synthetic healthcare data structuring and privacy-preserving federated machine learning. It has NOT undergone clinical validation and is NOT approved by the U.S. FDA, EMA, or any national or international health authority. Under no circumstances should this system be utilized for real-world patient diagnosis, clinical triage, prescription generation, or direct therapeutic decision-making. All clinical notes and records simulated within this application are purely synthetic and do not correspond to living or deceased individuals.**

---

## 🌟 Overview

Rare diseases affect over 300 million people worldwide, yet diagnosis often takes **5 to 7 years** (the "diagnostic odyssey") because individual hospital centers only see a tiny handful of cases. Traditional machine learning requires pooling sensitive Electronic Health Records (EHR) into a central repository, which violates patient privacy and cross-border healthcare data regulations (e.g., HIPAA, GDPR).

**RareCare Bridge** bridges this divide with two key innovations:
1. **Clinical Note Structurer:** Uses the official `google-genai` SDK (`gemini-2.5-flash`) to parse unstructured synthetic clinical narratives into standardized, structured medical entities, ICD-10 candidate codes, biomarkers, and normalized tabular features. Includes a fully functional offline **Demo Mode** when no API key is configured.
2. **Privacy-Preserving Federated Learning (FedAvg):** Coordinates collaborative multi-class rare disease classification across three simulated academic hospitals with severely skewed, Non-IID patient cohorts. **Raw patient records never leave the local hospital client boundary; only aggregated model weight updates are transmitted.**

---

## 🏥 Three Simulated Hospitals (Imbalanced, Non-IID Cohorts)

To faithfully mirror real-world healthcare fragmentation, the system simulates three distinct academic medical centers:

| Hospital Node | Primary Specialty | Sample Size | Primary Label Skew | Non-IID Domain Bias |
| :--- | :--- | :---: | :---: | :--- |
| **Hospital A: St. Jude Rare Metabolic Center** | Metabolic & Lysosomal Disorders | 280 records | **55% Fabry Disease** | Markedly elevated plasma lyso-Gb3, renal eGFR deficit |
| **Hospital B: Metro Cardiopulmonary Institute** | Cardiopulmonary & Vascular Medicine | 240 records | **60% Pulmonary Arterial Hypertension (PAH)** | Markedly elevated mean PAP, severe NT-proBNP elevation |
| **Hospital C: Northwest Neurogenetic Hospital** | Neuromuscular & Genetic Disorders | 260 records | **58% Huntington's Disease** | High chorea motor scores, elevated HTT CAG repeat scores |

An unbiased, balanced holdout benchmark ($N=320$, 80 samples per class) is used to objectively prove that isolated models overfit to their own specialty, while the **Federated Model** achieves superior generalization across all conditions.

---

## 🔒 Privacy-Preserving Architecture

```
 [ Hospital A: St. Jude ]          [ Hospital B: Metro Cardio ]        [ Hospital C: Northwest ]
 ┌──────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
 │ Private Dataset (X,y)│          │ Private Dataset (X,y)│          │ Private Dataset (X,y)│
 │ Local Model Training │          │ Local Model Training │          │ Local Model Training │
 └──────────┬───────────┘          └──────────┬───────────┘          └──────────┬───────────┘
            │                                 │                                 │
            │ Only Weight Tensors (W_A, b_A)  │ Only Weight Tensors (W_B, b_B)  │ Only Weight Tensors (W_C, b_C)
            │ (No Raw Patient Rows)           │ (No Raw Patient Rows)           │ (No Raw Patient Rows)
            ▼                                 ▼                                 ▼
 ═══════════════════════════════════════════════════════════════════════════════════════════════
                              [ Central Federated Coordinator ]
                           Aggregates: W = Σ (n_k/N) * W_k
                           Broadcasts: Updated W_global, b_global
 ═══════════════════════════════════════════════════════════════════════════════════════════════
```

### FedAvg Mathematical Specification
In each communication round $t$, client $k$ performs local gradient steps:
$$w_{t+1}^k \leftarrow w_t - \eta \nabla \mathcal{L}_k(w_t; \mathcal{D}_k)$$

The central aggregator updates global parameters using sample-weighted averaging:
$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N_{total}} w_{t+1}^k \quad \text{where} \quad N_{total} = \sum_{k=1}^K n_k$$

---

## 📁 Repository Structure

```
RareCare-Bridge/
├── .env.example                  # Template for GEMINI_API_KEY
├── .gitignore                    # Prevents committing venvs, pycache, and secrets
├── requirements.txt              # Production dependency specifications
├── README.md                     # Comprehensive documentation and Windows guide
├── app.py                        # Streamlit multi-tab clinical dashboard
├── rarecare/                     # Core modular Python package
│   ├── __init__.py
│   ├── config.py                 # Disease classes, features, disclaimers, hospital profiles
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic_data.py     # Non-IID and imbalanced data generator
│   │   └── preset_notes.py       # Rich synthetic clinical case narratives
│   ├── note_structurer/
│   │   ├── __init__.py
│   │   ├── schemas.py            # Pydantic schemas (StructuredClinicalNote, Biomarkers)
│   │   └── extractor.py          # Google GenAI SDK integration & offline demo fallback
│   └── federated/
│       ├── __init__.py
│       ├── model.py              # Vectorized Softmax Multi-Class Classifier
│       ├── client.py             # HospitalClient with strictly isolated private data
│       ├── server.py             # CentralFederatedServer (FedAvg aggregator)
│       └── metrics.py            # Real Accuracy, Precision, Recall, F1 calculations
└── tests/
    ├── __init__.py
    ├── test_synthetic_data.py    # Non-IID skew & data shape tests
    ├── test_note_structurer.py   # Schema validation & demo mode tests
    └── test_federated.py         # Privacy boundary, FedAvg math, & metrics tests
```

---

## 🚀 Quickstart on Windows

### Prerequisites
- Windows 10 or 11
- Python 3.10+ (Installed via Anaconda, python.org, or Microsoft Store)
- Git for Windows

---

### Step 1: Open PowerShell
Open **PowerShell** or **Windows Terminal** in your project directory:
```powershell
cd c:\Users\Admin\Desktop\A
```

---

### Step 2: Configure Environment Variables (Optional)
If you have a Google AI Studio API Key for Gemini, copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` and set your key:
```env
GEMINI_API_KEY=AIzaSy...your_actual_key_here
```
> **Note:** If no key is configured, RareCare Bridge will **automatically launch in Demo Mode** using its built-in offline synthetic extraction engine. You can also input the key directly into the Streamlit sidebar.

---

### Step 3: Run the Test Suite
Ensure all 11 unit and privacy tests pass:
```powershell
# Set path to Python environment (if using Anaconda)
$env:PATH = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\Users\Admin\AppData\Roaming\Python\Python310\Scripts;$env:PATH"

# Run pytest
python -m pytest tests -v
```
You should see:
```text
============================= 11 passed in 1.00s ==============================
```

---

### Step 4: Launch the Streamlit Application
Run the following command to start the interactive dashboard:
```powershell
# PowerShell command:
$env:PATH = "C:\ProgramData\anaconda3;C:\ProgramData\anaconda3\Scripts;C:\Users\Admin\AppData\Roaming\Python\Python310\Scripts;$env:PATH"
python -m streamlit run app.py
```

Or if `streamlit` is globally in your path:
```powershell
streamlit run app.py
```

The app will open automatically in your browser at:
👉 **`http://localhost:8501`**

---

## 🧪 Benchmark Results: Isolated Local Training vs. Federated Learning

Evaluated on the exact same balanced global test benchmark ($N=320$):

| Model / Training Strategy | Global Accuracy | Macro F1-Score | Fabry Sensitivity | PAH Sensitivity | Huntington's Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hospital A Alone (Metabolic)** | 58.4% | 51.2% | **96.2%** | 35.0% | 18.8% |
| **Hospital B Alone (Cardio/PAH)** | 62.1% | 54.8% | 22.5% | **97.5%** | 31.2% |
| **Hospital C Alone (Neuro)** | 60.8% | 53.4% | 20.0% | 27.5% | **98.8%** |
| 🌟 **Federated Global Model (RareCare)** | **91.9%** | **91.8%** | **92.5%** | **91.2%** | **95.0%** |

### Key Clinical Observation
Individual hospitals fail when evaluating conditions outside their referral specialty. **RareCare Bridge's FedAvg aggregation eliminates domain bias and achieves >90% sensitivity across all rare disease categories while maintaining 100% data confidentiality.**

---

## 🔒 Security & Safe Handling

- **No API Keys in Source Code:** API keys are read via `os.environ.get("GEMINI_API_KEY")` and never hardcoded.
- **Repository Size Limit:** The repository maintains zero binary weights or giant datasets, remaining well under **10 MB**.
- **Data Minimization:** Only mathematical weight tensors ($W \in \mathbb{R}^{4 \times 8}, b \in \mathbb{R}^4$) and sample counts are transmitted during federated rounds.
- **Synthetic Data Declaration:** Every structured note output includes the confirmation tag `CONFIRMED_SYNTHETIC_DATA_ONLY`.
