# 🩺 RareCare Bridge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Google GenAI SDK](https://img.shields.io/badge/Google_GenAI_SDK-Official-4285F4.svg)](https://github.com/googleapis/python-genai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Gemini-Powered Clinical Note Structuring & Privacy-Preserving Federated Learning for Rare Diseases.**

RareCare Bridge is a healthcare information platform designed to solve the rare disease diagnostic dilemma. Rare diseases affect over 300 million people worldwide, but fragmented health systems, severe class imbalances, and strict data privacy regulations (HIPAA, GDPR) prevent cross-institutional data sharing.

RareCare Bridge bridges this gap with two synchronized systems:
1. **Clinical Note Structurer**: Uses the official `google-genai` SDK and Gemini models to transform complex unstructured EHR consultation notes into structured, standardized clinical entities (phenotypes, HPO terms, lab flags, suspected rare conditions, and next diagnostic steps).
2. **Federated Learning Simulator**: Coordinates simulated hospital networks (Metropolitan, St. Jude, and Valley Children's) with non-IID, highly imbalanced rare disease cohorts to collaboratively train diagnostic models via **Federated Averaging (FedAvg)**. Patient records never leave local hospital nodes; only mathematical parameter updates are exchanged.

---

## 🌟 Key Features

- **Official Google GenAI SDK**: Built using `from google import genai` with structured Pydantic schema validation.
- **Graceful Demo Mode**: Automatically activates high-fidelity simulated extractions if no `GEMINI_API_KEY` is present.
- **Strict Institutional Privacy Boundary**: Hospital client nodes hold synthetic patient data locally. Aggregation functions accept only parameter weights ($\Delta W, \Delta b$) and reject any raw data matrices.
- **3 Simulated Non-IID Hospital Cohorts**:
  - **Metropolitan Rare Disease Institute**: Adult genetics cohort, high prevalence of lysosomal disorders (Gaucher, Fabry).
  - **St. Jude Regional Healthcare**: General community hospital with severe class imbalance (~6% rare cases).
  - **Valley Children's Research Center**: Pediatric metabolic center with extreme biomarker skews (Pompe disease).
- **Real Calculated Metrics**: Accuracy, Precision, Sensitivity/Recall, F1-Score, and ROC-AUC computed on a unified multicenter held-out benchmark using `scikit-learn`.
- **Comparative Benchmarking**: Direct side-by-side comparison illustrating how isolated local models fail due to population skew while the Federated Global Model excels across all demographics.
- **Clinical Governance & Safety**: Prominent medical disclaimers, input length and type validation, and zero claims of clinical validation.

---

## 🏗️ Project Architecture

```
rarecare-bridge/
├── .env.example              # Template for GEMINI_API_KEY
├── .gitignore                # Git exclusions (venv, cache, env)
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation and user guide
├── app.py                    # Streamlit healthcare dashboard entry point
├── src/
│   ├── __init__.py
│   ├── config.py             # Settings, hospital metadata, key loaders
│   ├── structuring/          # Clinical NLP Structuring
│   │   ├── __init__.py
│   │   ├── schemas.py        # Pydantic structured output models
│   │   ├── sample_notes.py   # Authentic synthetic clinical notes
│   │   └── extractor.py      # Google GenAI SDK & Demo Mode extractor
│   ├── federated/            # Federated Learning System
│   │   ├── __init__.py
│   │   ├── data_generator.py # 3 Non-IID hospital cohorts + global test set
│   │   ├── client.py         # Local hospital client (isolated patient data)
│   │   ├── server.py         # FederatedServer & FedAvg aggregator
│   │   └── evaluator.py      # Real scikit-learn metrics computation
│   └── ui/                   # Healthcare UI Layer
│       ├── __init__.py
│       ├── styles.py         # Custom clinical CSS design tokens
│       └── components.py     # Metric cards, hospital cards, diagrams
└── tests/                    # Automated Test Suite
    ├── __init__.py
    ├── test_structuring.py   # NLP schemas, demo extractor, input guards
    ├── test_federated.py     # Non-IID data gen, client training, FedAvg
    └── test_privacy.py       # Client isolation & privacy violation rejection
```

---

## 🚀 Windows Installation & Setup Guide

### 1. Prerequisites
- Windows 10/11
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 (64-bit)
- PowerShell or Command Prompt

### 2. Open PowerShell in Project Directory
```powershell
cd c:\Users\NMIT\Desktop\A\rarecare-bridge
```

### 3. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate in PowerShell
.\.venv\Scripts\Activate.ps1

# (Optional: If PowerShell execution policy blocks activation)
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\.venv\Scripts\Activate.ps1

# Or in traditional Windows Command Prompt (cmd.exe):
# .\.venv\Scripts\activate.bat
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Gemini API Key (Optional for Live AI Extraction)
Copy `.env.example` to `.env` and insert your Gemini API Key from [Google AI Studio](https://aistudio.google.com/):
```powershell
copy .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=AIzaSyYourActualKeyHere
```
*Note: If no key is configured, the application automatically runs in high-fidelity **Demo Mode** with simulated extractions.*

### 6. Run the Streamlit Application
```powershell
streamlit run app.py
```
Or run directly via the virtual environment's executable:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser at **http://localhost:8501**.

### 7. Run Automated Tests
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 🔒 Privacy & Federated Learning Protocol

In standard centralized machine learning, sensitive patient EHR records must be pooled into a central cloud database:

$$\mathcal{D}_{\text{central}} = \mathcal{D}_1 \cup \mathcal{D}_2 \cup \mathcal{D}_3 \quad \text{(HIPAA/GDPR Risk)}$$

In **RareCare Bridge**, patient data never leaves individual hospital clients:

$$W_{t+1} = \sum_{k=1}^K \frac{n_k}{N} W_{t+1}^k \quad \text{where} \quad N = \sum_{k=1}^K n_k$$

Each hospital node trains locally on private data using regularized gradient steps and transmits **only numerical parameter vectors** ($\Delta W \in \mathbb{R}^8, \Delta b \in \mathbb{R}$) to the server coordinator.

---

## ⚠️ Medical & Regulatory Disclaimer

**RareCare Bridge is an informational research and demonstration platform developed for AI capability exploration and federated learning prototyping. It is NOT a medical device, does NOT provide medical advice, diagnosis, prognosis, or treatment recommendations, and has NOT undergone clinical trials or regulatory approval (FDA, CE, or equivalent). All medical notes processed must be strictly synthetic or fully de-identified. Always consult a qualified, licensed healthcare provider for any medical decisions or emergency situations.**
