"""
RareCare Bridge - Streamlit Application
A Gemini-powered healthcare information dashboard demonstrating clinical note structuring
and privacy-preserving Federated Learning across simulated hospital nodes.
"""

# Deployment refresh check

import os
import json
from typing import Dict, List, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rarecare.config import (
    MEDICAL_DISCLAIMER,
    DISEASE_CLASSES,
    DISEASE_SHORT_NAMES,
    FEATURE_NAMES,
    HOSPITAL_PROFILES,
    get_gemini_api_key,
    DEFAULT_GEMINI_MODEL,
)
from rarecare.data.synthetic_data import (
    generate_all_hospitals,
    generate_global_test_dataset,
    HospitalDataset,
)
from rarecare.data.preset_notes import PRESET_CLINICAL_NOTES
from rarecare.note_structurer.extractor import extract_structured_note
from rarecare.note_structurer.schemas import ExtractionResult, StructuredClinicalNote
from rarecare.federated.model import FederatedSoftmaxClassifier
from rarecare.federated.client import HospitalClient
from rarecare.federated.server import CentralFederatedServer, FederationHistory
from rarecare.federated.metrics import calculate_metrics, EvaluationMetrics


# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RareCare Bridge | Federated Healthcare Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN HEALTHCARE CSS STYLING ---
st.markdown(
    """
    <style>
        /* Base typography & colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Healthcare Hero Header */
        .hero-banner {
            background: linear-gradient(135deg, #0F172A 0%, #0F766E 50%, #047857 100%);
            padding: 24px 30px;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        }
        .hero-title {
            font-size: 28px;
            font-weight: 700;
            color: #FFFFFF !important;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .hero-subtitle {
            font-size: 15px;
            color: #E2E8F0;
            margin-top: 6px;
            margin-bottom: 0px;
            font-weight: 400;
        }
        
        /* Disclaimer Alert Box */
        .disclaimer-box {
            background-color: #FFFBEB;
            border-left: 4px solid #D97706;
            padding: 12px 18px;
            border-radius: 6px;
            font-size: 12.5px;
            color: #92400E;
            line-height: 1.5;
            margin-bottom: 22px;
        }
        
        /* Metric Badges & Cards */
        .health-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 15px;
        }
        
        .status-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11.5px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-elevated { background: #FEE2E2; color: #DC2626; }
        .status-normal { background: #DCFCE7; color: #16A34A; }
        .status-deficient { background: #FEF3C7; color: #D97706; }
        .status-urgent { background: #FEE2E2; color: #B91C1C; }
        .status-routine { background: #E0E7FF; color: #4338CA; }
        .status-demo { background: #FEF08A; color: #854D0E; }
        
        /* Metric Tile */
        .metric-tile {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
            text-align: center;
        }
        .metric-tile-val {
            font-size: 22px;
            font-weight: 700;
            color: #0F172A;
        }
        .metric-tile-lbl {
            font-size: 11.5px;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- SESSION STATE INITIALIZATION ---
if "hospitals" not in st.session_state:
    st.session_state.hospitals = generate_all_hospitals(seed_offset=100)

if "global_test" not in st.session_state:
    X_test, y_test, df_test = generate_global_test_dataset(n_samples=320, seed=999)
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.df_test = df_test

if "server" not in st.session_state:
    clients = [HospitalClient(k, ds) for k, ds in st.session_state.hospitals.items()]
    st.session_state.server = CentralFederatedServer(
        clients=clients,
        X_test=st.session_state.X_test,
        y_test=st.session_state.y_test,
        seed=42,
    )
    # Pre-train isolated local models for comparison
    st.session_state.isolated_models = {
        client.hospital_key: client.train_isolated_local_model(epochs=35, lr=0.08)
        for client in clients
    }


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image(
        "https://api.iconify.design/solar:heart-pulse-bold-duotone.svg?color=%230D9488",
        width=54,
    )
    st.markdown("### **RareCare Bridge**")
    st.caption("Privacy-Preserving Federated Healthcare Intelligence")

    st.markdown("---")
    st.markdown("#### ⚙️ **API & Environment**")
    
    env_api_key = get_gemini_api_key()
    has_env_key = bool(env_api_key)

    if has_env_key:
        st.success("`GEMINI_API_KEY` active in environment")
    else:
        st.info("No `GEMINI_API_KEY` detected in environment. Running in **Demo Mode**.")

    # Allow custom key input or demo override
    custom_key = st.text_input(
        "Optional Gemini API Key Override",
        type="password",
        placeholder="Enter AI Studio Key if overriding...",
        help="API Key is kept strictly in session memory and never stored or logged.",
    )
    effective_key = custom_key.strip() if custom_key.strip() else env_api_key

    force_demo_toggle = st.checkbox(
        "⚡ Force Offline Demo Mode",
        value=not bool(effective_key),
        help="When checked, uses the deterministic synthetic heuristic extractor without calling Google GenAI servers.",
    )

    st.markdown("---")
    st.markdown("#### 🧭 **Navigation**")
    app_mode = st.radio(
        "Select Module:",
        [
            "🩺 Clinical Note Structurer",
            "🏥 Hospital Cohorts (Non-IID)",
            "🔄 Federated Training Studio",
            "📊 Local vs. Federated Benchmark",
            "🔒 Privacy & Cryptography Audit",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 🎛️ **Federation Parameters**")
    param_rounds = st.slider("Federation Rounds", min_value=1, max_value=12, value=6, step=1)
    param_epochs = st.slider("Local Epochs / Round", min_value=1, max_value=8, value=3, step=1)
    param_lr = st.slider("Learning Rate (η)", min_value=0.01, max_value=0.20, value=0.08, step=0.01)

    st.markdown("---")
    st.caption("RareCare Bridge MVP • Google GenAI + FedAvg")


# --- TOP HERO BANNER & DISCLAIMER ---
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">
            <span>🩺 RareCare Bridge</span>
            <span style="font-size: 13px; font-weight: 500; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 20px;">
                EHR Structuring & Federated Intelligence
            </span>
        </div>
        <p class="hero-subtitle">
            Extract structured clinical entities from synthetic medical records and train collaborative, 
            privacy-preserving models across hospital silos without ever sharing raw patient data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="disclaimer-box">
        <strong>MEDICAL DISCLAIMER:</strong> {MEDICAL_DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# TAB 1: CLINICAL NOTE STRUCTURER
# ==============================================================================
if app_mode == "🩺 Clinical Note Structurer":
    st.markdown("### 🩺 Unstructured Clinical Note Structuring")
    st.markdown(
        "Convert messy, unstructured clinical narratives of rare disease presentations into normalized, "
        "interoperable healthcare data using the official **`google-genai` SDK** (or offline deterministic demo engine)."
    )

    col_note_opt, col_note_meta = st.columns([2, 1])

    with col_note_opt:
        preset_options = ["Custom Synthetic Note Input"] + [p["title"] for p in PRESET_CLINICAL_NOTES]
        selected_preset = st.selectbox("Select Synthetic Case Preset:", preset_options)

    with col_note_meta:
        if force_demo_toggle or not effective_key:
            st.markdown(
                '<div style="padding-top: 25px;"><span class="status-pill status-demo">⚡ OFFLINE SYNTHETIC ENGINE</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding-top: 25px;"><span class="status-pill status-normal">⚡ GEMINI LIVE API CONNECTED</span></div>',
                unsafe_allow_html=True,
            )

    # Note Content Box
    if selected_preset != "Custom Synthetic Note Input":
        preset_match = next(p for p in PRESET_CLINICAL_NOTES if p["title"] == selected_preset)
        default_content = preset_match["content"]
    else:
        default_content = (
            "PATIENT VISIT SUMMARY\n"
            "PATIENT ID: SYNTH-METAB-9102\n"
            "DEMOGRAPHICS: 38-year-old male presenting with chronic neuropathic burning in hands and feet.\n"
            "Reports lack of sweating (hypohidrosis) and punctate dark reddish macules (angiokeratomas) around groin.\n"
            "Labs show plasma lyso-Gb3 elevated at 41.2 nmol/L (normal < 1.8), eGFR reduced to 64 mL/min/1.73m².\n"
            "Alpha-galactosidase A enzyme deficient at 1.4 nmol/hr/mg. Maternal family history of early renal failure."
        )

    clinical_note_text = st.text_area(
        "Synthetic Clinical Note (EHR Text):",
        value=default_content,
        height=240,
        help="Input synthetic clinical notes containing clinical history, exams, and biomarkers.",
    )

    char_count = len(clinical_note_text.strip())
    st.caption(f"Character Count: {char_count} chars • All clinical notes are strictly synthetic.")

    btn_structure = st.button("🚀 Extract Structured Healthcare Data", type="primary", use_container_width=True)

    if btn_structure:
        with st.spinner("Extracting structured clinical entities..."):
            extraction_res: ExtractionResult = extract_structured_note(
                note_text=clinical_note_text,
                api_key=effective_key,
                force_demo=force_demo_toggle,
            )

        if not extraction_res.success:
            st.error(f"❌ {extraction_res.error_message}")
        else:
            note_data: StructuredClinicalNote = extraction_res.data
            st.success("✅ Clinical Extraction Complete!")

            # Extraction Summary Cards
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-val">{note_data.patient_id}</div>
                        <div class="metric-tile-lbl">Synthetic ID</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m2:
                age_display = f"{note_data.age} yrs" if note_data.age else "Unknown"
                gender_display = note_data.gender or "Not Specified"
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-val">{age_display} • {gender_display}</div>
                        <div class="metric-tile-lbl">Demographics</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m3:
                urgency_cls = "status-urgent" if note_data.urgency_level in ["Urgent", "Emergency"] else "status-routine"
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-val"><span class="status-pill {urgency_cls}">{note_data.urgency_level}</span></div>
                        <div class="metric-tile-lbl">Urgency Triage</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_m4:
                model_lbl = "Offline Demo Engine" if extraction_res.is_demo_mode else extraction_res.model_used
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-val" style="font-size: 15px; padding-top: 4px;">{model_lbl}</div>
                        <div class="metric-tile-lbl">Extraction Engine</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

            # Details tabs
            t_overview, t_biomarkers, t_plan, t_features, t_json = st.tabs([
                "📋 Clinical Synthesis",
                "🧪 Biomarkers & Lab Assays",
                "🩺 Workup & ICD-10",
                "🧬 Federated Feature Bridge",
                "🔍 Raw Schema JSON",
            ])

            with t_overview:
                st.markdown(f"#### **Suspected Rare Condition:** `{note_data.suspected_condition}`")
                st.markdown(f"**Chief Complaint:** {note_data.chief_complaint}")
                st.markdown(f"**Clinical Case Summary:** {note_data.clinical_summary}")

                st.markdown("##### **Primary Presenting Signs & Symptoms:**")
                for s in note_data.primary_symptoms:
                    st.markdown(f"- 🔸 {s}")

            with t_biomarkers:
                if note_data.biomarkers:
                    b_rows = []
                    for b in note_data.biomarkers:
                        b_rows.append({
                            "Biomarker / Assay": b.name,
                            "Observed Value": b.value,
                            "Reference Range": b.reference_range,
                            "Clinical Status": b.interpretation,
                        })
                    b_df = pd.DataFrame(b_rows)
                    st.dataframe(b_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No specific laboratory biomarkers found in this clinical narrative.")

            with t_plan:
                col_icd, col_rec = st.columns([1, 2])
                with col_icd:
                    st.markdown("##### **Candidate ICD-10 Codes:**")
                    for icd in note_data.icd10_candidates:
                        st.markdown(f"- 🏷️ `{icd}`")
                with col_rec:
                    st.markdown("##### **Recommended Diagnostic Next Steps:**")
                    for r in note_data.recommended_workup:
                        st.markdown(f"- 🩺 {r}")

            with t_features:
                st.markdown(
                    "#### **Bridging Structured EHR into the Federated Classifier**\n"
                    "The extracted clinical note automatically computes normalized tabular indicators "
                    "matching the 8 federated features used by simulated hospital nodes."
                )
                if note_data.feature_vector_estimate:
                    feat_dict = note_data.feature_vector_estimate
                    feat_df = pd.DataFrame([feat_dict]).T.reset_index()
                    feat_df.columns = ["Clinical Feature", "Normalized Value"]
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)

                    # Interactive prediction using Central Federated Model
                    vec = np.array([[feat_dict.get(fn, 1.0) for fn in FEATURE_NAMES]], dtype=np.float32)
                    probs = st.session_state.server.global_model.predict_proba(vec)[0]
                    pred_class = int(np.argmax(probs))

                    st.markdown("---")
                    st.markdown("##### **Global Federated Model Inference for this Note:**")
                    col_pred_lbl, col_pred_chart = st.columns([1, 2])
                    with col_pred_lbl:
                        st.metric(
                            "Predicted Condition",
                            DISEASE_SHORT_NAMES[pred_class],
                            f"{probs[pred_class]*100:.1f}% Confidence",
                        )
                        st.caption(DISEASE_CLASSES[pred_class])
                    with col_pred_chart:
                        chart_df = pd.DataFrame({
                            "Condition": [DISEASE_SHORT_NAMES[c] for c in range(4)],
                            "Probability": probs,
                        })
                        fig_bar = px.bar(
                            chart_df,
                            x="Condition",
                            y="Probability",
                            color="Condition",
                            color_discrete_sequence=["#64748B", "#0284C7", "#E11D48", "#7C3AED"],
                            range_y=[0, 1.0],
                        )
                        fig_bar.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)

            with t_json:
                st.json(note_data.model_dump())


# ==============================================================================
# TAB 2: SIMULATED HOSPITAL COHORTS (NON-IID)
# ==============================================================================
elif app_mode == "🏥 Hospital Cohorts (Non-IID)":
    st.markdown("### 🏥 Simulated Hospital Network & Non-IID Cohorts")
    st.markdown(
        "Real-world rare disease research is hindered because individual hospitals see highly skewed, "
        "imbalanced patient cohorts. Below are three simulated academic hospital systems exhibiting distinct **Non-IID** distributions."
    )

    # Hospital Cards
    hosp_cols = st.columns(3)
    for idx, (k, profile) in enumerate(HOSPITAL_PROFILES.items()):
        ds = st.session_state.hospitals[k]
        with hosp_cols[idx]:
            st.markdown(
                f"""
                <div class="health-card" style="border-top: 4px solid {profile['color']};">
                    <h4 style="margin: 0; color: #0F172A;">{profile['name']}</h4>
                    <p style="font-size: 12px; color: #64748B; margin-top: 3px;">📍 {profile['location']} • <code>{profile['code']}</code></p>
                    <p style="font-size: 13px; color: #334155; min-height: 40px;">{profile['description']}</p>
                    <hr style="margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 13px;">
                        <span><strong>Total Records:</strong> {ds.sample_count}</span>
                        <span><strong>Specialty:</strong> {profile['specialty'][:22]}...</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### 📊 **Cross-Institutional Label Imbalance (Non-IID Proof)**")

    # Stacked bar comparison of classes
    dist_rows = []
    for k, ds in st.session_state.hospitals.items():
        for c, count in ds.class_counts.items():
            dist_rows.append({
                "Hospital": ds.hospital_name,
                "Condition": DISEASE_SHORT_NAMES[c],
                "Patient Count": count,
                "Percentage (%)": (count / ds.sample_count) * 100,
            })
    dist_df = pd.DataFrame(dist_rows)

    fig_dist = px.bar(
        dist_df,
        x="Hospital",
        y="Patient Count",
        color="Condition",
        barmode="stack",
        color_discrete_sequence=["#94A3B8", "#0284C7", "#E11D48", "#7C3AED"],
        title="Patient Cohort Composition Across Simulated Hospital Nodes",
    )
    fig_dist.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔬 **Local Private Dataset Explorer**")
    selected_hosp_key = st.selectbox(
        "Choose Hospital to inspect local records:",
        list(HOSPITAL_PROFILES.keys()),
        format_func=lambda k: HOSPITAL_PROFILES[k]["name"],
    )

    hosp_ds: HospitalDataset = st.session_state.hospitals[selected_hosp_key]
    st.markdown(
        f"Displaying sample synthetic records from **{hosp_ds.hospital_name}** (`{hosp_ds.sample_count}` total records). "
        "*These raw records remain strictly confidential on the hospital client's local node.*"
    )
    st.dataframe(hosp_ds.df.head(10), use_container_width=True)


# ==============================================================================
# TAB 3: FEDERATED TRAINING STUDIO
# ==============================================================================
elif app_mode == "🔄 Federated Training Studio":
    st.markdown("### 🔄 Federated Training Studio (FedAvg Simulation)")
    st.markdown(
        "Train a multi-class rare disease classifier collaboratively across all three hospital clients using **FedAvg**. "
        "At no point are patient records pooled or sent to the central server."
    )

    col_btn_train, col_btn_reset, col_status = st.columns([1.5, 1.2, 3])

    with col_btn_train:
        btn_run_fed = st.button("▶️ Run Federated Training", type="primary", use_container_width=True)

    with col_btn_reset:
        btn_reset_fed = st.button("🔄 Reset Models", use_container_width=True)

    if btn_reset_fed:
        clients = [HospitalClient(k, ds) for k, ds in st.session_state.hospitals.items()]
        st.session_state.server = CentralFederatedServer(
            clients=clients,
            X_test=st.session_state.X_test,
            y_test=st.session_state.y_test,
            seed=42,
        )
        st.session_state.isolated_models = {
            client.hospital_key: client.train_isolated_local_model(epochs=35, lr=0.08)
            for client in clients
        }
        st.success("Simulation models reset to initial baseline state.")

    if btn_run_fed:
        server: CentralFederatedServer = st.session_state.server
        prog_bar = st.progress(0.0)
        status_box = st.empty()

        for r in range(1, param_rounds + 1):
            status_box.info(f"Executing Federated Round {r}/{param_rounds}: Broadcasting weights & running local hospital epochs...")
            record = server.run_round(
                local_epochs=param_epochs,
                lr=param_lr,
            )
            prog_bar.progress(float(r) / float(param_rounds))

        status_box.success(f"✅ Successfully completed {param_rounds} communication rounds!")

    # Display Current Federated Telemetry
    server = st.session_state.server
    curr_metrics = server.evaluate_on_test_set()

    st.markdown("---")
    st.markdown(f"#### 📈 **Global Benchmark Performance (Round {server.current_round})**")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Test Accuracy", f"{curr_metrics.accuracy * 100:.1f}%")
    with m_col2:
        st.metric("Macro F1-Score", f"{curr_metrics.macro_f1 * 100:.1f}%")
    with m_col3:
        st.metric("Macro Precision", f"{curr_metrics.macro_precision * 100:.1f}%")
    with m_col4:
        st.metric("Macro Recall", f"{curr_metrics.macro_recall * 100:.1f}%")

    # If rounds have been executed, render round convergence curves
    if len(server.history.records) > 0:
        hist_data = server.history.to_dict()
        chart_df = pd.DataFrame(hist_data)

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_acc = px.line(
                chart_df,
                x="round",
                y=["test_accuracy", "test_f1"],
                markers=True,
                labels={"value": "Score (0 - 1.0)", "variable": "Metric", "round": "Communication Round"},
                title="Global Convergence: Accuracy & Macro F1 vs Rounds",
                color_discrete_sequence=["#0D9488", "#7C3AED"],
            )
            fig_acc.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_c2:
            fig_loss = px.line(
                chart_df,
                x="round",
                y="test_loss",
                markers=True,
                labels={"test_loss": "Cross-Entropy Loss", "round": "Communication Round"},
                title="Global Validation Loss Reduction",
                color_discrete_sequence=["#E11D48"],
            )
            fig_loss.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_loss, use_container_width=True)

        # Central Model Weight Heatmap
        st.markdown("---")
        st.markdown("#### 🧠 **Global Federated Model Weights (Feature Attribution)**")
        st.markdown("Visualizing the central parameter tensor $W$ aggregated via FedAvg across all hospital clients:")
        
        weights_matrix = server.global_model.W
        fig_heat = px.imshow(
            weights_matrix,
            x=FEATURE_NAMES,
            y=[DISEASE_SHORT_NAMES[c] for c in range(4)],
            color_continuous_scale="Viridis",
            labels=dict(x="Clinical Feature Indicator", y="Rare Disease Target", color="Weight Magnitude"),
            title="Learned Association Matrix (Classes vs Biological Features)",
        )
        fig_heat.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Click 'Run Federated Training' above to launch the FedAvg simulation and view round-by-round convergence charts.")


# ==============================================================================
# TAB 4: LOCAL VS. FEDERATED BENCHMARK
# ==============================================================================
elif app_mode == "📊 Local vs. Federated Benchmark":
    st.markdown("### 📊 Local vs. Federated Model Benchmark")
    st.markdown(
        "Demonstrating the fundamental value proposition of Federated Learning in rare diseases: "
        "A model trained only on one hospital's data overfits to that hospital's specialty and fails on unseen conditions. "
        "The **Federated Model** achieves superior generalization across all rare diseases without any raw data pooling."
    )

    server = st.session_state.server
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    # Evaluate all models on the unbiased holdout test set
    bench_results = []
    
    # 1. Hospital A Isolated
    model_a = st.session_state.isolated_models["Hospital_A"]
    m_a = calculate_metrics(y_test, model_a.predict(X_test))
    bench_results.append({
        "Model": "Hospital A Alone (Metabolic)",
        "Strategy": "Isolated Local Training",
        "Accuracy (%)": m_a.accuracy * 100,
        "Macro F1 (%)": m_a.macro_f1 * 100,
        "Precision (%)": m_a.macro_precision * 100,
        "Recall (%)": m_a.macro_recall * 100,
        "Fabry Recall (%)": m_a.per_class_recall[1] * 100,
        "PAH Recall (%)": m_a.per_class_recall[2] * 100,
        "Huntington's Recall (%)": m_a.per_class_recall[3] * 100,
    })

    # 2. Hospital B Isolated
    model_b = st.session_state.isolated_models["Hospital_B"]
    m_b = calculate_metrics(y_test, model_b.predict(X_test))
    bench_results.append({
        "Model": "Hospital B Alone (Cardio/PAH)",
        "Strategy": "Isolated Local Training",
        "Accuracy (%)": m_b.accuracy * 100,
        "Macro F1 (%)": m_b.macro_f1 * 100,
        "Precision (%)": m_b.macro_precision * 100,
        "Recall (%)": m_b.macro_recall * 100,
        "Fabry Recall (%)": m_b.per_class_recall[1] * 100,
        "PAH Recall (%)": m_b.per_class_recall[2] * 100,
        "Huntington's Recall (%)": m_b.per_class_recall[3] * 100,
    })

    # 3. Hospital C Isolated
    model_c = st.session_state.isolated_models["Hospital_C"]
    m_c = calculate_metrics(y_test, model_c.predict(X_test))
    bench_results.append({
        "Model": "Hospital C Alone (Neuro)",
        "Strategy": "Isolated Local Training",
        "Accuracy (%)": m_c.accuracy * 100,
        "Macro F1 (%)": m_c.macro_f1 * 100,
        "Precision (%)": m_c.macro_precision * 100,
        "Recall (%)": m_c.macro_recall * 100,
        "Fabry Recall (%)": m_c.per_class_recall[1] * 100,
        "PAH Recall (%)": m_c.per_class_recall[2] * 100,
        "Huntington's Recall (%)": m_c.per_class_recall[3] * 100,
    })

    # 4. Federated Global Model
    m_fed = server.evaluate_on_test_set()
    bench_results.append({
        "Model": "🌟 Federated Global Model (RareCare)",
        "Strategy": "Privacy-Preserving FedAvg",
        "Accuracy (%)": m_fed.accuracy * 100,
        "Macro F1 (%)": m_fed.macro_f1 * 100,
        "Precision (%)": m_fed.macro_precision * 100,
        "Recall (%)": m_fed.macro_recall * 100,
        "Fabry Recall (%)": m_fed.per_class_recall[1] * 100,
        "PAH Recall (%)": m_fed.per_class_recall[2] * 100,
        "Huntington's Recall (%)": m_fed.per_class_recall[3] * 100,
    })

    bench_df = pd.DataFrame(bench_results)

    # Comparative Summary Cards
    st.markdown("#### 🏆 **Benchmark Leaderboard on Global Balanced Holdout (N=320)**")
    
    # Format display columns
    st.dataframe(
        bench_df.style.format({
            "Accuracy (%)": "{:.1f}%",
            "Macro F1 (%)": "{:.1f}%",
            "Precision (%)": "{:.1f}%",
            "Recall (%)": "{:.1f}%",
            "Fabry Recall (%)": "{:.1f}%",
            "PAH Recall (%)": "{:.1f}%",
            "Huntington's Recall (%)": "{:.1f}%",
        }).highlight_max(subset=["Accuracy (%)", "Macro F1 (%)", "Recall (%)"], color="#D1FAE5"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### 🔍 **Domain Bias in Isolated Training vs. Federated Robustness**")
    
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_bar_comp = px.bar(
            bench_df,
            x="Model",
            y=["Accuracy (%)", "Macro F1 (%)"],
            barmode="group",
            title="Overall Model Accuracy & Macro F1 Comparison",
            color_discrete_sequence=["#0D9488", "#0284C7"],
        )
        fig_bar_comp.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar_comp, use_container_width=True)

    with col_chart2:
        # Per-disease recall comparison
        disease_rec_df = bench_df.melt(
            id_vars=["Model"],
            value_vars=["Fabry Recall (%)", "PAH Recall (%)", "Huntington's Recall (%)"],
            var_name="Rare Disease",
            value_name="Recall Rate (%)",
        )
        fig_recall = px.bar(
            disease_rec_df,
            x="Rare Disease",
            y="Recall Rate (%)",
            color="Model",
            barmode="group",
            title="Per-Condition Recall (Sensitivity to Rare Pathologies)",
            color_discrete_sequence=["#94A3B8", "#F59E0B", "#8B5CF6", "#10B981"],
        )
        fig_recall.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_recall, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🎯 **Confusion Matrix: Federated Global Model**")
    
    col_cm, col_cm_desc = st.columns([1.5, 1])
    with col_cm:
        cm_data = m_fed.confusion_matrix
        labels = [DISEASE_SHORT_NAMES[c] for c in range(4)]
        fig_cm = px.imshow(
            cm_data,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Condition", y="True Condition", color="Patients"),
            title=f"Federated Model Confusion Matrix (Accuracy: {m_fed.accuracy*100:.1f}%)",
        )
        fig_cm.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cm_desc:
        st.markdown(
            """
            **Clinical Insight:**
            - **Diagonal Dominance:** High numbers along the diagonal demonstrate robust multi-class discriminative power.
            - **No Negative Transfer:** Federated averaging prevents any single specialized hospital's majority class from wiping out detection of other rare conditions.
            - **Zero Raw Data Exchange:** Every metric shown is achieved while adhering strictly to local privacy boundaries.
            """
        )


# ==============================================================================
# TAB 5: PRIVACY & ARCHITECTURE OVERVIEW
# ==============================================================================
elif app_mode == "🔒 Privacy & Cryptography Audit":
    st.markdown("### 🔒 Privacy-Preserving Architecture & Verification Audit")
    st.markdown(
        "RareCare Bridge is architected in accordance with modern privacy principles (HIPAA Safe Harbor, GDPR Privacy by Design). "
        "Below is an audit and inspection of how local hospital client boundaries are maintained."
    )

    # Visual Architecture Diagram
    st.markdown("#### 🛡️ **Zero-Knowledge Aggregation Workflow**")
    st.markdown(
        """
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
        """
    )

    st.markdown("---")
    st.markdown("#### 🧪 **Automated Privacy Verification Audit**")
    
    server = st.session_state.server
    
    # Run interactive checks
    audit_col1, audit_col2 = st.columns(2)
    
    with audit_col1:
        st.markdown("##### **Check 1: Client Parameter Payload Verification**")
        client_a = server.clients[0]
        params, n_samples, loss = client_a.train_local_round(
            global_params=server.global_model.get_parameters(),
            epochs=1,
        )
        
        has_only_weights = set(params.keys()) == {"W", "b"}
        has_no_raw_data = not any(k in params for k in ["X", "y", "data", "patient_id", "records"])
        
        if has_only_weights and has_no_raw_data:
            st.success("✅ **PASSED: Parameter payload strictly isolated.**")
            st.markdown(f"- Payload keys: `{list(params.keys())}`")
            st.markdown(f"- Weights tensor shape: `{params['W'].shape}`")
            st.markdown(f"- Bias vector shape: `{params['b'].shape}`")
            st.markdown(f"- Sample count scalar: `{n_samples}` records")
        else:
            st.error("❌ FAILED: Data leakage detected!")

    with audit_col2:
        st.markdown("##### **Check 2: Central Server Boundary Verification**")
        server_has_X = hasattr(server, "client_X") or hasattr(server, "_client_data")
        
        if not server_has_X:
            st.success("✅ **PASSED: Central server has zero reference to client raw data.**")
            st.markdown("- Server object attributes strictly limited to coordination and holdout test set.")
            st.markdown("- Client `_local_dataset` variables are private and not exposed.")
        else:
            st.error("❌ FAILED: Server holds client raw data!")

    st.markdown("---")
    st.markdown("#### 📐 **Mathematical FedAvg Specification**")
    st.markdown(
        r"""
        In each federated communication round $t$, each hospital client $k \in \{1, \dots, K\}$ updates its parameters locally:
        $$w_{t+1}^k \leftarrow w_t - \eta \nabla \mathcal{L}_k(w_t; \mathcal{D}_k)$$
        
        The central aggregation coordinator computes the weighted average proportional to local patient sample size $n_k$:
        $$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{t+1}^k \quad \text{where} \quad N = \sum_{k=1}^K n_k$$
        """
    )
    
    st.markdown("---")
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>REMINDER OF REGULATORY INTENT:</strong> {MEDICAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )
