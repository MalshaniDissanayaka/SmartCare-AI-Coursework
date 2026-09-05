"""
SmartCare Hospital — 30-Day Readmission Risk Decision-Support Prototype
=======================================================================

CCS3440 Artificial Intelligence Coursework — Task 08

Run locally:
    streamlit run app.py

Expected model files:
    readmission_model.pkl
    model_metadata.json

The application:
1. Loads the trained ML pipeline.
2. Collects patient information through a guided, tabbed intake form.
3. Recreates the engineered features used during training.
4. Predicts 30-day readmission risk.
5. Displays probability on an at-a-glance risk gauge.
6. Provides a recommended decision-support action / clinical note.
7. Generates a local SHAP explanation for the prediction.

IMPORTANT:
This is an academic prototype and must not be used as a clinical diagnostic
or treatment system.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SmartCare — Readmission Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# APPLICATION CONSTANTS
# =============================================================================

APP_TITLE = "SmartCare Hospital"
APP_SUBTITLE = "30-Day Readmission Risk Decision-Support Prototype"

RISK_PALETTE = {
    "LOW": {"color": "#1E8E5A", "bg": "rgba(30,142,90,0.10)", "emoji": "🟢"},
    "MODERATE": {"color": "#C77700", "bg": "rgba(199,119,0,0.10)", "emoji": "🟠"},
    "HIGH": {"color": "#C62828", "bg": "rgba(198,40,40,0.10)", "emoji": "🔴"},
}


# =============================================================================
# CUSTOM CSS — a calmer clinical theme with a little bit of life to it
# =============================================================================

st.markdown(
    """
    <style>
        .main { padding-top: 0.6rem; }

        @keyframes smartcare-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(30,136,229,0.35); }
            70%  { box-shadow: 0 0 0 10px rgba(30,136,229,0); }
            100% { box-shadow: 0 0 0 0 rgba(30,136,229,0); }
        }

        .smartcare-header {
            padding: 1.4rem 1.8rem;
            border-radius: 16px;
            margin-bottom: 1.1rem;
            background: linear-gradient(
                120deg,
                rgba(30, 136, 229, 0.14),
                rgba(0, 150, 136, 0.10) 60%,
                rgba(94, 53, 177, 0.08)
            );
            border: 1px solid rgba(30, 136, 229, 0.22);
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .smartcare-pill {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #1E88E5;
            animation: smartcare-pulse 2.2s infinite;
            flex-shrink: 0;
        }

        .smartcare-tag {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            background: rgba(30,136,229,0.12);
            color: #1565C0;
            margin-right: 0.4rem;
        }

        .risk-banner {
            border-radius: 16px;
            padding: 1.1rem 1.4rem;
            border: 1px solid rgba(128,128,128,0.20);
        }

        .clinical-note {
            border-left: 4px solid #1E88E5;
            padding: 0.7rem 1rem;
            border-radius: 8px;
            background: rgba(30,136,229,0.05);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .footer {
            text-align: center;
            color: #777;
            font-size: 0.82rem;
            padding: 1.1rem 0 0.4rem 0;
        }

        div[data-testid="stMetric"] {
            border-radius: 12px;
            padding: 0.85rem;
            border: 1px solid rgba(128,128,128,0.18);
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_file(filename):
    """
    Find a required file in the project directory.

    Supported locations:
        ./filename
        ./models/filename
    """
    base_dir = Path(__file__).resolve().parent
    possible_paths = [
        base_dir / filename,
        base_dir / "models" / filename,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def options_for(feature_name, fallback_options, category_options):
    """Pull category choices from metadata, falling back to sensible defaults."""
    options = category_options.get(feature_name)
    if not options:
        return fallback_options
    return options


def bmi_category(bmi_value):
    if bmi_value < 18.5:
        return "Underweight"
    elif bmi_value < 25:
        return "Normal"
    elif bmi_value < 30:
        return "Overweight"
    else:
        return "Obese"


def bp_category(systolic, diastolic):
    if systolic >= 140 or diastolic >= 90:
        return "Hypertensive"
    elif systolic >= 130 or diastolic >= 80:
        return "Elevated"
    else:
        return "Normal"


def risk_bucket(proba):
    if proba >= 0.70:
        return "HIGH"
    elif proba >= 0.40:
        return "MODERATE"
    return "LOW"


def draw_risk_gauge(proba, risk_level):
    """A semicircular gauge — a friendlier read than a plain progress bar."""
    palette = RISK_PALETTE[risk_level]

    fig, ax = plt.subplots(figsize=(5.2, 3.0), subplot_kw={"aspect": "equal"})

    # Background arc segments: low / moderate / high zones
    zone_bounds = [(0, 0.40, "#CFE8D8"), (0.40, 0.70, "#F7DCA6"), (0.70, 1.0, "#F2C4C4")]
    for start, end, color in zone_bounds:
        theta1 = 180 - end * 180
        theta2 = 180 - start * 180
        wedge = plt.matplotlib.patches.Wedge(
            (0, 0), 1.0, theta1, theta2, width=0.32, facecolor=color, edgecolor="white"
        )
        ax.add_patch(wedge)

    # Needle
    needle_angle = math.radians(180 - proba * 180)
    needle_x = 0.78 * math.cos(needle_angle)
    needle_y = 0.78 * math.sin(needle_angle)
    ax.plot([0, needle_x], [0, needle_y], color="#333333", linewidth=2.6, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.045, color="#333333", zorder=5))

    ax.text(0, -0.22, f"{proba:.1%}", ha="center", va="center", fontsize=22, fontweight="bold", color=palette["color"])
    ax.text(0, -0.42, "30-day readmission probability", ha="center", va="center", fontsize=9, color="#666666")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.55, 1.05)
    ax.axis("off")
    plt.tight_layout()
    return fig


# =============================================================================
# LOAD MODEL + METADATA
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = find_file("readmission_model.pkl")
    metadata_path = find_file("model_metadata.json")

    if model_path is None:
        raise FileNotFoundError(
            "readmission_model.pkl was not found.\n\n"
            "Please place the model file either in the project root "
            "or inside a 'models' folder."
        )
    if metadata_path is None:
        raise FileNotFoundError(
            "model_metadata.json was not found.\n\n"
            "Please place the metadata file either in the project root "
            "or inside a 'models' folder."
        )

    pipeline = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return pipeline, metadata, model_path, metadata_path


try:
    pipeline, meta, model_path, metadata_path = load_model()
except Exception as error:
    st.error("SmartCare could not load the trained model.")
    st.code(str(error))
    st.info("Make sure your GitHub repository contains 'readmission_model.pkl' and 'model_metadata.json'.")
    st.stop()


# =============================================================================
# READ METADATA
# =============================================================================

category_options = meta.get("category_options", {})
best_model_name = meta.get("best_model_name", "Trained ML Model")
test_roc_auc = meta.get("test_roc_auc", None)
test_f1 = meta.get("test_f1", None)


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f"""
    <div class="smartcare-header">
        <div class="smartcare-pill"></div>
        <div>
            <span class="smartcare-tag">Decision Support</span>
            <span class="smartcare-tag">Academic Prototype</span>
            <h1 style="margin: 0.3rem 0 0.15rem 0;">🏥 {APP_TITLE}</h1>
            <p style="font-size:1.05rem; margin-bottom:0; color:#444;">{APP_SUBTITLE}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.metric("Model", str(best_model_name))
with metric2:
    st.metric("Test ROC-AUC", f"{float(test_roc_auc):.3f}" if test_roc_auc is not None else "N/A")
with metric3:
    st.metric("Test F1 Score", f"{float(test_f1):.3f}" if test_f1 is not None else "N/A")

st.info(
    "**Academic decision-support prototype:** This system was developed for CCS3440 "
    "Artificial Intelligence coursework. It is trained on a synthetic teaching dataset and "
    "is **not clinically validated**. The prediction must never replace professional "
    "clinical judgement.",
)

st.divider()


# =============================================================================
# PATIENT INPUT FORM — organized as tabs instead of one long scroll
# =============================================================================

st.subheader("👤 Patient Intake")

with st.form("patient_form"):

    tab_demo, tab_clinical, tab_history, tab_diagnosis = st.tabs(
        ["👥 Demographics", "🩺 Clinical", "🏨 Admission & History", "💊 Diagnosis & Billing"]
    )

    # -------------------------------------------------------------------
    # DEMOGRAPHICS
    # -------------------------------------------------------------------
    with tab_demo:
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", min_value=0, max_value=100, value=45, step=1)
        with col2:
            gender = st.selectbox("Gender", options_for("gender", ["Female", "Male"], category_options))
        with col3:
            department = st.selectbox(
                "Department",
                options_for("department", ["General", "Cardiology", "Neurology", "Orthopedics"], category_options),
            )
        room_type = st.selectbox(
            "Room Type",
            options_for("room_type", ["General Ward", "Semi-Private", "Private"], category_options),
        )

    # -------------------------------------------------------------------
    # CLINICAL MEASUREMENTS
    # -------------------------------------------------------------------
    with tab_clinical:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 130)
        with c2:
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
        with c3:
            blood_sugar = st.slider("Blood Sugar (mg/dL)", 60, 300, 110)
        with c4:
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 350, 200)
        with c5:
            bmi = st.slider("BMI", 15.0, 45.0, 25.0, step=0.1)

        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.caption(f"BMI category preview: **{bmi_category(bmi)}**")
        with preview_col2:
            st.caption(f"Blood pressure category preview: **{bp_category(systolic_bp, diastolic_bp)}**")

    # -------------------------------------------------------------------
    # ADMISSION & HISTORY
    # -------------------------------------------------------------------
    with tab_history:
        h1, h2, h3, h4, h5 = st.columns(5)
        with h1:
            length_of_stay = st.slider("Length of Stay (days)", 0, 30, 4)
        with h2:
            previous_admissions = st.slider("Previous Admissions", 0, 10, 1)
        with h3:
            previous_appointments = st.slider("Previous Appointments", 0, 15, 3)
        with h4:
            missed_appointments = st.slider("Missed Previous Appointments", 0, 10, 0)
        with h5:
            waiting_days = st.slider("Waiting Days", 0, 60, 14)

    # -------------------------------------------------------------------
    # DIAGNOSIS & TREATMENT
    # -------------------------------------------------------------------
    with tab_diagnosis:
        d1, d2, d3 = st.columns(3)
        with d1:
            diagnosis_group = st.selectbox(
                "Diagnosis Group",
                options_for("diagnosis_group", ["General", "Cardiac", "Respiratory", "Diabetes"], category_options),
            )
            lab_tests_count = st.slider("Lab Tests Count", 0, 15, 3)
        with d2:
            treatments_count = st.slider("Treatments Count", 0, 15, 3)
            payment_status = st.selectbox(
                "Payment Status",
                options_for("payment_status", ["Paid", "Pending", "Insurance"], category_options),
            )
        with d3:
            consultation_fee = st.number_input("Consultation Fee (LKR)", min_value=0, max_value=10000, value=2000, step=100)
            room_charge = st.number_input("Room Charge (LKR)", min_value=0, max_value=200000, value=5000, step=500)
            lab_charge = st.number_input("Lab Charge (LKR)", min_value=0, max_value=50000, value=5000, step=500)
            medicine_charge = st.number_input("Medicine Charge (LKR)", min_value=0, max_value=50000, value=8000, step=500)

    st.divider()
    submitted = st.form_submit_button("🔍 Predict 30-Day Readmission Risk", use_container_width=True, type="primary")


# =============================================================================
# PREDICTION
# =============================================================================

if submitted:

    # -------------------------------------------------------------------
    # DERIVED FEATURES
    # -------------------------------------------------------------------
    total_bill = consultation_fee + room_charge + lab_charge + medicine_charge
    cost_per_day = total_bill / (length_of_stay + 1)
    prior_utilization = previous_admissions + previous_appointments

    patient_row = pd.DataFrame(
        [
            {
                "age": age,
                "waiting_days": waiting_days,
                "previous_appointments": previous_appointments,
                "missed_previous_appointments": missed_appointments,
                "length_of_stay_days": length_of_stay,
                "previous_admissions": previous_admissions,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "blood_sugar_mg_dl": blood_sugar,
                "cholesterol_mg_dl": cholesterol,
                "bmi": bmi,
                "lab_tests_count": lab_tests_count,
                "treatments_count": treatments_count,
                "consultation_fee_lkr": consultation_fee,
                "room_charge_lkr": room_charge,
                "lab_charge_lkr": lab_charge,
                "medicine_charge_lkr": medicine_charge,
                "prior_utilization": prior_utilization,
                "cost_per_day": cost_per_day,
                "gender": gender,
                "department": department,
                "room_type": room_type,
                "payment_status": payment_status,
                "bmi_category": bmi_category(bmi),
                "bp_category": bp_category(systolic_bp, diastolic_bp),
                "diagnosis_group": diagnosis_group,
            }
        ]
    )

    # -------------------------------------------------------------------
    # MODEL PREDICTION
    # -------------------------------------------------------------------
    try:
        with st.spinner("Analysing patient information..."):
            probability_array = pipeline.predict_proba(patient_row)
            proba = float(probability_array[0, 1])
            prediction = int(pipeline.predict(patient_row)[0])
    except Exception as error:
        st.error("❌ Prediction failed.")
        st.code(str(error))
        st.warning(
            "This usually means the input feature names or preprocessing "
            "configuration does not exactly match the trained model."
        )
        st.stop()

    risk_level = risk_bucket(proba)
    palette = RISK_PALETTE[risk_level]

    # =====================================================================
    # PREDICTION RESULT — gauge + headline metrics side by side
    # =====================================================================
    st.divider()
    st.subheader("📊 Prediction Result")

    gauge_col, headline_col = st.columns([1.1, 1])

    with gauge_col:
        fig = draw_risk_gauge(proba, risk_level)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with headline_col:
        st.markdown(
            f"""
            <div class="risk-banner" style="background:{palette['bg']}; border-color:{palette['color']}44;">
                <h3 style="margin-top:0; color:{palette['color']};">
                    {palette['emoji']} {risk_level} READMISSION RISK
                </h3>
                <p style="margin-bottom:0.4rem;">
                    Estimated probability of 30-day readmission:
                    <b>{proba:.1%}</b>
                </p>
                <p style="margin-bottom:0; color:#555; font-size:0.9rem;">
                    Model classification:
                    <b>{"Readmission Risk" if prediction == 1 else "Lower Risk"}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Assessed {datetime.now().strftime('%d %b %Y, %H:%M')}")

    # =====================================================================
    # DECISION-SUPPORT NOTE
    # =====================================================================
    st.subheader("🩺 Decision-Support Recommendation")

    if risk_level == "HIGH":
        note = """
        **Priority follow-up recommended**

        - Consider structured discharge planning.
        - Consider a follow-up contact within 7 days.
        - Review medication adherence instructions.
        - Review follow-up appointment requirements.
        - Consider additional post-discharge monitoring.
        """
    elif risk_level == "MODERATE":
        note = """
        **Standard follow-up with additional attention**

        - Complete standard discharge planning.
        - Ensure follow-up appointment information is provided.
        - Consider appointment reminders.
        - Monitor relevant risk factors at the next visit.
        """
    else:
        note = """
        **Standard discharge pathway**

        - Continue the standard discharge process.
        - Provide routine follow-up instructions.
        - Continue normal patient monitoring.
        """

    st.markdown(f'<div class="clinical-note">{note}</div>', unsafe_allow_html=True)

    # =====================================================================
    # PATIENT SUMMARY
    # =====================================================================
    with st.expander("📋 View Patient Summary"):
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.write(
                {
                    "Age": age,
                    "Gender": gender,
                    "Department": department,
                    "Room Type": room_type,
                    "BMI": bmi,
                    "BMI Category": bmi_category(bmi),
                    "Systolic BP": systolic_bp,
                    "Diastolic BP": diastolic_bp,
                    "BP Category": bp_category(systolic_bp, diastolic_bp),
                    "Diagnosis": diagnosis_group,
                }
            )
        with summary_col2:
            st.write(
                {
                    "Length of Stay": length_of_stay,
                    "Previous Admissions": previous_admissions,
                    "Previous Appointments": previous_appointments,
                    "Missed Appointments": missed_appointments,
                    "Waiting Days": waiting_days,
                    "Lab Tests": lab_tests_count,
                    "Treatments": treatments_count,
                    "Payment Status": payment_status,
                    "Total Bill (LKR)": f"{total_bill:,.2f}",
                    "Cost per Day (LKR)": f"{cost_per_day:,.2f}",
                }
            )

    # =====================================================================
    # SHAP EXPLANATION
    # =====================================================================
    st.divider()
    st.subheader("🔎 Why This Prediction? — Explainable AI")
    st.caption("The chart shows the features that contributed most strongly to this individual prediction.")

    with st.spinner("Generating SHAP explanation..."):
        try:
            if hasattr(pipeline, "named_steps"):
                prep = pipeline.named_steps.get("prep")
                clf = pipeline.named_steps.get("clf")
            else:
                prep = None
                clf = pipeline

            if prep is not None:
                transformed = prep.transform(patient_row)
                if hasattr(transformed, "toarray"):
                    transformed = transformed.toarray()
                feature_names = list(prep.get_feature_names_out())
            else:
                transformed = patient_row
                feature_names = list(patient_row.columns)

            transformed_array = np.asarray(transformed)
            transformed_df = pd.DataFrame(transformed_array, columns=feature_names)

            shap_values = None
            tree_model_names = (
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
                "DecisionTreeClassifier",
            )

            if hasattr(clf, "feature_importances_") or clf.__class__.__name__ in tree_model_names:
                explainer = shap.TreeExplainer(clf)
                shap_result = explainer(transformed_df)
                shap_values = shap_result.values if hasattr(shap_result, "values") else shap_result
            else:
                try:
                    explainer = shap.Explainer(clf, transformed_df)
                    shap_result = explainer(transformed_df)
                    shap_values = shap_result.values if hasattr(shap_result, "values") else shap_result
                except Exception:
                    explainer = shap.LinearExplainer(clf, transformed_df)
                    shap_values = explainer.shap_values(transformed_df)

            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1] if shap_values.shape[2] >= 2 else shap_values[:, :, 0]
            elif shap_values.ndim == 1:
                shap_values = shap_values.reshape(1, -1)
            elif shap_values.ndim != 2:
                raise ValueError(f"Unsupported SHAP output shape: {shap_values.shape}")

            if shap_values.shape[1] != len(feature_names):
                raise ValueError("The number of SHAP values does not match the number of transformed features.")

            contribution = pd.DataFrame({"Feature": feature_names, "SHAP Value": shap_values[0]})
            contribution["Absolute Impact"] = contribution["SHAP Value"].abs()
            contribution = contribution.sort_values("Absolute Impact", ascending=False).head(8)
            contribution["Feature"] = (
                contribution["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
            )

            plot_data = contribution.sort_values("SHAP Value")

            fig, ax = plt.subplots(figsize=(9, 5))
            bar_colors = ["#DD8452" if v > 0 else "#4C72B0" for v in plot_data["SHAP Value"]]
            ax.barh(plot_data["Feature"], plot_data["SHAP Value"], color=bar_colors)
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("SHAP Value")
            ax.set_ylabel("Feature")
            ax.set_title("Top Factors Behind This Prediction")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            st.caption(
                "🟠 Positive SHAP values push the prediction toward higher readmission risk. "
                "🔵 Negative SHAP values push the prediction toward lower readmission risk."
            )

            with st.expander("📈 View SHAP Contribution Values"):
                display_contribution = contribution[["Feature", "SHAP Value"]].copy()
                display_contribution["SHAP Value"] = display_contribution["SHAP Value"].round(4)
                st.dataframe(display_contribution, use_container_width=True, hide_index=True)

        except Exception as error:
            st.warning(
                "The prediction was generated successfully, but the local SHAP explanation "
                "could not be calculated for this model configuration."
            )
            with st.expander("Technical explanation"):
                st.code(str(error))
                st.info(
                    "The prediction itself is still valid if the model prediction completed "
                    "successfully. This issue is limited to the SHAP visualization."
                )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🏥 SmartCare")
    st.markdown(
        """
        ### About this prototype
        SmartCare demonstrates how machine learning can be used as a
        **decision-support system** for estimating the probability of
        30-day hospital readmission.

        ### Workflow
        **1. Patient Data** — demographic, clinical, admission, diagnosis
        and treatment details.

        **2. Feature Engineering** — BMI category, blood pressure category,
        prior healthcare utilization, cost per day.

        **3. Machine Learning** — the trained pipeline generates a
        readmission probability.

        **4. Explainable AI** — SHAP identifies the main factors behind
        the individual prediction.

        **5. Decision Support** — a risk category and suggested follow-up
        considerations.
        """
    )

    st.divider()
    st.markdown("### ⚠️ Important")
    st.caption(
        "This application is an academic prototype based on a synthetic "
        "teaching dataset. It is not a clinically validated medical device."
    )

    st.divider()
    st.markdown("### Model Files")
    st.caption(f"Model: `{model_path.name}`")
    st.caption(f"Metadata: `{metadata_path.name}`")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    """
    <div class="footer">
        <b>SmartCare Hospital AI Coursework Prototype</b><br>
        CCS3440 Artificial Intelligence — Task 08<br>
        Built with Streamlit, Scikit-learn, SHAP and Python<br><br>
        ⚠️ Academic demonstration only — not for clinical diagnosis or treatment.
    </div>
    """,
    unsafe_allow_html=True,
)
