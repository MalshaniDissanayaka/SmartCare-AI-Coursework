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
2. Collects patient information through a friendly, colour-coded form.
3. Recreates the engineered features used during training.
4. Predicts 30-day readmission risk.
5. Displays the probability and risk category with clear visual cues.
6. Provides a recommended decision-support action.
7. Generates a local SHAP explanation for the prediction.

IMPORTANT:
This is an academic prototype and must not be used as a clinical diagnostic
or treatment system.
"""

import json
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

APP_TITLE = "🏥 SmartCare Hospital"
APP_SUBTITLE = "30-Day Readmission Risk Decision-Support Prototype"

# A small, consistent colour palette used throughout the app so that
# risk levels always mean the same thing wherever they appear.
COLOR_LOW = "#1FA97D"        # green  — reassuring
COLOR_MODERATE = "#F2A93B"   # amber  — caution
COLOR_HIGH = "#E6553A"       # red    — urgent
COLOR_PRIMARY = "#3A6FF7"    # blue   — brand / neutral accents
COLOR_SECONDARY = "#7B5CF0"  # violet — secondary accents


# =============================================================================
# CUSTOM CSS — warmer, more colourful visual language
# =============================================================================

st.markdown(
    f"""
    <style>
        .main {{
            padding-top: 1rem;
        }}

        /* Animated gradient banner for the header */
        .smartcare-header {{
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.2rem;
            background: linear-gradient(
                120deg,
                {COLOR_PRIMARY} 0%,
                {COLOR_SECONDARY} 50%,
                #21C7C7 100%
            );
            background-size: 200% 200%;
            animation: gradientShift 10s ease infinite;
            box-shadow: 0 8px 24px rgba(58, 111, 247, 0.25);
        }}

        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        .smartcare-header h1 {{
            color: white;
            margin-bottom: 0.2rem;
            text-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }}

        .smartcare-header p {{
            color: rgba(255,255,255,0.92);
        }}

        /* Section title chips */
        .section-chip {{
            display: inline-block;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            margin: 0.6rem 0 0.8rem 0;
            color: white;
        }}

        .chip-demographics {{ background: {COLOR_PRIMARY}; }}
        .chip-clinical {{ background: {COLOR_SECONDARY}; }}
        .chip-history {{ background: #21A5C7; }}
        .chip-treatment {{ background: #C721A0; }}

        /* Risk result cards */
        .risk-card {{
            padding: 1.4rem 1.6rem;
            border-radius: 16px;
            margin-top: 0.6rem;
            color: white;
            font-weight: 600;
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        }}

        .risk-card.low {{ background: linear-gradient(135deg, {COLOR_LOW}, #16805F); }}
        .risk-card.moderate {{ background: linear-gradient(135deg, {COLOR_MODERATE}, #D98A1E); }}
        .risk-card.high {{ background: linear-gradient(135deg, {COLOR_HIGH}, #B93A24); }}

        .risk-card h2 {{
            margin-top: 0;
            color: white;
        }}

        /* Recommendation card */
        .reco-card {{
            border-radius: 14px;
            padding: 1.1rem 1.4rem;
            border: 2px dashed rgba(128,128,128,0.35);
            background: rgba(123, 92, 240, 0.06);
        }}

        .metric-label {{
            font-size: 0.85rem;
            color: #666;
        }}

        .footer {{
            text-align: center;
            color: #777;
            font-size: 0.85rem;
            padding: 1rem 0;
        }}

        div[data-testid="stMetric"] {{
            border-radius: 12px;
            padding: 0.9rem;
            border: 1px solid rgba(128,128,128,0.20);
            background: rgba(58, 111, 247, 0.04);
        }}

        /* Make the primary submit button pop */
        button[kind="primary"] {{
            background: linear-gradient(90deg, {COLOR_PRIMARY}, {COLOR_SECONDARY}) !important;
            border: none !important;
        }}
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


def get_metadata_value(metadata, key, default=None):
    """Safely retrieve metadata values."""
    return metadata.get(key, default)


def section_chip(label, css_class):
    """Render a small colourful pill-shaped label above a form section."""
    st.markdown(
        f'<span class="section-chip {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


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


# =============================================================================
# LOAD MODEL
# =============================================================================

try:

    pipeline, meta, model_path, metadata_path = load_model()

except Exception as error:

    st.error("🚫 SmartCare could not load the trained model.")

    st.code(str(error))

    st.info(
        "Make sure your GitHub repository contains "
        "'readmission_model.pkl' and 'model_metadata.json'."
    )

    st.stop()


# =============================================================================
# READ METADATA
# =============================================================================

numeric_features = meta.get("numeric_features", [])
categorical_features = meta.get("categorical_features", [])
category_options = meta.get("category_options", {})
numeric_ranges = meta.get("numeric_ranges", {})

best_model_name = meta.get("best_model_name", "Trained ML Model")
test_roc_auc = meta.get("test_roc_auc", None)
test_f1 = meta.get("test_f1", None)


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f"""
    <div class="smartcare-header">
        <h1>{APP_TITLE}</h1>
        <p style="font-size:1.1rem; margin-bottom:0;">
            {APP_SUBTITLE}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# MODEL INFORMATION
# =============================================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("🤖 Model", str(best_model_name))

with metric2:
    if test_roc_auc is not None:
        st.metric("📈 Test ROC-AUC", f"{float(test_roc_auc):.3f}")
    else:
        st.metric("📈 Test ROC-AUC", "N/A")

with metric3:
    if test_f1 is not None:
        st.metric("🎯 Test F1 Score", f"{float(test_f1):.3f}")
    else:
        st.metric("🎯 Test F1 Score", "N/A")


st.info(
    "ℹ️ **Academic decision-support prototype:** "
    "This system was developed for CCS3440 Artificial Intelligence coursework. "
    "It is trained on a synthetic teaching dataset and is **not clinically validated**. "
    "The prediction must never replace professional clinical judgement.",
)

st.divider()


# =============================================================================
# ENGINEERED FEATURES
# =============================================================================

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


# =============================================================================
# CATEGORY OPTION HELPERS
# =============================================================================

def options_for(feature_name, fallback_options):

    options = category_options.get(feature_name)

    if options is None or len(options) == 0:
        return fallback_options

    return options


# =============================================================================
# PATIENT INPUT FORM
# =============================================================================

st.subheader("👤 Patient Information")
st.caption("Fill in the sections below, then click the button at the bottom to run the prediction.")

with st.form("patient_form"):

    # -------------------------------------------------------------------------
    # DEMOGRAPHICS
    # -------------------------------------------------------------------------

    section_chip("👥 Demographics", "chip-demographics")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider(
            "Age",
            min_value=0,
            max_value=100,
            value=45,
            step=1,
            help="Patient's age in years.",
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            options_for("gender", ["Female", "Male"]),
        )

    with col3:

        department = st.selectbox(
            "Department",
            options_for(
                "department",
                ["General", "Cardiology", "Neurology", "Orthopedics"],
            ),
        )

    room_type = st.selectbox(
        "Room Type",
        options_for(
            "room_type",
            ["General Ward", "Semi-Private", "Private"],
        ),
    )

    st.divider()

    # -------------------------------------------------------------------------
    # CLINICAL MEASUREMENTS
    # -------------------------------------------------------------------------

    section_chip("🩺 Clinical Measurements", "chip-clinical")

    clinical_col1, clinical_col2, clinical_col3, clinical_col4, clinical_col5 = (
        st.columns(5)
    )

    with clinical_col1:

        systolic_bp = st.slider(
            "Systolic BP (mmHg)",
            90,
            200,
            130,
            help="Top number of a blood pressure reading.",
        )

    with clinical_col2:

        diastolic_bp = st.slider(
            "Diastolic BP (mmHg)",
            50,
            130,
            80,
            help="Bottom number of a blood pressure reading.",
        )

    with clinical_col3:

        blood_sugar = st.slider(
            "Blood Sugar (mg/dL)",
            60,
            300,
            110,
        )

    with clinical_col4:

        cholesterol = st.slider(
            "Cholesterol (mg/dL)",
            100,
            350,
            200,
        )

    with clinical_col5:

        bmi = st.slider(
            "BMI",
            15.0,
            45.0,
            25.0,
            step=0.1,
            help="Body Mass Index — weight (kg) / height² (m²).",
        )

    # Live preview chips so the categories are understandable before submitting
    preview_bmi_cat = bmi_category(bmi)
    preview_bp_cat = bp_category(systolic_bp, diastolic_bp)
    st.caption(f"🔎 Live preview → BMI category: **{preview_bmi_cat}** · Blood pressure category: **{preview_bp_cat}**")

    st.divider()

    # -------------------------------------------------------------------------
    # ADMISSION & HISTORY
    # -------------------------------------------------------------------------

    section_chip("🏨 Admission & Patient History", "chip-history")

    history_col1, history_col2, history_col3, history_col4, history_col5 = (
        st.columns(5)
    )

    with history_col1:

        length_of_stay = st.slider(
            "Length of Stay (days)",
            0,
            30,
            4,
        )

    with history_col2:

        previous_admissions = st.slider(
            "Previous Admissions",
            0,
            10,
            1,
        )

    with history_col3:

        previous_appointments = st.slider(
            "Previous Appointments",
            0,
            15,
            3,
        )

    with history_col4:

        missed_appointments = st.slider(
            "Missed Previous Appointments",
            0,
            10,
            0,
        )

    with history_col5:

        waiting_days = st.slider(
            "Waiting Days",
            0,
            60,
            14,
        )

    st.divider()

    # -------------------------------------------------------------------------
    # DIAGNOSIS & TREATMENT
    # -------------------------------------------------------------------------

    section_chip("💊 Diagnosis & Treatment", "chip-treatment")

    diagnosis_col1, diagnosis_col2, diagnosis_col3 = st.columns(3)

    with diagnosis_col1:

        diagnosis_group = st.selectbox(
            "Diagnosis Group",
            options_for(
                "diagnosis_group",
                ["General", "Cardiac", "Respiratory", "Diabetes"],
            ),
        )

        lab_tests_count = st.slider(
            "Lab Tests Count",
            0,
            15,
            3,
        )

    with diagnosis_col2:

        treatments_count = st.slider(
            "Treatments Count",
            0,
            15,
            3,
        )

        payment_status = st.selectbox(
            "Payment Status",
            options_for(
                "payment_status",
                ["Paid", "Pending", "Insurance"],
            ),
        )

    with diagnosis_col3:

        consultation_fee = st.number_input(
            "Consultation Fee (LKR)",
            min_value=0,
            max_value=10000,
            value=2000,
            step=100,
        )

        room_charge = st.number_input(
            "Room Charge (LKR)",
            min_value=0,
            max_value=200000,
            value=5000,
            step=500,
        )

        lab_charge = st.number_input(
            "Lab Charge (LKR)",
            min_value=0,
            max_value=50000,
            value=5000,
            step=500,
        )

        medicine_charge = st.number_input(
            "Medicine Charge (LKR)",
            min_value=0,
            max_value=50000,
            value=8000,
            step=500,
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict 30-Day Readmission Risk",
        use_container_width=True,
        type="primary",
    )


# =============================================================================
# PREDICTION
# =============================================================================

if submitted:

    # -------------------------------------------------------------------------
    # DERIVED FEATURES
    # -------------------------------------------------------------------------

    total_bill = (
        consultation_fee
        + room_charge
        + lab_charge
        + medicine_charge
    )

    cost_per_day = total_bill / (length_of_stay + 1)

    prior_utilization = (
        previous_admissions
        + previous_appointments
    )

    # -------------------------------------------------------------------------
    # CREATE PATIENT DATAFRAME
    # -------------------------------------------------------------------------

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
                "bp_category": bp_category(
                    systolic_bp,
                    diastolic_bp,
                ),
                "diagnosis_group": diagnosis_group,
            }
        ]
    )

    # -------------------------------------------------------------------------
    # MODEL PREDICTION
    # -------------------------------------------------------------------------

    try:

        with st.spinner("🧠 Analysing patient information..."):

            probability_array = pipeline.predict_proba(
                patient_row
            )

            proba = float(probability_array[0, 1])

            prediction = int(
                pipeline.predict(patient_row)[0]
            )

    except Exception as error:

        st.error("❌ Prediction failed.")

        st.code(str(error))

        st.warning(
            "This usually means the input feature names or preprocessing "
            "configuration does not exactly match the trained model."
        )

        st.stop()

    # =========================================================================
    # PREDICTION RESULT
    # =========================================================================

    st.divider()

    st.subheader("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric("Readmission Probability", f"{proba:.1%}")

    with result_col2:

        if proba >= 0.70:
            risk_level = "HIGH"
            risk_css = "high"
            risk_emoji = "🔴"
        elif proba >= 0.40:
            risk_level = "MODERATE"
            risk_css = "moderate"
            risk_emoji = "🟠"
        else:
            risk_level = "LOW"
            risk_css = "low"
            risk_emoji = "🟢"

        st.metric("Risk Level", f"{risk_emoji} {risk_level}")

    with result_col3:

        st.metric(
            "Model Classification",
            "Readmission Risk" if prediction == 1 else "Lower Risk",
        )

    st.progress(min(max(proba, 0.0), 1.0))

    # =========================================================================
    # RISK INTERPRETATION — colour-coded card
    # =========================================================================

    if risk_css == "high":

        st.markdown(
            f"""
            <div class="risk-card high">
                <h2>⚠️ HIGH READMISSION RISK</h2>
                <p style="font-size:1.05rem; margin-bottom:0;">
                    The model estimates a <b>{proba:.1%} probability</b> of
                    30-day readmission for this patient.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif risk_css == "moderate":

        st.markdown(
            f"""
            <div class="risk-card moderate">
                <h2>🟠 MODERATE READMISSION RISK</h2>
                <p style="font-size:1.05rem; margin-bottom:0;">
                    The model estimates a <b>{proba:.1%} probability</b> of
                    30-day readmission for this patient.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="risk-card low">
                <h2>✅ LOWER READMISSION RISK</h2>
                <p style="font-size:1.05rem; margin-bottom:0;">
                    The model estimates a <b>{proba:.1%} probability</b> of
                    30-day readmission for this patient.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # =========================================================================
    # RECOMMENDED DECISION SUPPORT
    # =========================================================================

    st.subheader("🩺 Decision-Support Recommendation")

    if risk_css == "high":

        st.markdown(
            """
            <div class="reco-card">
            <b>🚨 Priority follow-up recommended</b>
            <ul>
                <li>Consider structured discharge planning.</li>
                <li>Consider a follow-up contact within 7 days.</li>
                <li>Review medication adherence instructions.</li>
                <li>Review follow-up appointment requirements.</li>
                <li>Consider additional post-discharge monitoring.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif risk_css == "moderate":

        st.markdown(
            """
            <div class="reco-card">
            <b>🟠 Standard follow-up with additional attention</b>
            <ul>
                <li>Complete standard discharge planning.</li>
                <li>Ensure follow-up appointment information is provided.</li>
                <li>Consider appointment reminders.</li>
                <li>Monitor relevant risk factors at the next visit.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="reco-card">
            <b>✅ Standard discharge pathway</b>
            <ul>
                <li>Continue the standard discharge process.</li>
                <li>Provide routine follow-up instructions.</li>
                <li>Continue normal patient monitoring.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # PATIENT SUMMARY
    # =========================================================================

    with st.expander("📋 View Patient Summary"):

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.markdown("**👤 Demographics & Clinical**")
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
                    "BP Category": bp_category(
                        systolic_bp,
                        diastolic_bp,
                    ),
                    "Diagnosis": diagnosis_group,
                }
            )

        with summary_col2:

            st.markdown("**🏨 History & Billing**")
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

    # =========================================================================
    # SHAP EXPLANATION
    # =========================================================================

    st.divider()

    st.subheader("🔎 Why This Prediction? — Explainable AI")

    st.caption(
        "The chart shows the features that contributed most strongly "
        "to this individual prediction. 🟠 pushes risk up, 🔵 pushes risk down."
    )

    with st.spinner("🧮 Generating SHAP explanation..."):

        try:

            # -----------------------------------------------------------------
            # Extract preprocessing and classifier
            # -----------------------------------------------------------------

            if hasattr(pipeline, "named_steps"):

                prep = pipeline.named_steps.get("prep")
                clf = pipeline.named_steps.get("clf")

            else:

                prep = None
                clf = pipeline

            # -----------------------------------------------------------------
            # Transform patient data
            # -----------------------------------------------------------------

            if prep is not None:

                transformed = prep.transform(
                    patient_row
                )

                if hasattr(
                    transformed,
                    "toarray",
                ):
                    transformed = transformed.toarray()

                feature_names = list(
                    prep.get_feature_names_out()
                )

            else:

                transformed = patient_row

                feature_names = list(
                    patient_row.columns
                )

            transformed_array = np.asarray(
                transformed
            )

            transformed_df = pd.DataFrame(
                transformed_array,
                columns=feature_names,
            )

            # -----------------------------------------------------------------
            # SHAP calculation
            # -----------------------------------------------------------------

            shap_values = None

            # Tree models
            tree_model_names = (
                "RandomForestClassifier",
                "ExtraTreesClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
                "DecisionTreeClassifier",
            )

            if (
                hasattr(clf, "feature_importances_")
                or clf.__class__.__name__
                in tree_model_names
            ):

                explainer = shap.TreeExplainer(
                    clf
                )

                shap_result = explainer(
                    transformed_df
                )

                if hasattr(
                    shap_result,
                    "values",
                ):

                    shap_values = shap_result.values

                else:

                    shap_values = shap_result

            else:

                # Linear / generic fallback
                try:

                    explainer = shap.Explainer(
                        clf,
                        transformed_df,
                    )

                    shap_result = explainer(
                        transformed_df
                    )

                    if hasattr(
                        shap_result,
                        "values",
                    ):

                        shap_values = shap_result.values

                    else:

                        shap_values = shap_result

                except Exception:

                    explainer = shap.LinearExplainer(
                        clf,
                        transformed_df,
                    )

                    shap_values = explainer.shap_values(
                        transformed_df
                    )

            # -----------------------------------------------------------------
            # Normalize SHAP output
            # -----------------------------------------------------------------

            shap_values = np.asarray(
                shap_values
            )

            if shap_values.ndim == 3:

                # Binary classification:
                # [samples, features, classes]

                if shap_values.shape[2] >= 2:

                    shap_values = shap_values[
                        :,
                        :,
                        1,
                    ]

                else:

                    shap_values = shap_values[
                        :,
                        :,
                        0,
                    ]

            elif shap_values.ndim == 2:

                pass

            elif shap_values.ndim == 1:

                shap_values = shap_values.reshape(
                    1,
                    -1,
                )

            else:

                raise ValueError(
                    f"Unsupported SHAP output shape: "
                    f"{shap_values.shape}"
                )

            # -----------------------------------------------------------------
            # Validate dimensions
            # -----------------------------------------------------------------

            if shap_values.shape[1] != len(
                feature_names
            ):

                raise ValueError(
                    "The number of SHAP values does not "
                    "match the number of transformed features."
                )

            # -----------------------------------------------------------------
            # Build contribution table
            # -----------------------------------------------------------------

            contribution = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": shap_values[0],
                }
            )

            contribution["Absolute Impact"] = (
                contribution["SHAP Value"]
                .abs()
            )

            contribution = (
                contribution
                .sort_values(
                    "Absolute Impact",
                    ascending=False,
                )
                .head(8)
            )

            # Remove preprocessing prefixes for readability
            contribution["Feature"] = (
                contribution["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )

            # -----------------------------------------------------------------
            # Plot — colourful horizontal bar chart
            # -----------------------------------------------------------------

            plot_data = contribution.sort_values("SHAP Value")

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")

            bar_colors = [
                COLOR_HIGH if value > 0 else COLOR_PRIMARY
                for value in plot_data["SHAP Value"]
            ]

            bars = ax.barh(
                plot_data["Feature"],
                plot_data["SHAP Value"],
                color=bar_colors,
                edgecolor="white",
                linewidth=0.6,
            )

            ax.bar_label(
                bars,
                fmt="%.3f",
                padding=4,
                fontsize=8,
                color="#444",
            )

            ax.axvline(0, linewidth=1.2, color="#555")

            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_ylabel("")
            ax.set_title(
                "Top Factors Behind This Prediction",
                fontsize=13,
                fontweight="bold",
                color="#333",
            )

            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            plt.tight_layout()

            st.pyplot(fig, clear_figure=True)

            plt.close(fig)

            st.caption(
                "🟠 Positive SHAP values push the prediction toward "
                "higher readmission risk. 🔵 Negative SHAP values push "
                "the prediction toward lower readmission risk."
            )

            # -----------------------------------------------------------------
            # Contribution table
            # -----------------------------------------------------------------

            with st.expander("📈 View SHAP Contribution Values"):

                display_contribution = (
                    contribution[
                        [
                            "Feature",
                            "SHAP Value",
                        ]
                    ].copy()
                )

                display_contribution["SHAP Value"] = display_contribution[
                    "SHAP Value"
                ].round(4)

                st.dataframe(
                    display_contribution,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:

            st.warning(
                "⚠️ The prediction was generated successfully, "
                "but the local SHAP explanation could not be calculated "
                "for this model configuration."
            )

            with st.expander("Technical explanation"):

                st.code(str(error))

                st.info(
                    "The prediction itself is still valid if the model "
                    "prediction completed successfully. This issue is "
                    "limited to the SHAP visualization."
                )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {COLOR_PRIMARY}, {COLOR_SECONDARY});
            padding: 1rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1rem;
        ">
            <h2 style="color:white; margin:0;">🏥 SmartCare</h2>
            <p style="margin:0; opacity:0.9;">Readmission Risk Assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### 📖 About this prototype

        SmartCare demonstrates how machine learning can be used as a
        **decision-support system** for estimating the probability of
        30-day hospital readmission.

        ### 🔄 Workflow

        **1. 👤 Patient Data**
        Enter demographic, clinical, admission, diagnosis and treatment data.

        **2. ⚙️ Feature Engineering**
        The application calculates derived variables such as:
        - BMI category
        - Blood pressure category
        - Prior healthcare utilization
        - Cost per day

        **3. 🤖 Machine Learning**
        The trained pipeline generates a readmission probability.

        **4. 🔎 Explainable AI**
        SHAP identifies the main factors contributing to the individual
        prediction.

        **5. 🩺 Decision Support**
        The application provides a risk category and suggested follow-up
        considerations.
        """
    )

    st.divider()

    st.markdown("### ⚠️ Important")

    st.warning(
        "This application is an academic prototype based on a synthetic "
        "teaching dataset. It is **not** a clinically validated medical device.",
        icon="⚠️",
    )

    st.divider()

    st.markdown("### 🗂️ Model Files")

    st.caption(f"📦 Model: `{model_path.name}`")
    st.caption(f"🗒️ Metadata: `{metadata_path.name}`")


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.markdown(
    f"""
    <div class="footer">
        <b>SmartCare Hospital AI Coursework Prototype</b><br>
        CCS3440 Artificial Intelligence — Task 08<br>
        Built with 🐍 Python, ⚡ Streamlit, 🌲 Scikit-learn, and 🔎 SHAP<br><br>
        ⚠️ Academic demonstration only — not for clinical diagnosis or treatment.
    </div>
    """,
    unsafe_allow_html=True,
)
