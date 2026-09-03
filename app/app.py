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
2. Collects patient information.
3. Recreates the engineered features used during training.
4. Predicts 30-day readmission risk.
5. Displays probability and risk category.
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


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .smartcare-header {
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            background: linear-gradient(
                135deg,
                rgba(30, 136, 229, 0.10),
                rgba(0, 150, 136, 0.08)
            );
            border: 1px solid rgba(30, 136, 229, 0.20);
        }

        .risk-card {
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-top: 0.5rem;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #666;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .footer {
            text-align: center;
            color: #777;
            font-size: 0.85rem;
            padding: 1rem 0;
        }

        div[data-testid="stMetric"] {
            border-radius: 10px;
            padding: 0.8rem;
            border: 1px solid rgba(128,128,128,0.20);
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


def get_metadata_value(metadata, key, default=None):
    """Safely retrieve metadata values."""
    return metadata.get(key, default)


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

    st.error(" SmartCare could not load the trained model.")

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
    st.metric(
        "Model",
        str(best_model_name),
    )

with metric2:
    if test_roc_auc is not None:
        st.metric(
            "Test ROC-AUC",
            f"{float(test_roc_auc):.3f}",
        )
    else:
        st.metric(
            "Test ROC-AUC",
            "N/A",
        )

with metric3:
    if test_f1 is not None:
        st.metric(
            "Test F1 Score",
            f"{float(test_f1):.3f}",
        )
    else:
        st.metric(
            "Test F1 Score",
            "N/A",
        )


st.info(
    " **Academic decision-support prototype:** "
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

with st.form("patient_form"):

    # -------------------------------------------------------------------------
    # DEMOGRAPHICS
    # -------------------------------------------------------------------------

    st.markdown("### 👥 Demographics")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.slider(
            "Age",
            min_value=0,
            max_value=100,
            value=45,
            step=1,
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            options_for(
                "gender",
                ["Female", "Male"],
            ),
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

    st.markdown("### 🩺 Clinical Measurements")

    clinical_col1, clinical_col2, clinical_col3, clinical_col4, clinical_col5 = (
        st.columns(5)
    )

    with clinical_col1:

        systolic_bp = st.slider(
            "Systolic BP (mmHg)",
            90,
            200,
            130,
        )

    with clinical_col2:

        diastolic_bp = st.slider(
            "Diastolic BP (mmHg)",
            50,
            130,
            80,
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
        )

    st.divider()

    # -------------------------------------------------------------------------
    # ADMISSION & HISTORY
    # -------------------------------------------------------------------------

    st.markdown("### 🏨 Admission & Patient History")

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

    st.markdown("### 💊 Diagnosis & Treatment")

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

        with st.spinner("Analysing patient information..."):

            probability_array = pipeline.predict_proba(
                patient_row
            )

            proba = float(probability_array[0, 1])

            prediction = int(
                pipeline.predict(patient_row)[0]
            )

    except Exception as error:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(error)
        )

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

        st.metric(
            "Readmission Probability",
            f"{proba:.1%}",
        )

    with result_col2:

        if proba >= 0.70:
            risk_level = "HIGH"
        elif proba >= 0.40:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        st.metric(
            "Risk Level",
            risk_level,
        )

    with result_col3:

        st.metric(
            "Model Classification",
            "Readmission Risk" if prediction == 1 else "Lower Risk",
        )

    st.progress(
        min(
            max(proba, 0.0),
            1.0,
        )
    )

    # =========================================================================
    # RISK INTERPRETATION
    # =========================================================================

    if proba >= 0.70:

        st.error(
            f"""
            ### ⚠️ HIGH READMISSION RISK

            The model estimates a **{proba:.1%} probability** of
            30-day readmission.
            """
        )

    elif proba >= 0.40:

        st.warning(
            f"""
            ### 🟠 MODERATE READMISSION RISK

            The model estimates a **{proba:.1%} probability** of
            30-day readmission.
            """
        )

    else:

        st.success(
            f"""
            ### ✅ LOWER READMISSION RISK

            The model estimates a **{proba:.1%} probability** of
            30-day readmission.
            """
        )

    # =========================================================================
    # RECOMMENDED DECISION SUPPORT
    # =========================================================================

    st.subheader("🩺 Decision-Support Recommendation")

    if proba >= 0.70:

        st.markdown(
            """
            **Priority follow-up recommended**

            - Consider structured discharge planning.
            - Consider a follow-up contact within 7 days.
            - Review medication adherence instructions.
            - Review follow-up appointment requirements.
            - Consider additional post-discharge monitoring.
            """
        )

    elif proba >= 0.40:

        st.markdown(
            """
            **Standard follow-up with additional attention**

            - Complete standard discharge planning.
            - Ensure follow-up appointment information is provided.
            - Consider appointment reminders.
            - Monitor relevant risk factors at the next visit.
            """
        )

    else:

        st.markdown(
            """
            **Standard discharge pathway**

            - Continue the standard discharge process.
            - Provide routine follow-up instructions.
            - Continue normal patient monitoring.
            """
        )

    # =========================================================================
    # PATIENT SUMMARY
    # =========================================================================

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
                    "BP Category": bp_category(
                        systolic_bp,
                        diastolic_bp,
                    ),
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

    # =========================================================================
    # SHAP EXPLANATION
    # =========================================================================

    st.divider()

    st.subheader("🔎 Why This Prediction? — Explainable AI")

    st.caption(
        "The chart shows the features that contributed most strongly "
        "to this individual prediction."
    )

    with st.spinner("Generating SHAP explanation..."):

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
                .str.replace(
                    "num__",
                    "",
                    regex=False,
                )
                .str.replace(
                    "cat__",
                    "",
                    regex=False,
                )
            )

            # -----------------------------------------------------------------
            # Plot
            # -----------------------------------------------------------------

            plot_data = contribution.sort_values(
                "SHAP Value"
            )

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            bar_colors = [
                "#DD8452"
                if value > 0
                else "#4C72B0"
                for value in plot_data["SHAP Value"]
            ]

            ax.barh(
                plot_data["Feature"],
                plot_data["SHAP Value"],
                color=bar_colors,
            )

            ax.axvline(
                0,
                linewidth=1,
            )

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top Factors Behind This Prediction"
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            plt.close(fig)

            st.caption(
                "🟠 Positive SHAP values push the prediction toward "
                "higher readmission risk. 🔵 Negative SHAP values push "
                "the prediction toward lower readmission risk."
            )

            # -----------------------------------------------------------------
            # Contribution table
            # -----------------------------------------------------------------

            with st.expander(
                "📈 View SHAP Contribution Values"
            ):

                display_contribution = (
                    contribution[
                        [
                            "Feature",
                            "SHAP Value",
                        ]
                    ].copy()
                )

                display_contribution[
                    "SHAP Value"
                ] = display_contribution[
                    "SHAP Value"
                ].round(4)

                st.dataframe(
                    display_contribution,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:

            st.warning(
                " The prediction was generated successfully, "
                "but the local SHAP explanation could not be calculated "
                "for this model configuration."
            )

            with st.expander(
                "Technical explanation"
            ):

                st.code(
                    str(error)
                )

                st.info(
                    "The prediction itself is still valid if the model "
                    "prediction completed successfully. This issue is "
                    "limited to the SHAP visualization."
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

        **1. Patient Data**

        Enter demographic, clinical, admission, diagnosis and treatment data.

        **2. Feature Engineering**

        The application calculates derived variables such as:

        - BMI category
        - Blood pressure category
        - Prior healthcare utilization
        - Cost per day

        **3. Machine Learning**

        The trained pipeline generates a readmission probability.

        **4. Explainable AI**

        SHAP identifies the main factors contributing to the individual
        prediction.

        **5. Decision Support**

        The application provides a risk category and suggested follow-up
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

    st.caption(
        f"Model: `{model_path.name}`"
    )

    st.caption(
        f"Metadata: `{metadata_path.name}`"
    )


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
