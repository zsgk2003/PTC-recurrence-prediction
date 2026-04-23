"""
Thyroid Cancer Recurrence Prediction - Streamlit Deployment App
Uses the best-performing model (LightGBM) for online prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Thyroid Cancer Recurrence Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Global constants
# ============================================================
MODEL_DIR = "./thyroid_cancer_9models_results_pureNoStacking_335PTCpatients_lightGBM"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "BEST_MODEL_LightGBM.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_NAMES = [
    'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
    'Thyroid Function', 'Physical Examination', 'Adenopathy',
    'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response'
]

FEATURE_MAPS = {
    'Gender': {'Female (F)': 0, 'Male (M)': 1},
    'Smoking': {'No': 0, 'Yes': 1},
    'Hx Smoking': {'No': 0, 'Yes': 1},
    'Hx Radiothreapy': {'No': 0, 'Yes': 1},
    'Thyroid Function': {
        'Clinical Hyperthyroidism': 0,
        'Clinical Hypothyroidism': 1,
        'Euthyroid': 2,
        'Subclinical Hyperthyroidism': 3,
        'Subclinical Hypothyroidism': 4,
    },
    'Physical Examination': {
        'Diffuse goiter': 0,
        'Multinodular goiter': 1,
        'Normal': 2,
        'Single nodular goiter-left': 3,
        'Single nodular goiter-right': 4,
    },
    'Adenopathy': {
        'Bilateral': 0,
        'Extensive': 1,
        'Left': 2,
        'No': 3,
        'Posterior': 4,
        'Right': 5,
    },
    'Pathology': {
        'Micropapillary': 0,
        'Papillary': 1,
    },
    'Focality': {'Multi-Focal': 0, 'Uni-Focal': 1},
    'Risk': {'Low': 0, 'Intermediate': 1, 'High': 2},
    'T': {'T1a': 0, 'T1b': 1, 'T2': 2, 'T3a': 3, 'T3b': 4, 'T4a': 5, 'T4b': 6},
    'N': {'N0': 0, 'N1a': 1, 'N1b': 2},
    'M': {'M0': 0, 'M1': 1},
    'Stage': {'I': 0, 'II': 1, 'III': 2, 'IVA': 3, 'IVB': 4},
    'Response': {
        'Excellent': 0,
        'Indeterminate': 1,
        'Structural Incomplete': 2,
        'Biochemical Incomplete': 3,
    },
}


# ============================================================
# Cached loaders
# ============================================================
@st.cache_resource
def load_model_and_scaler():
    """Load the best model and scaler."""
    if not os.path.exists(BEST_MODEL_PATH):
        st.error(f"Model file not found: {BEST_MODEL_PATH}")
        st.stop()
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler


@st.cache_data
def load_model_ranking():
    """Load model ranking table."""
    ranking_path = os.path.join(MODEL_DIR, "model_ranking.csv")
    if os.path.exists(ranking_path):
        return pd.read_csv(ranking_path, index_col=0)
    return None


@st.cache_data
def load_dataset_info():
    """Load dataset metadata."""
    info_path = os.path.join(MODEL_DIR, "dataset_info.txt")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
    return info


# ============================================================
# Prediction helpers
# ============================================================
def predict_recurrence(model, input_df, scaler=None):
    """Run prediction on a single record."""
    X = input_df[FEATURE_NAMES].values
    # Apply scaling if scaler is provided
    if scaler is not None:
        X = scaler.transform(X)
    y_pred = model.predict(X)[0]
    y_prob = model.predict_proba(X)[0, 1]
    return int(y_pred), float(y_prob)


def risk_level(prob):
    """Return (label, color) for a probability."""
    if prob < 0.2:
        return "Low Risk", "#2ecc71"
    elif prob < 0.5:
        return "Low-Moderate Risk", "#f1c40f"
    elif prob < 0.75:
        return "Moderate-High Risk", "#e67e22"
    else:
        return "High Risk", "#e74c3c"


# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Single Prediction", "Batch Prediction", "Model Performance", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.success(
    "**Current Model**: LightGBM (Best)\n\n"
    "**AUC**: 0.9830\n\n"
    "**F1-score**: 0.9412\n\n"
    "**Accuracy**: 97.03%"
)

# ============================================================
# Load resources
# ============================================================
model, scaler = load_model_and_scaler()
ranking_df = load_model_ranking()
dataset_info = load_dataset_info()

# ============================================================
# Main header
# ============================================================
st.title("Papillary Thyroid Carcinoma Recurrence Predictor")
st.markdown("""
<div style='background-color:#e8f4f8;padding:15px;border-radius:10px;border-left:5px solid #3498db;'>
<b>About this system:</b> Built on clinical data from 335 Papillary Thyroid Carcinoma (PTC)
patients, this system uses machine learning (LightGBM) to predict recurrence risk.
It supports <b>single-case</b> and <b>batch</b> predictions, and provides model
performance and interpretability insights.
</div>
""", unsafe_allow_html=True)
st.markdown("")


# ============================================================
# Page 1: Single Prediction
# ============================================================
if page == "Single Prediction":
    st.header("Single Patient Recurrence Prediction")
    st.markdown("Fill in the **16 clinical features** below to predict recurrence probability.")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age = st.number_input("Age", min_value=15, max_value=100, value=45, step=1)
            gender = st.selectbox("Gender", list(FEATURE_MAPS['Gender'].keys()))
            smoking = st.selectbox("Smoking", list(FEATURE_MAPS['Smoking'].keys()))
            hx_smoking = st.selectbox("History of Smoking (Hx Smoking)", list(FEATURE_MAPS['Hx Smoking'].keys()))
            hx_radiotherapy = st.selectbox("History of Radiotherapy (Hx Radiothreapy)", list(FEATURE_MAPS['Hx Radiothreapy'].keys()))

        with col2:
            st.subheader("Clinical Findings")
            thyroid_fn = st.selectbox("Thyroid Function", list(FEATURE_MAPS['Thyroid Function'].keys()), index=2)
            physical = st.selectbox("Physical Examination", list(FEATURE_MAPS['Physical Examination'].keys()), index=2)
            adenopathy = st.selectbox("Adenopathy", list(FEATURE_MAPS['Adenopathy'].keys()), index=3)
            pathology = st.selectbox("Pathology", list(FEATURE_MAPS['Pathology'].keys()), index=1)
            focality = st.selectbox("Focality", list(FEATURE_MAPS['Focality'].keys()), index=1)

        with col3:
            st.subheader("Staging & Response")
            risk = st.selectbox("Risk Stratification", list(FEATURE_MAPS['Risk'].keys()))
            t_stage = st.selectbox("T stage (Tumor)", list(FEATURE_MAPS['T'].keys()))
            n_stage = st.selectbox("N stage (Node)", list(FEATURE_MAPS['N'].keys()))
            m_stage = st.selectbox("M stage (Metastasis)", list(FEATURE_MAPS['M'].keys()))
            stage = st.selectbox("Overall Stage", list(FEATURE_MAPS['Stage'].keys()))
            response = st.selectbox("Treatment Response", list(FEATURE_MAPS['Response'].keys()))

        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        input_data = {
            'Age': age,
            'Gender': FEATURE_MAPS['Gender'][gender],
            'Smoking': FEATURE_MAPS['Smoking'][smoking],
            'Hx Smoking': FEATURE_MAPS['Hx Smoking'][hx_smoking],
            'Hx Radiothreapy': FEATURE_MAPS['Hx Radiothreapy'][hx_radiotherapy],
            'Thyroid Function': FEATURE_MAPS['Thyroid Function'][thyroid_fn],
            'Physical Examination': FEATURE_MAPS['Physical Examination'][physical],
            'Adenopathy': FEATURE_MAPS['Adenopathy'][adenopathy],
            'Pathology': FEATURE_MAPS['Pathology'][pathology],
            'Focality': FEATURE_MAPS['Focality'][focality],
            'Risk': FEATURE_MAPS['Risk'][risk],
            'T': FEATURE_MAPS['T'][t_stage],
            'N': FEATURE_MAPS['N'][n_stage],
            'M': FEATURE_MAPS['M'][m_stage],
            'Stage': FEATURE_MAPS['Stage'][stage],
            'Response': FEATURE_MAPS['Response'][response],
        }
        input_df = pd.DataFrame([input_data])

        pred, prob = predict_recurrence(model, input_df, scaler)
        level, color = risk_level(prob)

        st.markdown("---")
        st.header("Prediction Result")

        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
        with res_col1:
            st.metric("Prediction", "Recurrence" if pred == 1 else "No Recurrence")
        with res_col2:
            st.metric("Recurrence Probability", f"{prob*100:.2f}%")
        with res_col3:
            st.metric("Risk Level", level)

        follow_up = (
            "Close follow-up and more aggressive treatment strategies are recommended."
            if prob >= 0.5
            else "Standard follow-up with routine monitoring is recommended."
        )
        st.markdown(f"""
        <div style='background-color:{color}20;padding:20px;border-radius:10px;border-left:6px solid {color};margin-top:15px;'>
        <h3 style='color:{color};margin:0;'>Risk Assessment: {level}</h3>
        <p style='margin-top:10px;font-size:16px;'>
        Predicted recurrence probability is <b>{prob*100:.2f}%</b>. {follow_up}
        </p>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.barh([0], [100], color='lightgray', height=0.5, alpha=0.3, zorder=0)
        ax.barh([0], [prob*100], color=color, height=0.5, edgecolor='black')
        ax.axvline(x=20, color='green', linestyle='--', alpha=0.6, label='Low (20%)')
        ax.axvline(x=50, color='orange', linestyle='--', alpha=0.6, label='Moderate (50%)')
        ax.axvline(x=75, color='red', linestyle='--', alpha=0.6, label='High (75%)')
        ax.text(prob*100, 0, f' {prob*100:.1f}%', va='center', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel('Recurrence Probability (%)', fontsize=12)
        ax.set_title('Patient Recurrence Risk', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("View encoded input"):
            st.dataframe(input_df.T.rename(columns={0: 'Encoded Value'}), use_container_width=True)

        st.warning(
            "**Disclaimer**: This prediction is for clinical reference only and cannot "
            "replace professional medical diagnosis. All treatment decisions should be "
            "made by a qualified physician based on a complete clinical evaluation."
        )


# ============================================================
# Page 2: Batch Prediction
# ============================================================
elif page == "Batch Prediction":
    st.header("Batch Prediction")
    st.markdown("Upload a CSV file for batch prediction. The file must contain the following **16 columns** (encoded as integers):")

    st.code(", ".join(FEATURE_NAMES), language="text")

    col_a, col_b = st.columns([2, 1])
    with col_b:
        st.download_button(
            "Download CSV template",
            data=pd.DataFrame(columns=FEATURE_NAMES).to_csv(index=False).encode('utf-8-sig'),
            file_name="prediction_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        test_set_path = os.path.join(MODEL_DIR, "testing_set.csv")
        if os.path.exists(test_set_path):
            with open(test_set_path, "rb") as f:
                st.download_button(
                    "Download test-set example",
                    data=f.read(),
                    file_name="testing_set_example.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            missing = [c for c in FEATURE_NAMES if c not in df_in.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                st.success(f"Loaded {len(df_in)} records")
                st.dataframe(df_in.head(), use_container_width=True)

                if st.button("Run batch prediction", type="primary", use_container_width=True):
                    X = df_in[FEATURE_NAMES].values
                    # Apply scaling if scaler is available
                    if scaler is not None:
                        X = scaler.transform(X)
                    probs = model.predict_proba(X)[:, 1]
                    preds = model.predict(X)

                    result = df_in.copy()
                    result['Recurrence_Probability'] = np.round(probs, 4)
                    result['Prediction'] = np.where(preds == 1, 'Recurrence', 'No Recurrence')
                    result['Risk_Level'] = [risk_level(p)[0] for p in probs]

                    st.markdown("### Results")
                    st.dataframe(result, use_container_width=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Samples", len(result))
                    c2.metric("Predicted Recurrence", int((preds == 1).sum()))
                    c3.metric("Predicted No Recurrence", int((preds == 0).sum()))
                    c4.metric("Mean Probability", f"{probs.mean()*100:.2f}%")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(probs, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
                    ax.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
                    ax.set_xlabel('Recurrence Probability', fontsize=12)
                    ax.set_ylabel('Number of Samples', fontsize=12)
                    ax.set_title('Batch Prediction Probability Distribution', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)

                    st.download_button(
                        "Download results (CSV)",
                        data=result.to_csv(index=False).encode('utf-8-sig'),
                        file_name="prediction_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Failed to read file: {e}")


# ============================================================
# Page 3: Model Performance
# ============================================================
elif page == "Model Performance":
    st.header("Model Performance Comparison")

    if ranking_df is not None:
        st.markdown("### Test-set performance of 9 machine learning models")
        # Select only numeric columns for formatting
        numeric_cols = ranking_df.select_dtypes(include=['float64', 'int64']).columns
        styled = ranking_df.style.format({col: "{:.4f}" for col in numeric_cols}).background_gradient(
            subset=['AUC', 'F1-score', 'Accuracy'], cmap='Greens'
        ).background_gradient(subset=['Brier Score'], cmap='Reds_r')
        st.dataframe(styled, use_container_width=True)

        st.info(
            f"**Best model**: `{ranking_df.index[0]}`\n\n"
            f"- **AUC**: {ranking_df.iloc[0]['AUC']:.4f}\n"
            f"- **F1-score**: {ranking_df.iloc[0]['F1-score']:.4f}\n"
            f"- **Accuracy**: {ranking_df.iloc[0]['Accuracy']:.4f}"
        )

        st.markdown("### Metric Visualization")
        metric_choice = st.selectbox("Select a metric", ['AUC', 'F1-score', 'Accuracy', 'Precision', 'Recall', 'Brier Score'])
        fig, ax = plt.subplots(figsize=(12, 5))
        sorted_df = ranking_df.sort_values(metric_choice, ascending=(metric_choice == 'Brier Score'))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_df)))
        bars = ax.barh(sorted_df.index, sorted_df[metric_choice], color=colors, edgecolor='black')
        for bar, v in zip(bars, sorted_df[metric_choice]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f' {v:.4f}',
                    va='center', fontsize=10)
        ax.set_xlabel(metric_choice, fontsize=12)
        ax.set_title(f'{metric_choice} Across Models', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("### Training Visualizations")
    image_files = {
        "ROC Curves": "02_roc_curves_test.png",
        "Performance Bar Chart": "01_performance_comparison.png",
        "Calibration Curves": "03_calibration_curves_test.png",
        "Confusion Matrices": "04_confusion_matrices_test.png",
        "PR Curves": "05_pr_curves_test.png",
        "Radar Chart (Top 4)": "06_radar_chart.png",
        "Feature Importance": "07_feature_importance.png",
        "SHAP Global Importance": "08_shap_global.png",
        "SHAP Beeswarm": "09_shap_beeswarm.png",
        "SHAP - High Risk Case": "10_shap_high_risk.png",
        "SHAP - Low Risk Case": "11_shap_low_risk.png",
    }

    choice = st.selectbox("Select a chart", list(image_files.keys()))
    img_path = os.path.join(MODEL_DIR, image_files[choice])
    if os.path.exists(img_path):
        st.image(img_path, caption=choice, use_container_width=True)
    else:
        st.warning(f"Image not found: {img_path}")


# ============================================================
# Page 4: About
# ============================================================
elif page == "About":
    st.header("About This System")

    st.markdown("""
    ### Background
    Differentiated Thyroid Cancer (DTC) is the most common endocrine malignancy.
    Although its overall prognosis is favorable, a notable proportion of patients
    experience recurrence after treatment. Accurate recurrence-risk prediction is
    essential for individualized treatment and follow-up planning.

    ### Dataset
    - **Source**: UCI Machine Learning Repository - Thyroid Differentiated Cancer Recurrence
    - **Sample size**: 335 Papillary Thyroid Cancer (PTC) patients
    - **Features**: 16 clinical variables
    - **Train / Test split**: 234 / 101 (7:3, stratified)

    ### Modeling Workflow
    1. **Preprocessing**: categorical encoding and standardization
    2. **Training**: 9 machine learning models (LR, DT, SVM, KNN, NB, RF, XGBoost, LightGBM, CatBoost)
    3. **Hyperparameter tuning**: 5-fold stratified CV with GridSearchCV
    4. **Evaluation**: Accuracy, Precision, Recall, F1, AUC, Brier Score
    5. **Interpretability**: SHAP value analysis

    ### Best Model
    **LightGBM** achieved the best test-set performance:
    - AUC = **0.9830**
    - F1-score = **0.9412**
    - Accuracy = **97.03%**
    - Brier Score = **0.0264**

    ### 16 Clinical Features
    """)

    feat_info = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Type': ['Continuous', 'Binary', 'Binary', 'Binary', 'Binary',
                 'Multi-class (5)', 'Multi-class (5)', 'Multi-class (6)',
                 'Binary', 'Binary', 'Multi-class (3)',
                 'Multi-class (7)', 'Multi-class (3)', 'Binary',
                 'Multi-class (5)', 'Multi-class (4)'],
        'Meaning': ['Age', 'Gender', 'Current smoking', 'Smoking history',
                    'Radiotherapy history', 'Thyroid function',
                    'Physical examination', 'Adenopathy', 'Pathology type',
                    'Focality', 'Risk stratification', 'T stage', 'N stage',
                    'M stage', 'Overall stage', 'Treatment response']
    })
    st.dataframe(feat_info, use_container_width=True, hide_index=True)

    if dataset_info:
        st.markdown("### Dataset Information")
        info_df = pd.DataFrame([
            {'Item': k, 'Value': v} for k, v in dataset_info.items() if k != 'feature_names'
        ])
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("""
    ### Disclaimer
    This system is intended for research and educational purposes only. Predictions
    must not replace professional medical diagnosis. Any medical decision must be
    made by a qualified physician based on a complete clinical evaluation.

    ### Tech Stack
    - **Python 3.8+**
    - **scikit-learn** - classical machine learning
    - **LightGBM / XGBoost / CatBoost** - gradient boosting
    - **SHAP** - model interpretability
    - **Streamlit** - web application framework
    """)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:13px;'>"
    "Thyroid Cancer Recurrence Prediction System | "
    "Best Model: LightGBM (AUC=0.983) | "
    "Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
