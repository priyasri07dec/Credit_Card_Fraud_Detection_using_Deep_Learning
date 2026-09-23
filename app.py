import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Main title */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .probability {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Sidebar */
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.25);
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# LOAD MODEL AND ARTIFACTS

@st.cache_resource
def load_model_and_artifacts():

    model = load_model("fraud_detection_ann.keras")

    scaler = joblib.load("scaler.pkl")

    feature_columns = joblib.load("feature_columns.pkl")

    threshold = joblib.load("threshold.pkl")

    return model, scaler, feature_columns, threshold


model, scaler, feature_columns, threshold = load_model_and_artifacts()

# SIDEBAR

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">💳 Fraud Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**Model:** Artificial Neural Network (ANN)")
    st.write("**Features:** 30")
    st.write(f"**Decision Threshold:** {threshold:.2f}")

    st.markdown("---")

    st.markdown("### Model Performance")

    st.write("**Accuracy:** 99.94%")
    st.write("**Precision:** 87.65%")
    st.write("**Recall:** 74.74%")
    st.write("**F1-Score:** 80.68%")
    st.write("**ROC-AUC:** 94.03%")
    st.write("**PR-AUC:** 66.47%")

    st.markdown("---")

    st.caption(
        "The model was trained on the Credit Card Fraud Detection dataset. "
        "V1–V28 are anonymized PCA-transformed features."
    )

# HEADER

st.markdown(
    '<div class="main-title">💳 Credit Card Fraud Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Artificial Neural Network based fraud detection system'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the transaction features below and click **Predict Transaction** "
    "to obtain the model's fraud classification."
)

# TRANSACTION INFORMATION

st.markdown(
    '<div class="section-title">📊 Transaction Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

input_data = {}


with col1:

    input_data["Time"] = st.number_input(
        "Transaction Time",
        value=0.0,
        format="%.4f",
        help="Time elapsed since the first transaction in the dataset."
    )


with col2:

    input_data["Amount"] = st.number_input(
        "Transaction Amount",
        value=0.0,
        min_value=0.0,
        format="%.4f",
        help="Transaction amount."
    )

# PCA FEATURES

st.markdown(
    '<div class="section-title">🔢 Anonymized PCA Features</div>',
    unsafe_allow_html=True
)

st.caption(
    "V1–V28 are anonymized PCA-transformed features from the original dataset."
)


pca_features = [
    feature for feature in feature_columns
    if feature.startswith("V")
]


# Create two columns
left_col, right_col = st.columns(2)

for i, feature in enumerate(pca_features):

    target_column = left_col if i % 2 == 0 else right_col

    with target_column:

        input_data[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )

# PREDICTION BUTTON

st.markdown("---")

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

    predict_button = st.button(
        "🔍 Predict Transaction",
        use_container_width=True
    )

# PREDICTION

if predict_button:

    # Create dataframe using exact feature order
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Generate probability
    probability = model.predict(
        input_scaled,
        verbose=0
    ).ravel()[0]

    # Apply threshold
    prediction = int(probability >= threshold)

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📌 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )

    with result_col2:

        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )


    # Result message
    if prediction == 1:

        st.error(
            "⚠️ Potentially Fraudulent Transaction"
        )

        st.warning(
            "The predicted fraud probability is at or above "
            "the model's decision threshold."
        )

    else:

        st.success(
            "✅ Legitimate Transaction"
        )

        st.success(
            "The predicted fraud probability is below "
            "the model's decision threshold."
        )


    # Show probability bar
    st.progress(
        min(float(probability), 1.0)
    )


    # Detailed result
    result_table = pd.DataFrame({
        "Output": [
            "Predicted Class",
            "Fraud Probability",
            "Decision Threshold"
        ],
        "Value": [
            "Fraud" if prediction == 1 else "Legitimate",
            f"{probability * 100:.2f}%",
            f"{threshold * 100:.0f}%"
        ]
    })

    st.dataframe(
        result_table,
        use_container_width=True,
        hide_index=True
    )

# FOOTER

st.markdown(
    """
    <div class="footer">
        Credit Card Fraud Detection | Artificial Neural Network | Deep Learning 
    </div>
    """,
    unsafe_allow_html=True
)