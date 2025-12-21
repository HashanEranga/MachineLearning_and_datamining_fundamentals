import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🏦 Customer Churn Prediction System")
st.markdown("""
This application predicts whether a bank customer is likely to churn (leave the bank)
based on their profile and account information using an Artificial Neural Network (ANN) model.
""")

# Load the model and encoders
@st.cache_resource
def load_model_and_encoders():
    """Load the trained model, encoders, and scaler"""
    model = load_model('model.keras')

    with open('label_encoder_geography.pkl', 'rb') as file:
        label_encoder_geo = pickle.load(file)

    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_geo, label_encoder_gender, scaler

# Load resources
try:
    model, label_encoder_geo, label_encoder_gender, scaler = load_model_and_encoders()
    st.sidebar.success("✅ Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

# Sidebar for input
st.sidebar.header("Customer Information")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Information")
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.slider("Age", 18, 92, 35)

with col2:
    st.subheader("Account Information")
    credit_score = st.slider("Credit Score", 350, 850, 650)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=80000.0, step=1000.0)

col3, col4 = st.columns(2)

with col3:
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_credit_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

with col4:
    is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=75000.0, step=1000.0)

# Predict button
if st.button("🔮 Predict Churn Probability", type="primary"):
    # Create input dataframe
    input_data = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_credit_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }

    input_df = pd.DataFrame(input_data)

    # Preprocessing
    # Step 1: Encode Gender
    input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])

    # Step 2: Encode Geography using OneHotEncoder
    geo_encoded = label_encoder_geo.transform(input_df[['Geography']])
    geo_encoded_df = pd.DataFrame(
        data=geo_encoded,
        columns=label_encoder_geo.get_feature_names_out()
    )

    # Step 3: Combine encoded geography with other features
    input_df = pd.concat([input_df.drop(['Geography'], axis=1), geo_encoded_df], axis=1)

    # Step 4: Scale the features
    input_scaled = scaler.transform(input_df)

    # Step 5: Make prediction
    prediction = model.predict(input_scaled, verbose=0)
    churn_probability = prediction[0][0]

    # Determine risk level
    if churn_probability > 0.7:
        risk_level = "High"
        risk_color = "🔴"
    elif churn_probability > 0.4:
        risk_level = "Medium"
        risk_color = "🟡"
    else:
        risk_level = "Low"
        risk_color = "🟢"

    # Display results
    st.markdown("---")
    st.header("📊 Prediction Results")

    # Create three columns for results
    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_probability * 100:.2f}%"
        )

    with result_col2:
        st.metric(
            label="Predicted Outcome",
            value="Will Churn" if churn_probability  > 0.5 else "Will Stay"
        )

    with result_col3:
        st.metric(
            label="Risk Level",
            value=f"{risk_color} {risk_level}"
        )

    # Progress bar for visualization
    st.subheader("Churn Probability Visualization")
    st.progress(float(churn_probability))

    # Detailed interpretation
    st.subheader("📝 Interpretation")
    if churn_probability > 0.7:
        st.error(f"""
        **High Risk Customer**: This customer has a {churn_probability * 100:.2f}% probability of churning.

        **Recommended Actions:**
        - Immediate customer retention intervention required
        - Consider offering personalized retention incentives
        - Schedule a personal call from relationship manager
        - Review account for any service issues
        """)
    elif churn_probability > 0.4:
        st.warning(f"""
        **Medium Risk Customer**: This customer has a {churn_probability * 100:.2f}% probability of churning.

        **Recommended Actions:**
        - Monitor customer engagement closely
        - Offer product recommendations or upgrades
        - Send targeted communication about value-added services
        - Consider loyalty program enrollment
        """)
    else:
        st.success(f"""
        **Low Risk Customer**: This customer has only a {churn_probability * 100:.2f}% probability of churning.

        **Status:**
        - Customer appears satisfied and engaged
        - Continue standard engagement practices
        - Monitor for any changes in behavior
        """)

    # Display input summary
    with st.expander("📋 View Input Summary"):
        summary_df = pd.DataFrame({
            'Feature': ['Geography', 'Gender', 'Age', 'Credit Score', 'Tenure', 'Balance',
                       'Number of Products', 'Has Credit Card', 'Is Active Member', 'Estimated Salary'],
            'Value': [geography, gender, age, credit_score, tenure, f"${balance:,.2f}",
                     num_products, 'Yes' if has_credit_card == 1 else 'No',
                     'Yes' if is_active_member == 1 else 'No', f"${estimated_salary:,.2f}"]
        })
        st.table(summary_df)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and TensorFlow | ANN Churn Prediction Model</p>
</div>
""", unsafe_allow_html=True)
