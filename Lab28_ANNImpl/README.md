# Customer Churn Prediction - Streamlit App

This is a Streamlit web application for predicting customer churn using an Artificial Neural Network (ANN) model.

## Features

- Interactive user interface for inputting customer information
- Real-time churn probability prediction
- Risk level assessment (Low, Medium, High)
- Visualization of prediction results
- Actionable recommendations based on risk level

## Prerequisites

- Python 3.8 or higher
- Required packages (see requirements.txt)

## Installation

1. Navigate to the project directory:
```bash
cd Lab28_ANNImpl
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit app, execute the following command in your terminal:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Enter Customer Information**: Use the sidebar and main interface to input:
   - Personal Information: Geography, Gender, Age
   - Account Information: Credit Score, Tenure, Balance
   - Product Information: Number of Products, Credit Card status, Active Member status, Estimated Salary

2. **Get Prediction**: Click the "Predict Churn Probability" button

3. **View Results**: The app will display:
   - Churn probability percentage
   - Predicted outcome (Will Churn / Will Stay)
   - Risk level with color-coded indicator
   - Visualization and detailed recommendations

## Model Details

- **Model Type**: Artificial Neural Network (ANN)
- **Architecture**:
  - Input Layer: 12 features
  - Hidden Layer 1: 64 neurons (ReLU activation)
  - Hidden Layer 2: 32 neurons (ReLU activation)
  - Output Layer: 1 neuron (Sigmoid activation)
- **Encoders Used**:
  - LabelEncoder for Gender
  - OneHotEncoder for Geography
  - StandardScaler for feature scaling

## Files Required

Make sure the following files are in the same directory as `app.py`:
- `model.keras` - Trained ANN model
- `label_encoder_gender.pkl` - Gender label encoder
- `label_encoder_geography.pkl` - Geography one-hot encoder
- `scaler.pkl` - Standard scaler for features

## Input Features

The model accepts the following customer features:
1. Credit Score (350-850)
2. Geography (France, Germany, Spain)
3. Gender (Male, Female)
4. Age (18-92)
5. Tenure (0-10 years)
6. Balance (0-250,000)
7. Number of Products (1-4)
8. Has Credit Card (Yes/No)
9. Is Active Member (Yes/No)
10. Estimated Salary (0-200,000)

## Output

- **Churn Probability**: Percentage likelihood of customer churn
- **Risk Level**:
  - Low Risk: < 40%
  - Medium Risk: 40-70%
  - High Risk: > 70%
- **Recommendations**: Actionable insights based on risk level

## License

This project is for educational purposes.
