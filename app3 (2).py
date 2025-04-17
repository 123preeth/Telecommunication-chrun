import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Page Configuration
st.set_page_config(page_title="Telecommunication Churn Prediction", layout="wide")

# Title
st.title("Telecommunication Churn Prediction")
st.write("Upload your dataset and select a machine learning model to predict churn.")

# Sidebar for file upload
data_file = st.sidebar.file_uploader("Upload your dataset (EXCEL)", type=["xlsx"])

if data_file:
    # Load the dataset
    df = pd.read_excel(data_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

# Data Preprocessing
    st.write("### Data Preprocessing")
    st.write("Automatically handling missing values and encoding categorical data...")

    # Handle missing values for numeric columns only
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    st.write("Processed Data:")
    st.dataframe(df.head())

    # Feature and Target Selection
    features = st.sidebar.multiselect("Select Feature Columns", options=df.columns)
    target = st.sidebar.selectbox("Select Target Column", options=df.columns)

    if features and target:
        X = df[features]
        y = df[target]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Selection
        model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display Metrics
        st.write("### Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Make predictions based on user input
        st.write("### Make Predictions")
        st.write("Enter values for prediction below:")

        # Collect input for prediction
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(f"Enter value for {feature}", value=float(X[feature].mean()))

        if st.button("Predict Churn"):
            input_df = pd.DataFrame([input_data])

            # Align input data with training columns
            input_df = pd.get_dummies(input_df, drop_first=True).reindex(columns=X.columns, fill_value=0)

            # Make prediction
            prediction = model.predict(input_df)

            st.write("### Prediction Result")
            st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

else:
    st.write("Upload a dataset to start.")
