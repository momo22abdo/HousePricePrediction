import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]
# Set page configuration
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Load saved models and preprocessing pipeline
try:
    preprocessing = joblib.load('preprocessing.pkl')
    lin_reg = joblib.load('linear_regression_model.pkl')
    best_forest = joblib.load('random_forest_model.pkl')
    best_xgb = joblib.load('xgboost_model.pkl')
    st.success("Models and preprocessing pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading models or preprocessing pipeline: {e}")
    st.error("Please ensure 'preprocessing.pkl', 'linear_regression_model.pkl', 'random_forest_model.pkl', and 'xgboost_model.pkl' are in the same directory as this script ('C:\\Users\\Momo\\PycharmProjects\\PythonProject2').")
    st.stop()

# Load test data for visualizations
try:
    strat_train_set = pd.read_csv(r"C:\Users\Momo\Downloads\housing.csv")
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    from sklearn.model_selection import train_test_split
    housing_train, housing_test, labels_train, labels_test = train_test_split(
        housing, housing_labels, test_size=0.2, random_state=42
    )
    st.success("Test data loaded successfully!")
except Exception as e:
    st.error(f"Error loading test data: {e}")
    st.error("Please ensure 'housing.csv' is located at 'C:\\Users\\Momo\\Downloads\\housing.csv' or update the path in the script.")
    st.stop()

# Streamlit app
st.title("🏠 California House Price Prediction")
st.markdown("""
This app predicts house prices using Linear Regression, Random Forest, and XGBoost models.
Enter the house details below and click **Predict** to see the results.
""")

# Input form
st.header("Enter House Details")
with st.form("house_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        longitude = st.number_input("Longitude", min_value=-124.35, max_value=-114.31, value=-122.41, step=0.01)
        latitude = st.number_input("Latitude", min_value=32.54, max_value=41.95, value=37.77, step=0.01)
        housing_median_age = st.number_input("Housing Median Age", min_value=1.0, max_value=52.0, value=35.0, step=1.0)
        total_rooms = st.number_input("Total Rooms", min_value=2.0, max_value=39320.0, value=1000.0, step=1.0)
        total_bedrooms = st.number_input("Total Bedrooms", min_value=1.0, max_value=6445.0, value=200.0, step=1.0)
    
    with col2:
        population = st.number_input("Population", min_value=3.0, max_value=35682.0, value=500.0, step=1.0)
        households = st.number_input("Households", min_value=1.0, max_value=6082.0, value=200.0, step=1.0)
        median_income = st.number_input("Median Income ($)", min_value=0.5, max_value=15.0, value=3.0, step=0.1)
        ocean_proximity = st.selectbox("Ocean Proximity", 
                                       options=['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])
    
    submitted = st.form_submit_button("Predict")

# Prepare input data for prediction
if submitted:
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    
    try:
        # Transform input data
        input_transformed = preprocessing.transform(input_data)
        
        # Make predictions
        lin_reg_pred = lin_reg.predict(input_data)
        rf_pred = best_forest.predict(input_data)
        xgb_pred = best_xgb.predict(input_data)
        
        st.header("Prediction Results")
        st.write(f"**Linear Regression Prediction**: ${lin_reg_pred[0]:,.2f}")
        st.write(f"**Random Forest Prediction**: ${rf_pred[0]:,.2f}")
        st.write(f"**XGBoost Prediction**: ${xgb_pred[0]:,.2f}")
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        st.error("Please check input values and ensure they are valid (e.g., within the allowed ranges).")

# Generate test set predictions
st.header("Model Performance Visualizations")
st.markdown("Below are the prediction vs. true value plots and error distributions for the test set.")

try:
    test_predictions_lr = lin_reg.predict(housing_test)
    test_predictions_rf = best_forest.predict(housing_test)
    test_predictions_xgb = best_xgb.predict(housing_test)
    st.success("Test set predictions generated successfully!")

    # Define models dictionary after predictions are computed
    models = {
        'Linear Regression': test_predictions_lr,
        'Random Forest': test_predictions_rf,
        'XGBoost': test_predictions_xgb
    }

    for model_name, predictions in models.items():
        percent_error = 100 * (predictions - labels_test) / labels_test

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax1.scatter(labels_test, predictions, alpha=0.3)
        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title(f"Test Predictions vs True Values ({model_name})")

        # Error histogram
        ax2.hist(percent_error, bins=50, edgecolor='black')
        ax2.set_xlabel('Percentage Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Test Prediction Error Distribution ({model_name})')
        ax2.grid(True)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

except Exception as e:
    st.error(f"Error generating test set predictions: {e}")
    st.error(
        "This could be due to a mismatch between the preprocessing pipeline and test data, or missing model files.")
    st.stop()

# Feature Importance
st.header("Feature Importance")
st.markdown("The following plots show the feature importance for Random Forest and XGBoost models.")

# Random Forest Feature Importance
try:
    best_rf = best_forest.named_steps['randomforestregressor']
    feature_importance_rf = pd.Series(best_rf.feature_importances_, index=preprocessing.get_feature_names_out())
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance_rf.sort_values().plot(kind='barh', ax=ax)
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
except Exception as e:
    st.error(f"Error plotting Random Forest feature importance: {e}")

# XGBoost Feature Importance
try:
    best_xgb_model = best_xgb.named_steps['xgbregressor']
    feature_importance_xgb = pd.Series(best_xgb_model.feature_importances_, index=preprocessing.get_feature_names_out())
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importance_xgb.sort_values().plot(kind='barh', ax=ax)
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
except Exception as e:
    st.error(f"Error plotting XGBoost feature importance: {e}")