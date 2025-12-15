import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved pipeline
# Ensure the path is correct relative to where you run this script
try:
    pipeline = joblib.load(r'G:\Other computers\My Laptop\Education and Bootcamp\COMSATS\5th Semester - Fall 2025\MGT - 244 Business Application using Machine Learning\Project\model.pkl')
except FileNotFoundError:
    st.error("Model file not found! Please check the path.")
    st.stop()

# 2. App Title and Description
st.title("Customer Segmentation Predictor")
st.markdown("Enter customer and purchase details to predict their segment.")

# 3. Create Inputs (Sidebar or Main Column)
with st.form("prediction_form"):
    st.header("Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quantity = st.slider("Quantity", 1, 10, 1)
        unit_price = st.number_input("Unit Price", value=20.0)
        base_price = st.number_input("Base Price (Cost)", value=15.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
    with col2:
        payment_method = st.selectbox("Payment Method", ["Credit Card", "PayPal", "Bank Transfer"])
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        category = st.selectbox("Category", ["Cleaning", "Outdoors", "Kitchen", "Personal Care", "Storage"])
        loyalty_tier = st.selectbox("Loyalty Tier", ["Bronze", "Silver", "Gold"])
        
    submitted = st.form_submit_button("Predict Segment")

# 4. Prediction Logic
if submitted:
    # Calculate derived features
    total_price = quantity * unit_price
    
    # Calculate price_tier logic
    if base_price <= 14.67:
        price_tier = "Budget"
    elif base_price <= 23.05:
        price_tier = "Mid"
    elif base_price <= 31.96:
        price_tier = "Premium"
    else:
        price_tier = "Luxury"

    # Create DataFrame
    input_df = pd.DataFrame({
        'quantity': [quantity],
        'unit_price': [unit_price],
        'total_price': [total_price],
        'base_price': [base_price],
        'gender': [gender],
        'payment_method': [payment_method],
        'region_x': [region],
        'category': [category],
        'price_tier': [price_tier],
        'loyalty_tier': [loyalty_tier]
    })

    # --- CRITICAL FIX: Add dummy 'labels' column ---
    # This prevents the "columns are missing: {'labels'}" error
    input_df['labels'] = 0 
    # -----------------------------------------------

    try:
        prediction = pipeline.predict(input_df)[0]
        
        # Display Result
        st.success(f"Predicted Cluster: **{prediction}**")
        
        if prediction == 0:
            st.info("Cluster 0 Analysis: likely High Value / Loyal customers.")
        elif prediction == 1:
            st.info("Cluster 1 Analysis: likely Budget / Occasional buyers.")
        elif prediction == 2:
            st.info("Cluster 2 Analysis: likely Mid-Range / New customers.")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        