# Customer-Segmentation-using-Clustering-Classfication


## 📌 Project Overview
This project focuses on analyzing customer and sales data to identify meaningful customer segments using **unsupervised machine learning** techniques and then predicting these segments for new customers using **supervised learning**.  
The complete solution is deployed as a **Streamlit web application** for real-time customer segmentation.

---

## 🎯 Objectives
- Segment customers based on purchasing behavior and demographics
- Build a classification model to predict customer segments for new data
- Deploy the trained model as an interactive web application
- Enable data-driven customer targeting and business insights

---

## 🗂️ Dataset Description

### Sales Data (`sales_data.csv`)
- `order_id`
- `product_id`
- `quantity`
- `unit_price`
- `payment_method`
- `delivery_status`

### Customer Data (`customer_info.csv`)
- `customer_id`
- `gender`
- `region`
- `loyalty_tier`

### Product Data (`product_info.csv`)
- `product_id`
- `category`
- `base_price`
- `supplier_code`

All datasets are merged into a single dataset to capture complete customer behavior.

---

## 🧹 Data Preprocessing
- Converted textual quantities (e.g., `"three"`, `"five"`) into numeric values
- Normalized inconsistent categorical values (delivery status, payment method)
- Corrected typographical errors in region and loyalty tier names
- Handled missing values using:
  - **Median imputation** for numerical features
  - **Most frequent imputation** for categorical features

---

## 🧠 Feature Engineering
- **Total Price** = `quantity × unit_price`
- **Price Tier** categorized into:
  - Budget
  - Mid
  - Premium
  - Luxury

---

## 🔍 Unsupervised Learning: Customer Segmentation
- **Algorithm:** K-Means Clustering
- **Number of Clusters:** 3

### Features Used
- Quantity  
- Unit Price  
- Total Price  
- Gender  
- Region  
- Loyalty Tier  
- Product Category  

### Evaluation Metrics
- **Silhouette Score:** 0.137  
- **Davies–Bouldin Index:** 2.29  

**Visualization:**  
Principal Component Analysis (PCA) was used to visualize clusters in two dimensions.

---

## 🎯 Supervised Learning: Segment Prediction
To enable real-time predictions, a classification model was trained using the generated cluster labels.

- **Model:** Random Forest Classifier
- **n_estimators:** 100

### Preprocessing Pipeline
- Numerical features → Median imputation
- Categorical features → Mode imputation + One-Hot Encoding
- Ordinal features (Loyalty Tier) → Ordinal Encoding

### Model Performance
- **Accuracy:** 99%
- **Precision / Recall / F1-Score:** ~0.99 for all classes

> **Note:** High accuracy is expected since the classifier learns to replicate K-Means cluster assignments.

---

## 🚀 Deployment (Streamlit App)
The trained model and preprocessing pipeline were deployed using **Streamlit**.

### Application Features
- User-friendly input interface
- Loads trained model (`model.pkl`)
- Predicts customer segment in real time
- Displays business-friendly labels:
  - High-Value Customers
  - Occasional Buyers
  - New / Churn-Risk Customers

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Models:** K-Means, Random Forest  
- **Visualization:** Matplotlib, PCA  
- **Deployment:** Streamlit  

---

## 📂 Project Structure
```bash
├── data/
│   ├── sales_data.csv
│   ├── customer_info.csv
│   └── product_info.csv
├── notebooks/
│   └── ML Project.ipynb
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
