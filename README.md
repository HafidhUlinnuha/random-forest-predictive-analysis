# Predictive Inventory System for Local F&B Business (Moms Chicken)

This repository contains a data-driven web application developed using Python and Streamlit to forecast daily food inventory requirements for a local F&B merchant, mitigating operational inefficiencies and food waste.

## 🚀 Live Demo
You can access the live interactive application here: https://random-forest-predictive-analysis-fvsvgj948xayugamyaudzb.streamlit.app

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Framework:** Streamlit (For Interactive UI)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest Regression Algorithm)

## 📊 Project Alur / Logic
1. **Data Ingestion:** Processed historical raw sales fluctuations data from the merchant.
2. **Feature Engineering:** Cleaned missing values and prepared structured metrics using Pandas.
3. **Model Training:** Utilized Random Forest Regression to project future inventory needs.
4. **Deployment:** Built a user-friendly UI via Streamlit tailored for non-technical business owners.

## 💻 How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit web app:
   ```bash
   streamlit run app.py
   ```
