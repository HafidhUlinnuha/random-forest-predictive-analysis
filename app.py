import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.express as px  # Chart interaktif

st.set_page_config(page_title="Moms Chicken AI Forecast", layout="wide")

st.title("🚀 AI Demand Forecasting Moms Chicken")
st.markdown("---")

# SIDEBAR UPLOAD
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Pilih Excel (Tanggal, Penjualan Ayam, Hari)", type=['xlsx','xls'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success(f"✅ Data loaded: {len(df)} hari")
    st.dataframe(df.tail(), use_container_width=True)
    
    # RUN AI (sama kayak script)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Hari_num'] = pd.Categorical(df['Hari']).codes
    df['Lag1'] = df['Penjualan Ayam'].shift(1)
    df['Lag7'] = df['Penjualan Ayam'].shift(7)
    df['Trend'] = np.arange(len(df))
    df.dropna(inplace=True)
    
    X = df[['Hari_num', 'Lag1', 'Lag7', 'Trend']]
    y = df['Penjualan Ayam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # METRICS
    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    accuracy = (1 - mae/np.mean(y)) * 100
    
    st.metric("🎯 Akurasi", f"{accuracy:.1f}%")
    st.metric("📏 Error Rata-Rata", f"{mae:.1f} ayam")
    
    # PREDIKSI
    future_dates = pd.date_range(start=df['Tanggal'].max() + timedelta(1), periods=7)
    future_hari_num = pd.Categorical(future_dates.day_name()).codes
    future_lag1 = y.iloc[-1]
    future_lag7 = y.iloc[-7] if len(y)>=7 else future_lag1
    future_trend = np.arange(len(df), len(df)+7)
    X_future = pd.DataFrame({'Hari_num': future_hari_num, 'Lag1': future_lag1, 'Lag7': future_lag7, 'Trend': future_trend})
    pred_future = model.predict(X_future)
    
    # CHART INTERAKTIF
    df_plot = df.tail(60)
    fig = px.line(df_plot, x='Tanggal', y='Penjualan Ayam', 
                  title="AI Forecasting", labels={'Penjualan Ayam': 'Ayam (unit)'})
    fig.add_scatter(x=future_dates, y=pred_future, mode='lines+markers', 
                    name='Prediksi', line=dict(color='red', width=4))
    st.plotly_chart(fig, use_container_width=True)
    
    # TABEL PREDIKSI
    pred_df = pd.DataFrame({'Tanggal': future_dates, 'Prediksi': np.round(pred_future).astype(int)})
    st.subheader("🔮 Prediksi 7 Hari")
    st.table(pred_df)
    
    # INSIGHT
    importance = dict(zip(X.columns, model.feature_importances_))
    st.subheader("💡 Insight AI")
    imp_df = pd.DataFrame(list(importance.items()), columns=['Fitur', 'Pentingnya'])
    imp_df['Pentingnya'] = (imp_df['Pentingnya']*100).round(1)
    st.bar_chart(imp_df.set_index('Fitur'))

else:
    st.info("👆 Upload Excel dulu → Otomatis analisa + prediksi!")

st.markdown("---")
st.caption("Made with ❤️ untuk Moms Chicken | Streamlit + Scikit-learn")