import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor

# Load model
with open("catboost_model_quikr.pkl", "rb") as f:
    model = pickle.load(f)

# Define category values manually (must match training set)
company_models = [
    'Audi_Audi', 'BMW_BMW', 'Chevrolet_Chevrolet', 'Datsun_Datsun', 'Fiat_Fiat',
    'Force_Force', 'Ford_Ford', 'Hindustan_Hindustan', 'Honda_Honda', 'Hyundai_Hyundai',
    'Jeep_Jeep', 'Mahindra_Mahindra', 'Maruti_Maruti', 'Mercedes_Mercedes', 'Mitsubishi_Mitsubishi',
    'Nissan_Nissan', 'Renault_Renault', 'Skoda_Skoda', 'Tata_Tata', 'Toyota_Toyota', 'Volkswagen_Volkswagen'
]
fuel_types = ['Petrol', 'Diesel', 'LPG', 'CNG']

# Page layout
st.set_page_config(page_title="Prediksi Harga Mobil Bekas", layout="centered")
st.title("🚗 Prediksi Harga Mobil Bekas")

# User Inputs
company_model = st.selectbox("Pilih Merek & Model", sorted(company_models))
fuel_type = st.selectbox("Jenis Bahan Bakar", fuel_types)
age = st.slider("Umur Mobil (tahun)", 0, 30, 5)
kms_driven = st.number_input("Jarak Tempuh (dalam KM)", min_value=0, step=1000)

# Estimasi harga dasar berdasarkan merek untuk segmentasi awal
base_prices = {
    'Audi_Audi': 1500000,
    'BMW_BMW': 1400000,
    'Mercedes_Mercedes': 1600000,
    'Toyota_Toyota': 800000,
    'Honda_Honda': 700000,
    'Hyundai_Hyundai': 600000,
    'Maruti_Maruti': 300000,
    'Tata_Tata': 250000
}
estimated_price = base_prices.get(company_model, 400000)
segment = pd.cut([estimated_price], bins=[0, 200000, 400000, 600000, 800000, 1200000, 3000000],
                 labels=['ultra_low', 'low', 'mid_low', 'mid_high', 'high', 'lux'])[0]

fuel_age = f"{fuel_type}_{age}"
company_segment = f"{company_model}_{segment}"
log_km = np.log1p(kms_driven)
log_km_per_year = np.log1p(kms_driven / max(age, 1))

lux_brands = ['BMW_BMW', 'Mercedes_Mercedes', 'Audi_Audi']
mid_brands = ['Toyota_Toyota', 'Honda_Honda', 'Hyundai_Hyundai', 'Volkswagen_Volkswagen', 'Skoda_Skoda', 'Jeep_Jeep']
budget_brands = ['Tata_Tata', 'Datsun_Datsun', 'Maruti_Maruti', 'Hindustan_Hindustan', 'Force_Force', 'Fiat_Fiat']

if company_model in lux_brands:
    brand_category = 'luxury'
elif company_model in mid_brands:
    brand_category = 'midrange'
elif company_model in budget_brands:
    brand_category = 'budget'
else:
    brand_category = 'general'

is_premium = int(brand_category == 'luxury' and age <= 3)
is_high_value = int(brand_category in ['luxury', 'midrange'] and age <= 5 and kms_driven <= 50000)
is_low_budget = int(brand_category == 'budget' and age >= 10 and kms_driven >= 100000)

# Prediction
features = pd.DataFrame([{
    'company_model': company_model,
    'fuel_type': fuel_type,
    'log_km': log_km,
    'age': age,
    'segment': segment,
    'fuel_age': fuel_age,
    'company_segment': company_segment,
    'log_km_per_year': log_km_per_year,
    'brand_category': brand_category,
    'is_premium': is_premium,
    'is_high_value': is_high_value,
    'is_low_budget': is_low_budget
}])

if st.button("Prediksi Harga"):
    pred_log = model.predict(features)[0]
    pred_rp = np.expm1(pred_log)
    st.subheader(f"💰 Estimasi Harga: Rp {pred_rp:,.0f}")
