import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Konfigurasi Halaman Utama
st.set_page_config(page_title="Klasifikasi Harga Mobil", layout="wide")
st.title("🚗 Dashboard Klasifikasi Kategori Harga Mobil Bekas (XGBoost)")
st.write("Studi Kasus: Memprediksi apakah sebuah mobil bekas masuk kategori kelas **Murah, Menengah, atau Mahal**.")

# Cache Data & Model agar tidak perlu dimuat ulang setiap kali ada interaksi UI
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("autos_clean.csv")
    
    # Cleaning dasar: Filter anomali harga yang tidak masuk akal
    df = df[(df['price'] > 100) & (df['price'] < 100000)].copy()
    
    # Seleksi fitur yang relevan
    features = ['yearOfRegistration', 'powerPS', 'kilometer', 'vehicleType', 'gearbox', 'fuelType']
    df = df[features + ['price']].dropna()
    
    # Membentuk target KLASIFIKASI: Membagi harga menjadi 3 kelas yang seimbang
    df['price_category'] = pd.qcut(df['price'], 3, labels=['Murah', 'Menengah', 'Mahal'])
    
    return df, features

@st.cache_resource
def train_model(df, features):
    X = df[features].copy()
    y = df['price_category'].copy()
    
    encoders = {}
    cat_cols = ['vehicleType', 'gearbox', 'fuelType']
    
    # Label Encoding untuk kolom kategorikal (Syarat wajib untuk XGBoost jika tidak pakai One-Hot)
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        
    # Encoding target string ('Murah', dll) ke numerik (0, 1, 2)
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Inisialisasi dan training model XGBoost Classifier
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, encoders, target_le, acc

# Eksekusi pemrosesan di balik layar
with st.spinner("Mengekstraksi dataset dan melatih algoritma XGBoost..."):
    df, features = load_and_prep_data()
    model, encoders, target_le, accuracy = train_model(df, features)

st.success(f"✅ Model XGBoost berhasil dilatih! (Akurasi Prediksi: {accuracy:.2%})")

# Konfigurasi Sidebar untuk Input User
st.sidebar.header("Input Spesifikasi Mobil")

def user_input_features():
    year = st.sidebar.slider("Tahun Registrasi", 1990, 2016, 2005)
    power = st.sidebar.number_input("Tenaga Mesin (PS)", min_value=0, max_value=500, value=100)
    km = st.sidebar.selectbox("Kilometer Terpakai", sorted(df['kilometer'].unique()))
    
    v_type = st.sidebar.selectbox("Tipe Kendaraan", df['vehicleType'].unique())
    gear = st.sidebar.selectbox("Transmisi", df['gearbox'].unique())
    fuel = st.sidebar.selectbox("Tipe Bahan Bakar", df['fuelType'].unique())
    
    data = {
        'yearOfRegistration': year,
        'powerPS': power,
        'kilometer': km,
        'vehicleType': v_type,
        'gearbox': gear,
        'fuelType': fuel
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Spesifikasi Kendaraan:")
st.dataframe(input_df)

# Eksekusi Prediksi
if st.button("Prediksi Kategori Harga"):
    input_encoded = input_df.copy()
    
    # Transformasi input user menggunakan encoder yang sama dengan saat training
    for col, le in encoders.items():
        if input_encoded[col][0] in le.classes_:
            input_encoded[col] = le.transform(input_encoded[col])
        else:
            input_encoded[col] = 0 
            
    # Proses Prediksi
    pred_encoded = model.predict(input_encoded)
    pred_label = target_le.inverse_transform(pred_encoded)[0]
    
    st.markdown("---")
    st.subheader("Hasil Klasifikasi:")
    if pred_label == 'Murah':
        st.info(f"Kategori Estimasi: **{pred_label.upper()}** 📉")
    elif pred_label == 'Menengah':
        st.warning(f"Kategori Estimasi: **{pred_label.upper()}** 📊")
    else:
        st.error(f"Kategori Estimasi: **{pred_label.upper()}** 📈")
        
    st.write("*Model XGBoost ini memprediksi rentang kelas harga berdasarkan pola klasifikasi, bukan angka regresi absolut.*")
