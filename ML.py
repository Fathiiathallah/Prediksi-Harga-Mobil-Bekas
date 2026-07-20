"""
===============================================================================
STREAMLIT APP — LIVE DEMO: PREDIKSI HARGA MOBIL BEKAS (XGBOOST)
===============================================================================
Studi kasus Boosting — Used Cars Database (eBay Kleinanzeigen, Jerman)

Cara menjalankan:
    1. pip install -r requirements.txt
    2. Taruh file "autos_clean.csv" di folder yang sama dengan app.py
       (atau upload langsung lewat sidebar saat aplikasi berjalan)
    3. streamlit run app.py

Tiga tab yang tersedia:
    1) Prediksi Live         -> input spesifikasi mobil, dapat estimasi harga instan
    2) Cara Hitung MAE&RMSE  -> rumus + perhitungan manual interaktif dari sampel data uji
    3) Evaluasi & Interpretasi -> feature importance, grafik aktual vs prediksi
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

RANDOM_STATE = 42
DEFAULT_PATH = "autos_clean.csv"

# Palet warna—disamakan dengan grafik di laporan (navy, teal, coral, sand)
NAVY, TEAL, CORAL, SAND = "#1f3a5f", "#2a9d8f", "#e76f51", "#e9c46a"

st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas — XGBoost",
    page_icon="🚗",
    layout="wide",
)


def format_euro(value: float) -> str:
    """Format angka ke gaya Eropa/Indonesia: 1.313,48 (bukan 1,313.48)."""
    s = f"{value:,.2f}"
    return s.replace(",", "@").replace(".", ",").replace("@", ".")


# ==============================================================================
# TAHAP 1-3: LOAD, BERSIHKAN, DAN FEATURE ENGINEERING DATA (di-cache)
# ==============================================================================
@st.cache_data(show_spinner="Memuat & membersihkan data...")
def load_and_clean_data(file):
    df = pd.read_csv(file)
    tahun_referensi = int(pd.to_datetime(df["dateCrawled"]).dt.year.max())

    df = df[
        df["price"].between(100, 150_000)
        & df["yearOfRegistration"].between(1950, tahun_referensi)
        & df["powerPS"].between(1, 1000)
    ].copy()

    kolom_kategorikal_missing = ["vehicleType", "gearbox", "model", "fuelType", "notRepairedDamage"]
    for col in kolom_kategorikal_missing:
        df[col] = df[col].fillna("tidak_diketahui")

    df["vehicleAge"] = tahun_referensi - df["yearOfRegistration"]

    kolom_dibuang = [
        "dateCrawled", "name", "seller", "offerType", "abtest",
        "nrOfPictures", "dateCreated", "lastSeen", "yearOfRegistration",
    ]
    df = df.drop(columns=kolom_dibuang)
    return df, tahun_referensi


# ==============================================================================
# TAHAP 4: ENCODING (di-cache, encoder disimpan supaya bisa dipakai ulang untuk
# mengubah input form prediksi live menjadi kode yang sama seperti saat training)
# ==============================================================================
@st.cache_data(show_spinner="Encoding fitur kategorikal...")
def encode_data(df):
    df_enc = df.copy()
    kolom_kategorikal = ["vehicleType", "gearbox", "model", "fuelType", "brand", "notRepairedDamage"]
    encoders = {}
    for col in kolom_kategorikal:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
    return df_enc, encoders


# ==============================================================================
# TAHAP 5-6: TRAINING MODEL XGBOOST (di-cache sebagai resource, bukan data biasa,
# supaya tidak dilatih ulang tiap kali ada interaksi di form)
# ==============================================================================
@st.cache_resource(show_spinner="Melatih model XGBoost (300 pohon)... mohon tunggu sebentar")
def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ==============================================================================
# APLIKASI UTAMA
# ==============================================================================
st.title("🚗 Prediksi Harga Mobil Bekas — XGBoost Regressor")
st.caption("Studi Kasus Boosting · Used Cars Database (eBay Kleinanzeigen, Jerman)")

with st.sidebar:
    st.header("📁 Sumber Data")
    uploaded = st.file_uploader("Upload autos_clean.csv (opsional)", type="csv")
    data_source = uploaded if uploaded is not None else DEFAULT_PATH
    st.caption(
        f"Jika tidak upload, aplikasi mencoba membaca **{DEFAULT_PATH}** "
        "dari folder yang sama dengan app.py."
    )

try:
    df_clean, tahun_referensi = load_and_clean_data(data_source)
except FileNotFoundError:
    st.error(
        f"File '{DEFAULT_PATH}' tidak ditemukan di folder ini. "
        "Silakan upload file CSV lewat sidebar di sebelah kiri."
    )
    st.stop()

df_enc, encoders = encode_data(df_clean)

X = df_enc.drop(columns=["price"])
y = df_enc["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model = train_model(X_train, y_train)

y_pred_test = model.predict(X_test)
mae_full = mean_absolute_error(y_test, y_pred_test)
rmse_full = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
r2_full = r2_score(y_test, y_pred_test)

with st.sidebar:
    st.divider()
    st.header("📊 Ringkasan Model")
    st.metric("Data setelah cleaning", f"{len(df_clean):,} baris".replace(",", "."))
    st.metric("Data uji (test set)", f"{len(X_test):,} baris".replace(",", "."))
    st.metric("R² Score", f"{r2_full:.4f}")
    st.metric("MAE", f"€{format_euro(mae_full)}")
    st.metric("RMSE", f"€{format_euro(rmse_full)}")

tab1, tab2, tab3 = st.tabs(
    ["🔮 Prediksi Live", "📐 Cara Hitung MAE & RMSE", "📊 Evaluasi & Interpretasi"]
)

# ------------------------------------------------------------------------------
# TAB 1 — PREDIKSI LIVE
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Masukkan Spesifikasi Mobil")
    st.caption("Isi form di bawah ini, lalu klik tombol Prediksi untuk melihat estimasi harga.")

    col1, col2, col3 = st.columns(3)

    with col1:
        brand_options = sorted(df_clean["brand"].unique())
        brand = st.selectbox("Merek (Brand)", brand_options)

        model_options = sorted(df_clean.loc[df_clean["brand"] == brand, "model"].unique())
        car_model = st.selectbox("Model", model_options)

        vehicle_type_options = sorted(df_clean["vehicleType"].unique())
        vehicle_type = st.selectbox("Tipe Kendaraan", vehicle_type_options)

    with col2:
        gearbox_options = sorted(df_clean["gearbox"].unique())
        gearbox = st.selectbox("Transmisi", gearbox_options)

        fuel_options = sorted(df_clean["fuelType"].unique())
        fuel_type = st.selectbox("Bahan Bakar", fuel_options)

        damage_options = sorted(df_clean["notRepairedDamage"].unique())
        not_repaired = st.selectbox("Kerusakan Belum Diperbaiki", damage_options)

    with col3:
        tahun_registrasi = st.slider(
            "Tahun Registrasi", 1950, tahun_referensi,
            value=min(2010, tahun_referensi),
        )
        power_ps = st.slider("Tenaga Mesin (PS)", 1, 1000, value=120)

        km_options = sorted(df_clean["kilometer"].unique().tolist())
        kilometer = st.select_slider(
            "Jarak Tempuh (km)", options=km_options,
            value=km_options[len(km_options) // 2],
        )

    with st.expander("Detail tambahan (opsional)"):
        col4, col5 = st.columns(2)
        with col4:
            postal_code = st.number_input(
                "Kode Pos", value=int(df_clean["postalCode"].median()), step=1
            )
        with col5:
            month_reg = st.slider("Bulan Registrasi", 0, 12, value=6)

    predict_clicked = st.button("🔮 Prediksi Harga", type="primary", width="stretch")

    if predict_clicked:
        vehicle_age = tahun_referensi - tahun_registrasi

        input_dict = {
            "vehicleType": encoders["vehicleType"].transform([vehicle_type])[0],
            "gearbox": encoders["gearbox"].transform([gearbox])[0],
            "powerPS": power_ps,
            "model": encoders["model"].transform([car_model])[0],
            "kilometer": kilometer,
            "monthOfRegistration": month_reg,
            "fuelType": encoders["fuelType"].transform([fuel_type])[0],
            "brand": encoders["brand"].transform([brand])[0],
            "notRepairedDamage": encoders["notRepairedDamage"].transform([not_repaired])[0],
            "postalCode": postal_code,
            "vehicleAge": vehicle_age,
        }
        input_df = pd.DataFrame([input_dict])[X_train.columns]
        predicted_price = float(model.predict(input_df)[0])

        st.success("Prediksi berhasil dihitung.")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("💰 Estimasi Harga Jual", f"€{format_euro(predicted_price)}")
            st.caption(f"Umur kendaraan: {vehicle_age} tahun")

        with res_col2:
            if predicted_price > 40_000:
                st.info(
                    "Estimasi berada di segmen mobil mewah (di atas €40.000). "
                    "Berdasarkan evaluasi model (lihat Tab 3), model cenderung "
                    "**underestimate** di segmen ini karena data latih pada rentang "
                    "harga tinggi jumlahnya jauh lebih sedikit dibanding segmen ekonomi-menengah."
                )
            else:
                st.info(
                    "Estimasi berada di segmen ekonomi-menengah (di bawah €40.000), "
                    "yaitu segmen dengan data latih terbanyak — sehingga prediksi model "
                    "pada rentang ini secara umum lebih akurat (lihat Tab 3)."
                )

# ------------------------------------------------------------------------------
# TAB 2 — CARA HITUNG MAE & RMSE (interaktif)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📐 Bagaimana MAE dan RMSE Dihitung?")
    st.markdown(
        "MAE dan RMSE sama-sama mengukur seberapa jauh harga hasil prediksi "
        "menyimpang dari harga aktual, namun dengan cara perhitungan yang berbeda."
    )

    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.latex(r"MAE = \frac{1}{n}\sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|")
        st.caption("Selisih diabsolutkan terlebih dahulu, lalu dirata-rata.")
    with fcol2:
        st.latex(r"RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2}")
        st.caption("Selisih dikuadratkan, dirata-rata, lalu diakarkan kembali.")

    st.caption(
        "Keterangan: **n** = jumlah data, **yᵢ** = harga aktual data ke-i, "
        "**ŷᵢ** = harga hasil prediksi model untuk data ke-i."
    )

    st.divider()
    st.markdown("#### Contoh Ilustrasi (5 Mobil)")
    st.caption("Contoh sederhana untuk melihat cara kerja rumus di atas, sebelum masuk ke data sungguhan.")

    contoh = pd.DataFrame({
        "Harga Aktual (€)": [10000, 8000, 15000, 20000, 30000],
        "Harga Prediksi (€)": [9000, 8500, 14000, 21000, 24000],
    })
    contoh["Selisih (yᵢ−ŷᵢ)"] = contoh["Harga Aktual (€)"] - contoh["Harga Prediksi (€)"]
    contoh["|Selisih|"] = contoh["Selisih (yᵢ−ŷᵢ)"].abs()
    contoh["Selisih²"] = contoh["Selisih (yᵢ−ŷᵢ)"] ** 2
    contoh_tampil = contoh.copy()
    for kol in contoh_tampil.columns:
        contoh_tampil[kol] = contoh_tampil[kol].apply(format_euro)
    st.dataframe(contoh_tampil, width="stretch", hide_index=True)

    contoh_mae = float(contoh["|Selisih|"].mean())
    contoh_rmse = float(np.sqrt(contoh["Selisih²"].mean()))
    ic1, ic2 = st.columns(2)
    ic1.metric("MAE contoh ini", f"€{format_euro(contoh_mae)}")
    ic2.metric("RMSE contoh ini", f"€{format_euro(contoh_rmse)}")
    st.caption(
        "4 dari 5 mobil di atas selisihnya kecil (500–1.000), tapi 1 mobil dengan selisih "
        "besar (6.000) membuat RMSE naik jauh lebih tinggi daripada MAE — karena selisih "
        "itu dikuadratkan dulu sebelum dirata-rata."
    )

    st.divider()
    st.markdown("#### Coba Hitung Sendiri dari Sampel Data Uji")
    st.caption(
        "Ambil beberapa baris acak dari data uji, lalu lihat perhitungan MAE & RMSE "
        "dilakukan langkah demi langkah menggunakan angka aktual dari model."
    )

    slider_col, button_col = st.columns([3, 1])
    with slider_col:
        n_sample = st.slider("Jumlah sampel mobil", min_value=3, max_value=20, value=5)
    with button_col:
        st.write("")
        st.write("")
        resample_clicked = st.button("🎲 Ambil Sampel Baru", width="stretch")

    need_new_sample = (
        "sample_idx" not in st.session_state
        or len(st.session_state["sample_idx"]) != n_sample
        or resample_clicked
    )
    if need_new_sample:
        st.session_state["sample_idx"] = np.random.choice(
            len(X_test), size=n_sample, replace=False
        )

    idx = st.session_state["sample_idx"]
    actual_sample = y_test.iloc[idx].to_numpy(dtype=float)
    pred_sample = model.predict(X_test.iloc[idx])

    selisih = actual_sample - pred_sample
    abs_selisih = np.abs(selisih)
    kuadrat_selisih = selisih ** 2

    tabel_manual = pd.DataFrame(
        {
            "Harga Aktual (€)": [format_euro(v) for v in actual_sample],
            "Harga Prediksi (€)": [format_euro(v) for v in pred_sample],
            "Selisih (yᵢ−ŷᵢ)": [format_euro(v) for v in selisih],
            "|Selisih|": [format_euro(v) for v in abs_selisih],
            "Selisih²": [format_euro(v) for v in kuadrat_selisih],
        }
    )
    st.dataframe(tabel_manual, width="stretch", hide_index=True)

    mae_manual = float(abs_selisih.mean())
    rmse_manual = float(np.sqrt(kuadrat_selisih.mean()))

    calc_col1, calc_col2 = st.columns(2)
    with calc_col1:
        st.metric("MAE dari sampel ini", f"€{format_euro(mae_manual)}")
        angka_abs = " + ".join(f"{v:,.0f}".replace(",", ".") for v in abs_selisih)
        st.caption(f"= ({angka_abs}) / {n_sample}")
    with calc_col2:
        st.metric("RMSE dari sampel ini", f"€{format_euro(rmse_manual)}")
        angka_kuadrat = " + ".join(f"{v:,.0f}".replace(",", ".") for v in kuadrat_selisih)
        st.caption(f"= √[({angka_kuadrat}) / {n_sample}]")

    st.divider()
    jumlah_test_fmt = f"{len(X_test):,}".replace(",", ".")
    st.markdown(
        f"**Perbandingan dengan MAE/RMSE resmi** (dihitung dari seluruh "
        f"{jumlah_test_fmt} data uji, bukan cuma {n_sample} sampel di atas):"
    )
    off_col1, off_col2 = st.columns(2)
    off_col1.metric("MAE (seluruh data uji)", f"€{format_euro(mae_full)}")
    off_col2.metric("RMSE (seluruh data uji)", f"€{format_euro(rmse_full)}")

    st.info(
        "MAE dan RMSE dari sampel kecil di atas wajar berbeda dari angka resmi "
        "di atas, karena angka resmi dihitung dari puluhan ribu data sekaligus. "
        "Coba naikkan jumlah sampel secara bertahap — semakin banyak sampel yang "
        "diambil, semakin dekat angkanya ke nilai resmi tersebut."
    )

# ------------------------------------------------------------------------------
# TAB 3 — EVALUASI & INTERPRETASI
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📋 Ringkasan Metrik pada Data Uji")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"€{format_euro(mae_full)}", help="Rata-rata kesalahan prediksi")
    m2.metric("RMSE", f"€{format_euro(rmse_full)}", help="Sensitivitas terhadap outlier harga")
    m3.metric(
        "R² Score", f"{r2_full:.4f}",
        help=f"Model menjelaskan {r2_full * 100:.1f}% variasi harga",
    )

    st.divider()
    st.subheader("📊 Feature Importance")

    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.barh(feat_imp.index, feat_imp.values, color=NAVY)
    ax1.set_xlabel("Tingkat Kepentingan Fitur (Feature Importance)")
    ax1.set_title("Feature Importance — Model XGBoost Prediksi Harga Mobil Bekas")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    feat_imp_desc = feat_imp.sort_values(ascending=False)
    top3 = feat_imp_desc.head(3)
    st.markdown(
        "**Interpretasi:** 3 fitur teratas — "
        + ", ".join(f"**{nama}** (~{nilai * 100:.0f}%)" for nama, nilai in top3.items())
        + f" — bersama-sama menyumbang sekitar {top3.sum() * 100:.0f}% dari total kepentingan "
        "fitur pada model ini. Persentase di atas dihitung langsung dari data yang sedang "
        "dimuat, sehingga bisa sedikit berbeda dari angka di laporan tertulis kalau dataset "
        "yang di-upload di sini berbeda cakupannya."
    )

    st.divider()
    st.subheader("📈 Perbandingan Harga Aktual vs Prediksi")

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_test, y_pred_test, alpha=0.15, s=10, color=NAVY)
    batas_maks = float(max(y_test.max(), y_pred_test.max()))
    ax2.plot([0, batas_maks], [0, batas_maks], color=CORAL, linestyle="--",
             linewidth=2, label="Prediksi Sempurna")
    ax2.set_xlabel("Harga Aktual (Euro)")
    ax2.set_ylabel("Harga Prediksi (Euro)")
    ax2.set_title("Perbandingan Harga Aktual vs Prediksi")
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown(
        "**Interpretasi:** pada rentang €0–€40.000 (mobil ekonomi hingga menengah), "
        "titik-titik data rapat mengikuti garis referensi, menandakan prediksi yang akurat "
        "pada segmen dengan data latih terbanyak. Pada rentang €40.000–€150.000 (mobil "
        "mewah), sebaran melebar dan model cenderung *underestimate*, karena data latih "
        "pada segmen premium jauh lebih sedikit."
    )