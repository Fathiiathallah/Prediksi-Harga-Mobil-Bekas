"""
===============================================================================
DASHBOARD INTERAKTIF — KLASIFIKASI KATEGORI HARGA MOBIL BEKAS
Studi Kasus Bagian 5 (versi klasifikasi): Murah / Menengah / Mahal
Dibandingkan lewat 3 algoritma boosting: AdaBoost, Gradient Boosting, XGBoost
Dataset : Used Cars Database — eBay Kleinanzeigen (Jerman)
===============================================================================
Cara menjalankan:
    1) pip install -r requirements.txt
    2) streamlit run app.py
Taruh file 'autos_clean.csv' di folder yang sama dengan app.py ini, atau
upload manual lewat panel di sidebar sebelah kiri.
===============================================================================
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# =============================================================================
# 0. KONFIGURASI HALAMAN & TEMA VISUAL
# =============================================================================
st.set_page_config(
    page_title="Klasifikasi Harga Mobil Bekas — Boosting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#0E5A6B"
ACCENT = "#C9862B"
BG = "#F6F4EF"
CARD = "#FFFFFF"
TEXT = "#20242A"
TEXT_MUTED = "#5B6169"
BORDER = "#E3DFD5"
DANGER = "#B3492E"

FILE_PATH_DEFAULT = "autos_clean.csv"

KOLOM_KATEGORIKAL_MISSING = ["vehicleType", "gearbox", "model", "fuelType", "notRepairedDamage"]
KOLOM_DIBUANG = [
    "dateCrawled", "name", "seller", "offerType", "abtest",
    "nrOfPictures", "dateCreated", "lastSeen", "yearOfRegistration",
]
KOLOM_KATEGORIKAL = ["vehicleType", "gearbox", "model", "fuelType", "brand", "notRepairedDamage"]

LABEL_FITUR = {
    "vehicleType": "Tipe kendaraan", "gearbox": "Transmisi", "powerPS": "Tenaga mesin (PS)",
    "model": "Model kendaraan", "kilometer": "Jarak tempuh (km)", "monthOfRegistration": "Bulan registrasi",
    "fuelType": "Jenis bahan bakar", "brand": "Merek", "notRepairedDamage": "Kerusakan belum diperbaiki",
    "postalCode": "Kode pos", "vehicleAge": "Umur kendaraan (tahun)",
}

KATEGORI_ORDER = ["Murah", "Menengah", "Mahal"]
KATEGORI_COLOR = {"Murah": "#7FB3BD", "Menengah": "#C9862B", "Mahal": "#0E5A6B"}
MODEL_ORDER = ["AdaBoost", "Gradient Boosting", "XGBoost"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ visibility: hidden; }}

h1, h2, h3 {{ font-family: 'Inter', sans-serif; font-weight: 700; }}

.hero {{
    display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
    padding: 0 0 14px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 20px;
}}
.hero .eyebrow {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing: 0.1em;
    text-transform: uppercase; background: {PRIMARY}; color: #FFFFFF;
    padding: 4px 10px; border-radius: 999px; font-weight: 600; white-space: nowrap;
}}
.hero h1 {{ margin: 0; font-size: 24px; }}
.hero p {{ color: {TEXT_MUTED}; font-size: 14px; margin: 4px 0 0 0; width: 100%; max-width: 760px; }}

.metric-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 18px 20px; height: 100%;
}}
.metric-card .m-label {{
    font-size: 12.5px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 8px;
}}
.metric-card .m-value {{
    font-family: 'IBM Plex Mono', monospace; font-size: 30px; font-weight: 600; color: {TEXT}; line-height: 1.15;
}}
.metric-card .m-sub {{ font-size: 12.5px; color: {TEXT_MUTED}; margin-top: 6px; }}

.readout {{ background: {TEXT}; border-radius: 12px; padding: 28px 32px; text-align: center; }}
.readout .r-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; letter-spacing: 0.1em;
    text-transform: uppercase; color: {ACCENT}; margin-bottom: 10px;
}}
.readout .r-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 46px; font-weight: 600; color: #FFFFFF; line-height: 1; }}
.readout .r-sub {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; color: #C7CDD3; margin-top: 12px; }}

.info-box {{ background: #FBF4E8; border: 1px solid #ECD9B0; border-radius: 10px; padding: 14px 18px; font-size: 14px; color: {TEXT}; }}

.section-head {{ margin: 6px 0 16px 0; }}
.section-head .s-eyebrow {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.1em;
    text-transform: uppercase; color: {PRIMARY}; font-weight: 600;
}}
.section-head h2 {{ margin: 2px 0 0 0; font-size: 20px; }}

.kpi-strip {{ background: {TEXT}; border-radius: 12px; padding: 4px; margin-bottom: 26px; display: flex; flex-wrap: wrap; }}
.kpi-item {{ flex: 1; min-width: 150px; padding: 20px 24px; border-right: 1px solid rgba(255,255,255,0.12); }}
.kpi-item:last-child {{ border-right: none; }}
.kpi-item .k-label {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.08em;
    text-transform: uppercase; color: {ACCENT}; margin-bottom: 8px;
}}
.kpi-item .k-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 27px; font-weight: 600; color: #FFFFFF; line-height: 1.1; }}
.kpi-item .k-sub {{ font-size: 12px; color: #9BA3AC; margin-top: 4px; }}

.predict-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 18px 20px; height: 100%; }}
.predict-card .pc-model {{
    font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.06em;
    text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 10px;
}}
.predict-card .pc-result {{ font-family: 'IBM Plex Mono', monospace; font-size: 23px; font-weight: 600; margin-bottom: 14px; }}
.predict-card .bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; color: {TEXT_MUTED}; }}
.predict-card .bar-name {{ width: 72px; flex-shrink: 0; }}
.predict-card .bar-track {{ flex: 1; height: 7px; background: #EEEAE0; border-radius: 4px; overflow: hidden; }}
.predict-card .bar-fill {{ height: 100%; border-radius: 4px; }}
.predict-card .bar-pct {{ width: 34px; flex-shrink: 0; text-align: right; font-family: 'IBM Plex Mono', monospace; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Inter', sans-serif; font-weight: 600; font-size: 14.5px; padding: 10px 18px; }}

.stButton > button, .stFormSubmitButton > button {{
    background-color: {PRIMARY}; color: white; border-radius: 8px; border: none; font-weight: 600;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ background-color: #0A4756; color: white; }}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    st.markdown(f"""<div class="metric-card"><div class="m-label">{label}</div>
    <div class="m-value">{value}</div><div class="m-sub">{sub}</div></div>""", unsafe_allow_html=True)


def kpi_strip(items):
    chips = "".join(f"""<div class="kpi-item"><div class="k-label">{l}</div>
        <div class="k-value">{v}</div><div class="k-sub">{s}</div></div>""" for l, v, s in items)
    st.markdown(f'<div class="kpi-strip">{chips}</div>', unsafe_allow_html=True)


def section_header(eyebrow, title):
    st.markdown(f"""<div class="section-head"><div class="s-eyebrow">{eyebrow}</div>
    <h2>{title}</h2></div>""", unsafe_allow_html=True)


def format_eur(value, decimals=0):
    s = f"{value:,.{decimals}f}"
    s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    return f"€{s}"


# =============================================================================
# 1. PEMUATAN & PEMBERSIHAN DATA (cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_and_clean(file_source):
    df = pd.read_csv(file_source)
    baris_awal = len(df)
    tahun_referensi = int(pd.to_datetime(df["dateCrawled"]).dt.year.max())
    df = df[
        df["price"].between(100, 150_000)
        & df["yearOfRegistration"].between(1950, tahun_referensi)
        & df["powerPS"].between(1, 1000)
    ].copy()
    baris_bersih = len(df)
    jumlah_terisi = {}
    for col in KOLOM_KATEGORIKAL_MISSING:
        jumlah_terisi[col] = int(df[col].isnull().sum())
        df[col] = df[col].fillna("tidak_diketahui")
    df["vehicleAge"] = tahun_referensi - df["yearOfRegistration"]
    df = df.drop(columns=KOLOM_DIBUANG)
    meta = {"baris_awal": baris_awal, "baris_bersih": baris_bersih,
            "tahun_referensi": tahun_referensi, "jumlah_terisi": jumlah_terisi}
    return df, meta


@st.cache_data(show_spinner=False)
def add_price_category(df_clean, low_pct, high_pct):
    df = df_clean.copy()
    batas_bawah = float(df["price"].quantile(low_pct / 100))
    batas_atas = float(df["price"].quantile(high_pct / 100))
    if batas_bawah >= batas_atas:
        batas_atas = batas_bawah + 1
    df["price_category"] = pd.cut(
        df["price"], bins=[-np.inf, batas_bawah, batas_atas, np.inf], labels=KATEGORI_ORDER
    ).astype(str)
    return df, batas_bawah, batas_atas


# =============================================================================
# 2. ENCODING & TRAINING 3 MODEL BOOSTING (cached sebagai resource)
# =============================================================================
@st.cache_resource(show_spinner=False)
def encode_and_train_classifiers(df_with_cat, n_estimators):
    df_enc = df_with_cat.drop(columns=["price"]).copy()
    encoders = {}
    for col in KOLOM_KATEGORIKAL:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    cat_encoder = LabelEncoder()
    cat_encoder.fit(KATEGORI_ORDER)
    y = cat_encoder.transform(df_enc["price_category"].astype(str))
    X = df_enc.drop(columns=["price_category"])
    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_specs = {
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=n_estimators, learning_rate=0.5, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_estimators, learning_rate=0.1, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1,
        ),
    }

    results = {}
    for name, clf in model_specs.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[name] = {
            "model": clf,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "feature_importance": pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False),
        }

    return {
        "results": results, "encoders": encoders, "cat_encoder": cat_encoder,
        "feature_cols": feature_cols, "n_train": len(X_train), "n_test": len(X_test),
    }


# =============================================================================
# 3. SIDEBAR — SUMBER DATA & PENGATURAN MODEL
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="background:{PRIMARY};border-radius:8px;padding:10px 12px;margin-bottom:16px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.08em;
        color:#BFE3EA;font-weight:600;text-transform:uppercase;">Bagian 5 — Studi Kasus</div>
        <div style="font-size:16px;font-weight:700;color:#FFFFFF;">Klasifikasi Harga Mobil Bekas</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"], help="Format sama seperti Used Cars Database eBay Kleinanzeigen")

    data_source, source_id = None, None
    if uploaded is not None:
        data_source = uploaded
        source_id = f"upload:{uploaded.name}:{uploaded.size}"
    elif os.path.exists(FILE_PATH_DEFAULT):
        data_source = FILE_PATH_DEFAULT
        source_id = f"default:{FILE_PATH_DEFAULT}"
        st.caption(f"📄 Memakai `{FILE_PATH_DEFAULT}` dari folder ini")

    st.divider()
    st.markdown("**Pengaturan kategori & model**")
    with st.form("training_form"):
        pct_range = st.slider(
            "Batas persentil Murah / Menengah / Mahal", 5, 95, (33, 67), step=1,
            help="Mis. (33,67) berarti 33% termurah = Murah, 33% teratas = Mahal, sisanya Menengah",
        )
        n_estimators = st.slider("n_estimators (jumlah pohon, berlaku utk ketiga model)", 50, 300, 150, step=10)
        submitted = st.form_submit_button("🚀 Latih 3 Model", use_container_width=True)

    st.divider()
    st.caption("Dataset: *Used Cars Database*, eBay Kleinanzeigen (Jerman), via Kaggle.")

# =============================================================================
# 4. MEMUAT DATA & (RE)TRAINING BERDASARKAN STATE
# =============================================================================
if data_source is not None:
    df_clean, meta = load_and_clean(data_source)
    df_cat, batas_bawah, batas_atas = add_price_category(df_clean, pct_range[0], pct_range[1])
    st.session_state["df_clean"] = df_clean
    st.session_state["df_cat"] = df_cat
    st.session_state["batas"] = (batas_bawah, batas_atas)
    st.session_state["meta"] = meta

    train_key = f"{source_id}|{pct_range[0]}-{pct_range[1]}|{n_estimators}"
    should_train = submitted or "results" not in st.session_state or st.session_state.get("trained_on") != train_key
    if should_train:
        with st.spinner("Melatih AdaBoost, Gradient Boosting, dan XGBoost... (bisa memakan waktu beberapa puluh detik)"):
            st.session_state["results"] = encode_and_train_classifiers(df_cat, n_estimators)
            st.session_state["trained_on"] = train_key

# =============================================================================
# 5. HEADER
# =============================================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AdaBoost · Gradient Boosting · XGBoost</div>
    <h1>Klasifikasi Kategori Harga Mobil Bekas</h1>
    <p>Membandingkan tiga algoritma boosting untuk mengklasifikasikan mobil bekas ke dalam tiga
    kategori harga — Murah, Menengah, Mahal — berdasarkan spesifikasi dan kondisi kendaraan.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 6. TAMPILAN UTAMA
# =============================================================================
if "df_cat" not in st.session_state or "results" not in st.session_state:
    st.markdown(f"""
    <div class="info-box"><b>Belum ada data dimuat.</b> Upload file CSV lewat sidebar di kiri, atau taruh file
    <code>{FILE_PATH_DEFAULT}</code> di folder yang sama dengan <code>app.py</code>, lalu klik
    <b>🚀 Latih 3 Model</b>.</div>
    """, unsafe_allow_html=True)
    st.stop()

df_clean = st.session_state["df_clean"]
df_cat = st.session_state["df_cat"]
batas_bawah, batas_atas = st.session_state["batas"]
meta = st.session_state["meta"]
res = st.session_state["results"]
model_results = res["results"]

best_model = max(model_results, key=lambda m: model_results[m]["accuracy"])
avg_f1 = float(np.mean([model_results[m]["f1"] for m in model_results]))

kpi_strip([
    ("Data latih", f"{res['n_train']:,}".replace(",", "."), f"+{res['n_test']:,} data uji".replace(",", ".")),
    ("Kategori harga", "3 kelas", "Murah · Menengah · Mahal"),
    ("Model terbaik", best_model, f"akurasi {model_results[best_model]['accuracy']*100:.1f}%"),
    ("Rata-rata F1", f"{avg_f1:.3f}", "ketiga model, weighted"),
])

# ------------------------------------------------------------- SECTION 01 ---
section_header("01 · Data", "Eksplorasi Data")

persen_terbuang = (meta["baris_awal"] - meta["baris_bersih"]) / meta["baris_awal"] * 100
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Baris awal", f"{meta['baris_awal']:,}".replace(",", "."))
with c2:
    metric_card("Setelah cleaning", f"{meta['baris_bersih']:,}".replace(",", "."),
                 f"{100 - persen_terbuang:.1f}% dari data awal")
with c3:
    metric_card("Tahun referensi", str(meta["tahun_referensi"]), "acuan hitung umur kendaraan")
with c4:
    metric_card("Jumlah fitur", str(len(res["feature_cols"])), "harga TIDAK dipakai sbg fitur")

st.write("")
st.markdown(f"""
<div class="info-box">
📏 Batas kategori (dari persentil di sidebar): <b>Murah</b> = harga di bawah {format_eur(batas_bawah)} ·
<b>Menengah</b> = {format_eur(batas_bawah)} – {format_eur(batas_atas)} · <b>Mahal</b> = di atas {format_eur(batas_atas)}.
Geser slider "Batas persentil" di sidebar buat ubah definisinya.
</div>
""", unsafe_allow_html=True)
st.write("")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Distribusi harga & batas kategori**")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df_clean["price"], nbinsx=50, marker_color=PRIMARY, opacity=0.85))
    fig_hist.add_vline(x=batas_bawah, line_dash="dash", line_color=DANGER, line_width=2)
    fig_hist.add_vline(x=batas_atas, line_dash="dash", line_color=DANGER, line_width=2)
    fig_hist.update_layout(
        plot_bgcolor="black", paper_bgcolor="black",
        xaxis_title="Harga (€)", yaxis_title="Jumlah iklan",
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        font=dict(family="Inter, sans-serif", color="#FFFFFF"), showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    st.markdown("**Jumlah iklan per kategori harga**")
    counts = df_cat["price_category"].value_counts().reindex(KATEGORI_ORDER)
    fig_cat = go.Figure(go.Bar(
        x=KATEGORI_ORDER, y=counts.values,
        marker_color=[KATEGORI_COLOR[k] for k in KATEGORI_ORDER],
        text=[f"{v:,}".replace(",", ".") for v in counts.values], textposition="outside",
    ))
    fig_cat.update_layout(
        plot_bgcolor="black", paper_bgcolor="black",
        xaxis_title="", yaxis_title="Jumlah iklan",
        margin=dict(l=10, r=10, t=20, b=10), height=320,
        font=dict(family="Inter, sans-serif", color="#FFFFFF"),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with st.expander("Lihat penanganan missing value, distribusi merek & contoh data bersih"):
    st.write("Kolom kategorikal yang kosong diisi dengan kategori `'tidak_diketahui'`:")
    isian_df = pd.DataFrame({
        "Kolom": list(meta["jumlah_terisi"].keys()),
        "Jumlah nilai kosong yang diisi": list(meta["jumlah_terisi"].values()),
    })
    st.dataframe(isian_df, hide_index=True, use_container_width=True)

    top_brand = df_clean["brand"].value_counts().head(10).sort_values()
    fig_brand = go.Figure(go.Bar(x=top_brand.values, y=top_brand.index, orientation="h", marker_color=ACCENT))
    fig_brand.update_layout(plot_bgcolor="black", paper_bgcolor="black", xaxis_title="Jumlah iklan",
                             margin=dict(l=10, r=10, t=10, b=10), height=320,
                             font=dict(family="Inter, sans-serif", color="#FFFFFF"))
    st.write("10 merek dengan iklan terbanyak:")
    st.plotly_chart(fig_brand, use_container_width=True)

    st.write("Contoh 10 baris data setelah dibersihkan (dengan kategori harga):")
    st.dataframe(df_cat.head(10), use_container_width=True)

st.divider()

# ------------------------------------------------------------- SECTION 02 ---
section_header("02 · Evaluasi", "Performa 3 Model Boosting")

st.markdown("**Perbandingan akurasi & F1-Score**")
accs = [model_results[m]["accuracy"] for m in MODEL_ORDER]
f1s = [model_results[m]["f1"] for m in MODEL_ORDER]
fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(name="Akurasi", x=MODEL_ORDER, y=accs, marker_color=PRIMARY,
                          text=[f"{v:.1%}" for v in accs], textposition="outside"))
fig_cmp.add_trace(go.Bar(name="F1-Score (weighted)", x=MODEL_ORDER, y=f1s, marker_color=ACCENT,
                          text=[f"{v:.1%}" for v in f1s], textposition="outside"))
fig_cmp.update_layout(
    barmode="group", plot_bgcolor="black", paper_bgcolor="black",
    yaxis_tickformat=".0%", yaxis_range=[0, 1.08],
    margin=dict(l=10, r=10, t=10, b=10), height=340,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    font=dict(family="Inter, sans-serif", color="#FFFFFF"),
)
st.plotly_chart(fig_cmp, use_container_width=True)

st.write("")
model_pilihan = st.selectbox("Lihat detail model:", MODEL_ORDER, index=MODEL_ORDER.index(best_model))
detail = model_results[model_pilihan]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(f"**Confusion matrix — {model_pilihan}**")
    cm = detail["confusion_matrix"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=KATEGORI_ORDER, y=KATEGORI_ORDER,
        colorscale=[[0, "#FFFFFF"], [1, PRIMARY]],
        text=cm, texttemplate="%{text}", textfont=dict(size=15, color=TEXT),
        showscale=False, xgap=3, ygap=3,
    ))
    fig_cm.update_layout(
        xaxis_title="Prediksi", yaxis_title="Aktual", yaxis_autorange="reversed",
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        font=dict(family="Inter, sans-serif", color=TEXT),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_b:
    st.markdown(f"**Feature importance — {model_pilihan}**")
    fi = detail["feature_importance"].rename(index=LABEL_FITUR)
    fig_fi = go.Figure(go.Bar(x=fi.values[::-1], y=fi.index[::-1], orientation="h", marker_color=PRIMARY))
    fig_fi.update_layout(
        plot_bgcolor="black", paper_bgcolor="black",
        xaxis_title="Tingkat kepentingan fitur", yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        font=dict(family="Inter, sans-serif", color="#FFFFFF"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown(f"""
<div class="info-box">
💡 Diagonal confusion matrix (kiri atas ke kanan bawah) menunjukkan prediksi yang benar. Kesalahan yang
paling sering terjadi biasanya antara kelas <b>bertetangga</b> (mis. Menengah tertukar Mahal), bukan antar
kelas yang jauh (Murah tertukar Mahal) — wajar karena batas harga antar kategori tetangga memang lebih tipis.
</div>
""", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------- SECTION 03 ---
section_header("03 · Interaktif", "Coba Klasifikasikan Mobil Sendiri")
st.markdown("Masukkan spesifikasi mobil di bawah — ketiga model akan menebak kategori harganya sekaligus, biar bisa dibandingkan.")
st.write("")

encoders = res["encoders"]
cat_encoder = res["cat_encoder"]
tahun_ref = meta["tahun_referensi"]

with st.form("predict_form"):
    colL, colM, colR = st.columns(3)

    with colL:
        brand_opts = sorted(encoders["brand"].classes_)
        brand_val = st.selectbox("Merek", brand_opts, index=brand_opts.index("volkswagen") if "volkswagen" in brand_opts else 0)

        model_opts_all = df_clean.loc[df_clean["brand"] == brand_val, "model"].unique().tolist()
        model_opts_all = sorted(model_opts_all) if model_opts_all else sorted(encoders["model"].classes_)
        model_val = st.selectbox("Model kendaraan", model_opts_all)

        vt_opts = sorted(encoders["vehicleType"].classes_)
        vehicle_type_val = st.selectbox("Tipe kendaraan", vt_opts)

    with colM:
        gearbox_opts = sorted(encoders["gearbox"].classes_)
        gearbox_val = st.selectbox("Transmisi", gearbox_opts)

        fuel_opts = sorted(encoders["fuelType"].classes_)
        fuel_val = st.selectbox("Jenis bahan bakar", fuel_opts)

        dmg_opts = sorted(encoders["notRepairedDamage"].classes_)
        dmg_val = st.selectbox("Kerusakan belum diperbaiki", dmg_opts)

    with colR:
        min_power = int(df_clean["powerPS"].quantile(0.02))
        max_power = int(df_clean["powerPS"].quantile(0.98))
        power_val = st.slider("Tenaga mesin (PS)", min_power, max_power, int(df_clean["powerPS"].median()))

        min_km, max_km = int(df_clean["kilometer"].min()), int(df_clean["kilometer"].max())
        km_val = st.slider("Jarak tempuh (km)", min_km, max_km, int(df_clean["kilometer"].median()), step=5000)

        default_tahun = min(2010, tahun_ref)
        tahun_val = st.slider("Tahun pembuatan", 1950, tahun_ref, default_tahun)

    month_val = st.slider("Bulan registrasi", 1, 12, 6)
    default_postal = int(df_clean["postalCode"].median())

    predict_clicked = st.form_submit_button("🔮 Klasifikasikan", use_container_width=True)

if predict_clicked:
    input_dict = {
        "vehicleType": vehicle_type_val, "gearbox": gearbox_val, "powerPS": power_val,
        "model": model_val, "kilometer": km_val, "monthOfRegistration": month_val,
        "fuelType": fuel_val, "brand": brand_val, "notRepairedDamage": dmg_val,
        "postalCode": default_postal, "vehicleAge": tahun_ref - tahun_val,
    }
    row = {}
    for col in res["feature_cols"]:
        if col in encoders:
            val = input_dict[col]
            le = encoders[col]
            row[col] = int(le.transform([val])[0]) if val in le.classes_ else -1
        else:
            row[col] = input_dict[col]
    row_df = pd.DataFrame([row])[res["feature_cols"]]

    st.write("")
    preds = {}
    cols = st.columns(3)
    for i, name in enumerate(MODEL_ORDER):
        clf = model_results[name]["model"]
        pred_idx = int(clf.predict(row_df)[0])
        proba = clf.predict_proba(row_df)[0]
        label = cat_encoder.inverse_transform([pred_idx])[0]
        preds[name] = label

        bars = "".join(f"""
        <div class="bar-row">
            <div class="bar-name">{k}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{p*100:.0f}%;background:{KATEGORI_COLOR[k]};"></div></div>
            <div class="bar-pct">{p*100:.0f}%</div>
        </div>""" for k, p in zip(KATEGORI_ORDER, proba))

        with cols[i]:
            st.markdown(f"""
            <div class="predict-card">
                <div class="pc-model">{name}</div>
                <div class="pc-result" style="color:{KATEGORI_COLOR[label]};">{label}</div>
                {bars}
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    if len(set(preds.values())) == 1:
        st.markdown(f"""<div class="info-box">✅ Ketiga model <b>sepakat</b>: mobil ini masuk kategori
        <b style="color:{KATEGORI_COLOR[list(preds.values())[0]]};">{list(preds.values())[0]}</b>.</div>""",
                    unsafe_allow_html=True)
    else:
        ringkas = " · ".join(f"{k}: {v}" for k, v in preds.items())
        st.markdown(f"""<div class="info-box">🤔 Model <b>berbeda pendapat</b> di kasus ini — {ringkas}.
        Biasanya terjadi kalau spesifikasi mobilnya pas di sekitar garis batas dua kategori.</div>""",
                    unsafe_allow_html=True)
