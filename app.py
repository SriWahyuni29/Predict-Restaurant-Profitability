import joblib
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
import re

st.set_page_config(page_title="Menu Profitability Predictor", page_icon="🍽️", layout="centered")

# =========================
# Loaders
# =========================
@st.cache_resource
def load_bundle():
    for p in ["UAS/model/menu_model.joblib", "model/menu_model.joblib", "menu_model.joblib"]:
        try:
            return joblib.load(p)
        except Exception:
            pass
    st.error("❌ Model bundle tidak ditemukan. Letakkan 'menu_model.joblib' di salah satu path: "
             "`UAS/model/`, `model/`, atau root project.")
    st.stop()

@st.cache_resource
def load_dataset():
    path = Path("/mnt/data/restaurant_menu_optimization_data.csv")
    if path.exists():
        df = pd.read_csv(path)
        needed = {"RestaurantID", "MenuCategory", "Price"}
        if not needed.issubset(df.columns):
            st.warning("Dataset ditemukan, tapi kolom wajib tidak lengkap: butuh RestaurantID, MenuCategory, Price.")
            return None
        return df
    return None

bundle = load_bundle()
model = bundle["model"]
scaler = bundle.get("scaler", None)
feature_columns = bundle["feature_columns"]
numerical_features = bundle.get("numerical_features", ["Price"])
class_names = bundle.get("class_names", {0: "Low", 1: "Medium", 2: "High"})

df_data = load_dataset()

# =========================
# Helpers
# =========================
def _norm(s: str) -> str:
    # normalisasi label: trim, rapikan spasi, casefold
    s = re.sub(r"\s+", " ", str(s).strip())
    return s.casefold()

def build_category_lookup(prefix="MenuCategory_"):
    """
    Menghasilkan:
      - options: daftar label kategori untuk dropdown (human-friendly)
      - label_to_col: mapping dari label dropdown -> nama kolom one-hot di model
    Menggabungkan kategori dari model (feature_columns) + data (R001-R003) dengan normalisasi
    sehingga tidak ada kategori yang hilang karena beda kapital/spasi.
    """
    # 1) Dari model
    model_cols = [c for c in feature_columns if c.startswith(prefix)]
    model_labels = [c[len(prefix):] for c in model_cols]
    model_norm_to_label = {_norm(l): l for l in model_labels}
    model_label_to_col = {l: f"{prefix}{l}" for l in model_labels}

    # 2) Dari data (dibatasi R001-R003)
    data_labels = set()
    if df_data is not None and "MenuCategory" in df_data.columns and "RestaurantID" in df_data.columns:
        allowed_ids = {"R001", "R002", "R003"}
        sub = df_data[df_data["RestaurantID"].astype(str).isin(allowed_ids)]
        data_labels = {re.sub(r"\s+", " ", str(x).strip()) for x in sub["MenuCategory"].dropna().unique()}

    # 3) Gabungkan dengan normalisasi (hanya ambil yang ada di model agar aman saat one-hot)
    final_labels = set(model_labels)
    for dl in data_labels:
        n = _norm(dl)
        if n in model_norm_to_label:
            final_labels.add(model_norm_to_label[n])

    # 4) Siapkan mapping ke kolom feature
    options = sorted(final_labels)
    label_to_col = {lbl: model_label_to_col[lbl] for lbl in options if lbl in model_label_to_col}
    return options, label_to_col

def build_row(price: float, category_label: str, label_to_col: dict) -> pd.DataFrame:
    row = {c: 0.0 for c in feature_columns}
    if "Price" in row:
        row["Price"] = float(price)
    # aktifkan one-hot sesuai kolom model yang tepat
    cat_col = label_to_col.get(category_label)
    if cat_col and cat_col in row:
        row[cat_col] = 1.0

    X = pd.DataFrame([row], columns=feature_columns)

    # scaling aman
    if scaler is not None and numerical_features:
        cols_to_scale = [c for c in numerical_features if c in X.columns]
        if cols_to_scale:
            X.loc[:, cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X

# =========================
# UI
# =========================
st.title("🍽️ Menu Profitability Predictor")
st.caption("Masukkan **Restaurant ID**, **Price**, dan **Menu Category** untuk memprediksi tingkat **Profitability**.")

with st.container():
    c1, c2, c3 = st.columns([1.2, 1, 1.2])

    # Restaurant ID otomatis dari data, dibatasi ke R001–R003
    allowed_ids = ["R001", "R002", "R003"]
    if df_data is not None:
        rid_list = sorted([str(rid) for rid in df_data["RestaurantID"].dropna().unique() if rid in allowed_ids])
        if not rid_list:
            rid_list = allowed_ids
    else:
        rid_list = allowed_ids
    restaurant_id = c1.selectbox("Restaurant ID", options=rid_list, index=0)

    # Prefill dari data sesuai RestaurantID
    if df_data is not None:
        pre = df_data[df_data["RestaurantID"] == restaurant_id].head(1)
        default_price = float(pre["Price"].iloc[0]) if not pre.empty else 18.50
        default_cat_raw = str(pre["MenuCategory"].iloc[0]) if not pre.empty else None
    else:
        default_price, default_cat_raw = 18.50, None

    # Build opsi kategori + mapping aman ke kolom model
    cat_opts, label_to_col = build_category_lookup()

    # Sinkronkan default dari data ke opsi (normalisasi)
    if default_cat_raw is not None:
        n = _norm(default_cat_raw)
        # cari label di cat_opts yang norm-nya sama
        match = next((lbl for lbl in cat_opts if _norm(lbl) == n), None)
        default_idx = cat_opts.index(match) if match else 0
    else:
        default_idx = 0

    price = c2.number_input("Price", min_value=0.0, value=default_price, step=0.50, format="%.2f")
    category = c3.selectbox("Menu Category", options=cat_opts, index=default_idx)

    predict_btn = st.button("Predict", type="primary", use_container_width=True)

# =========================
# Predict & Output
# =========================
if predict_btn:
    X = build_row(price=price, category_label=category, label_to_col=label_to_col)
    y = int(model.predict(X)[0])
    pred_label = class_names.get(y, str(y))

    st.success("Prediction complete.")
    st.metric(label="Predicted Profitability", value=pred_label)

    st.dataframe(pd.DataFrame([{
        "Restaurant ID": restaurant_id,
        "Price": price,
        "Menu Category": category,
        "Predicted Profitability": pred_label
    }]), use_container_width=True)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        proba_df = pd.DataFrame({
            "Class": [class_names.get(i, str(i)) for i in range(len(proba))],
            "Probability": np.round(proba, 4)
        })
        st.caption("Class probabilities")
        st.dataframe(proba_df, use_container_width=True)
