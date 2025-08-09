import joblib
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path

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
def category_options(prefix="MenuCategory_"):
    # Paksa 4 kategori utama selalu tampil
    return ["Appetizers", "Beverages", "Desserts", "Main Course"]

def build_row(price: float, category: str) -> pd.DataFrame:
    row = {c: 0.0 for c in feature_columns}
    if "Price" in row:
        row["Price"] = float(price)
    # aktifkan one-hot hanya jika kolomnya memang ada di model
    cat_col = f"MenuCategory_{category}"
    if cat_col in row:
        row[cat_col] = 1.0

    X = pd.DataFrame([row], columns=feature_columns)

    # scaling aman jika scaler & kolom numerik tersedia
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

    # Restaurant ID otomatis dari data, dibatasi ke tiga ID
    allowed_ids = ["R001", "R002", "R003"]
    if df_data is not None:
        rid_list = sorted([str(rid) for rid in df_data["RestaurantID"].dropna().unique() if rid in allowed_ids]) or allowed_ids
    else:
        rid_list = allowed_ids
    restaurant_id = c1.selectbox("Restaurant ID", options=rid_list, index=0)

    # Prefill price & category dari baris pertama dengan RestaurantID terpilih
    if df_data is not None:
        pre = df_data[df_data["RestaurantID"] == restaurant_id].head(1)
        default_price = float(pre["Price"].iloc[0]) if not pre.empty else 18.50
        default_cat = str(pre["MenuCategory"].iloc[0]) if not pre.empty else "Main Course"
    else:
        default_price, default_cat = 18.50, "Main Course"

    price = c2.number_input("Price", min_value=0.0, value=default_price, step=0.50, format="%.2f")

    cat_opts = category_options()
    default_idx = cat_opts.index(default_cat) if (default_cat in cat_opts) else 0
    category = c3.selectbox("Menu Category", options=cat_opts, index=default_idx)

    predict_btn = st.button("Predict", type="primary", use_container_width=True)

# =========================
# Predict & Output
# =========================
if predict_btn:
    X = build_row(price=price, category=category)
    y = model.predict(X)[0]
    # jika model mengembalikan index int, map ke class_names; jika string, pakai langsung
    try:
        pred_label = class_names.get(int(y), str(y))
    except Exception:
        pred_label = str(y)

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
