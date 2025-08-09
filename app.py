import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Menu Profitability Predictor", page_icon="🍽️")

@st.cache_resource
def load_bundle():
    for p in ["UAS/model/menu_model.joblib", "model/menu_model.joblib", "menu_model.joblib"]:
        try:
            return joblib.load(p)
        except Exception:
            pass
    st.stop()

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]
numerical_features = bundle.get("numerical_features", ["Price","Ingredients_count"])
class_names = bundle.get("class_names", {0:"Low",1:"Medium",2:"High"})

st.title("🍽️ Menu Profitability Predictor")

def build_row(price, ing_count, cat, item):
    row = {c:0.0 for c in feature_columns}
    if "Price" in row: row["Price"] = float(price)
    if f"MenuCategory_{cat}" in row: row[f"MenuCategory_{cat}"] = 1.0
    X = pd.DataFrame([row], columns=feature_columns)
    if numerical_features: X[numerical_features] = scaler.transform(X[numerical_features])
    return X

def opts(prefix):
    return sorted([c.split(prefix,1)[1] for c in feature_columns if c.startswith(prefix)])

col1, col2 = st.columns(2)
price = col1.number_input("Price", 0.0, value=18.5, step=0.5, format="%.2f")
cat = st.selectbox("Menu category", opts("MenuCategory_") or ["Main Course"])

if st.button("Predict"):
    X = build_row(price, ing, cat, item)
    y  = int(model.predict(X)[0])
    st.metric("Predicted profitability", class_names.get(y, str(y)))
    if hasattr(model, "predict_proba"):
        import numpy as np
        proba = model.predict_proba(X)[0]
        st.dataframe(pd.DataFrame({"Class":[class_names.get(i,str(i)) for i in range(len(proba))],
                                   "Probability":np.round(proba,4)}), use_container_width=True)
