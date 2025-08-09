import joblib
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Menu Profitability Predictor", page_icon="🍽️")

@st.cache_resource
def load_bundle():
    # Try several common paths
    for p in ["UAS/model/menu_model.joblib", "model/menu_model.joblib", "menu_model.joblib"]:
        try:
            return joblib.load(p)
        except Exception:
            pass
    st.error("Model bundle not found. Make sure menu_model.joblib is in one of the expected paths.")
    st.stop()

bundle = load_bundle()
model = bundle["model"]
scaler = bundle.get("scaler", None)
feature_columns = bundle["feature_columns"]
numerical_features = bundle.get("numerical_features", ["Price"])  # keep only what you actually trained
class_names = bundle.get("class_names", {0: "Low", 1: "Medium", 2: "High"})

st.title("🍽️ Menu Profitability Predictor")

def category_options(prefix="MenuCategory_"):
    return sorted([c.split(prefix, 1)[1] for c in feature_columns if c.startswith(prefix)]) or ["Main Course"]

def build_row(price: float, category: str) -> pd.DataFrame:
    # start with zeroed vector for all features your model expects
    row = {c: 0.0 for c in feature_columns}

    # numeric feature(s)
    if "Price" in row:
        row["Price"] = float(price)

    # one-hot category if present in training columns
    cat_col = f"MenuCategory_{category}"
    if cat_col in row:
        row[cat_col] = 1.0

    X = pd.DataFrame([row], columns=feature_columns)

    # scale only if scaler exists *and* there are numeric columns to scale
    if scaler is not None and numerical_features:
        cols_to_scale = [c for c in numerical_features if c in X.columns]
        if cols_to_scale:
            X.loc[:, cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X

# --- UI ---
c1, c2, c3 = st.columns([1, 1, 1.2])
restaurant_id = c1.text_input("Restaurant ID", value="R-001")
price = c2.number_input("Price", min_value=0.0, value=18.50, step=0.50, format="%.2f")
category = c3.selectbox("Menu Category", category_options())

if st.button("Predict"):
    X = build_row(price=price, category=category)
    y = int(model.predict(X)[0])
    pred_label = class_names.get(y, str(y))

    st.subheader("Result")
    st.metric("Predicted Profitability", pred_label)

    # show the inputs alongside the prediction for clarity
    st.write("**Your input & prediction**")
    st.dataframe(
        pd.DataFrame(
            [{
                "Restaurant ID": restaurant_id,
                "Price": price,
                "Menu Category": category,
                "Predicted Profitability": pred_label
            }]
        ),
        use_container_width=True
    )

    # (Optional) class probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        proba_df = pd.DataFrame({
            "Class": [class_names.get(i, str(i)) for i in range(len(proba))],
            "Probability": np.round(proba, 4)
        })
        st.write("**Class probabilities**")
        st.dataframe(proba_df, use_container_width=True)
