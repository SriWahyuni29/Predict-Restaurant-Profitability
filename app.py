import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

APP_NAME = os.getenv("APP_NAME", "menu-profitability")

# Load model
bundle = joblib.load("UAS/model/menu_model.joblib")
model = bundle["model"]
scaler = bundle["scaler"]
numerical_features = bundle["numerical_features"]
feature_columns = bundle["feature_columns"]
class_names = bundle.get("class_names", {0: "Low", 1: "Medium", 2: "High"})

app = Flask(__name__)

def build_feature_row(price, ingredients_count, menu_category, menu_item):
    row = {col: 0.0 for col in feature_columns}
    if "Price" in row:
        row["Price"] = float(price)
    if "Ingredients_count" in row:
        row["Ingredients_count"] = float(ingredients_count)

    col_cat = f"MenuCategory_{menu_category}"
    if col_cat in row:
        row[col_cat] = 1.0

    col_item = f"MenuItem_{menu_item}"
    if col_item in row:
        row[col_item] = 1.0

    df = pd.DataFrame([row], columns=feature_columns)
    if numerical_features:
        df[numerical_features] = scaler.transform(df[numerical_features])
    return df

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "app": APP_NAME,
        "message": "Use POST /predict",
        "example_input": {
            "price": 18.5,
            "ingredients_count": 4,
            "menu_category": "Main Course",
            "menu_item": "Chicken Satay"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    required = ["price", "ingredients_count", "menu_category", "menu_item"]
    missing = [r for r in required if r not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    X = build_feature_row(
        data["price"],
        data["ingredients_count"],
        data["menu_category"],
        data["menu_item"]
    )

    pred = model.predict(X)[0]
    label = class_names.get(int(pred), str(pred))
    proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None

    return jsonify({
        "prediction_int": int(pred),
        "prediction_label": label,
        "probabilities": proba
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))