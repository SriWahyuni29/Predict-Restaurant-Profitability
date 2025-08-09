# train_dt.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("/mnt/data/restaurant_menu_optimization_data.csv")

# Pastikan kolom
# RestaurantID (hanya untuk tampilan, biasanya TIDAK dipakai fitur),
# MenuCategory (kategorikal), Price (numerik), Profitability (target)
X = df[["MenuCategory", "Price"]].copy()
y = df["Profitability"].astype("category")

# Definisikan 4 kategori yang harus ada (sesuai data kamu)
menu_categories = ["Appetizers", "Beverages", "Desserts", "Main Course"]

# Preprocessor
numeric_features = ["Price"]
categorical_features = ["MenuCategory"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), numeric_features),
        ("cat", OneHotEncoder(categories=[menu_categories], handle_unknown="ignore", sparse_output=False), categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Model
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=100,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt"
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", dt)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
print("\nDecision Tree Model Evaluation:")
print(classification_report(y_test, y_pred))

# Siapkan feature_columns agar app.py tahu urutan kolom:
#    - numeric (Price) + one-hot dari kategori di atas
ohe = pipe.named_steps["prep"].named_transformers_["cat"]
ohe_names = [f"MenuCategory_{c}" for c in menu_categories]  # pakai urutan yang ditetapkan
feature_columns = numeric_features + ohe_names

# class_names mapping (urut sesuai kelas unik y)
classes = list(y.cat.categories)
class_names = {i: cls for i, cls in enumerate(classes)}  # kalau model output index, tidak wajib dipakai

# Simpan bundle (model disimpan sebagai pipeline utuh)
bundle = {
    "model": pipe,                       # Pipeline lengkap (prep + clf)
    "scaler": None,                      # Tidak perlu karena scaler ada di Pipeline
    "feature_columns": feature_columns,  # Agar app bisa bangun vektor fitur
    "numerical_features": numeric_features,
    "class_names": class_names
}

Path("model").mkdir(exist_ok=True)
joblib.dump(bundle, "model/menu_model.joblib")
print("✅ Saved to model/menu_model.joblib")
