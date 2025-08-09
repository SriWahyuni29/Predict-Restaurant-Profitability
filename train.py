import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# === Load Data ===
data = pd.read_csv("UAS/data/restaurant_menu_optimization_data.csv")

# Hitung jumlah bahan di kolom Ingredients
data['Ingredients_count'] = data['Ingredients'].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)

# Hapus kolom yang tidak diperlukan
data = data.drop(columns=['Ingredients', 'RestaurantID'])

# One-Hot Encoding kategori
data_encoded = pd.get_dummies(data, columns=['MenuCategory', 'MenuItem'], drop_first=True)

# Label Encoding target
label_encoder = LabelEncoder()
data_encoded['Profitability'] = label_encoder.fit_transform(data_encoded['Profitability'])

# Isi missing values pada Price
data_encoded['Price'].fillna(data_encoded['Price'].mean(), inplace=True)

# Normalisasi fitur numerik
numerical_features = ['Price', 'Ingredients_count']
scaler = StandardScaler()
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])

# Pisah fitur & target
X = data_encoded.drop('Profitability', axis=1)
y = data_encoded['Profitability']

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model Decision Tree ===
dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=100,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt'
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\nEvaluasi Model Decision Tree:")
print(classification_report(y_test, dt_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Cross-validation
dt_cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
print("\nCross-validation scores:", dt_cv_scores)

# === Simpan Model ===
bundle = {
    "model": dt,
    "scaler": scaler,
    "label_encoder": label_encoder,
    "numerical_features": numerical_features,
    "feature_columns": X.columns.tolist(),
    "class_names": {i: label for i, label in enumerate(label_encoder.classes_)}
}

os.makedirs("model", exist_ok=True)
joblib.dump(bundle, "model/menu_model.joblib")
print("\nModel disimpan di model/menu_model.joblib")