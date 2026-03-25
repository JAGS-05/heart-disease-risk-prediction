import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("framingham_heart_study.csv")

# -----------------------------
# Handle missing values
# -----------------------------
df.fillna(df.mean(), inplace=True)

# -----------------------------
# DEFINE FEATURE ORDER (CRITICAL)
# Must match backend exactly
# -----------------------------
features = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose"
]

X = df[features]
y = df["TenYearCHD"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Handle class imbalance
# -----------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -----------------------------
# Feature scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate model (optional but good)
# -----------------------------
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# -----------------------------
# Save model + scaler
# -----------------------------
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully!")