import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("framingham_heart_study.csv")

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Features and target
X = df.drop("TenYearCHD", axis=1)
y = df["TenYearCHD"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "heart_model.pkl")

print("Model trained and saved!")