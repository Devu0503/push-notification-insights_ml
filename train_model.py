import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("enhanced_push_notification_data.csv")

# Create ResponseNumeric column if not already present
if "ResponseNumeric" not in df.columns:
    df["ResponseNumeric"] = df["Response"].map({"Yes": 1, "No": 0})

# Features and target
categorical_cols = ['Category', 'UserType', 'DeviceType', 'OS', 'DayOfWeek', 'NotificationType']
numerical_cols = ['DiscountOffered', 'HourOfDay', 'TimeDelayMin', 'NotificationTextLength']
target = 'ResponseNumeric'

X = df[categorical_cols + numerical_cols]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep numerical columns as-is
)

# Full pipeline: preprocessing + model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit model
pipeline.fit(X_train, y_train)

# Save pipeline
with open("models/response_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved successfully as models/response_model.pkl")
