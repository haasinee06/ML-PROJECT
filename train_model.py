"""
PLACEMENT PREDICTION - Model Training Script
==============================================
Run this once to train and save the model.
(You can also copy these cells into a Jupyter/Colab notebook - same code works.)
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------------
# STEP 1: Load data
# ----------------------------------------------------------------
df = pd.read_csv('/home/claude/placement_prediction/dataset/Placement_Data_Full_Class.csv')
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ----------------------------------------------------------------
# STEP 2: Clean data
# ----------------------------------------------------------------
# 'salary' is NaN for unplaced students - that's expected, not an error.
# We drop sl_no (just an index) and salary (would leak the answer -
# a placed student's salary is only known AFTER placement).
df = df.drop(columns=['sl_no', 'salary'])

# ----------------------------------------------------------------
# STEP 3: Encode categorical columns
# ----------------------------------------------------------------
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le   # save each encoder so the web app can reuse the same mapping

target_encoder = LabelEncoder()
df['status'] = target_encoder.fit_transform(df['status'])  # Placed=1, Not Placed=0
encoders['status'] = target_encoder

print("\nEncoded columns:", categorical_cols)
print("Status classes:", list(target_encoder.classes_), "->", list(target_encoder.transform(target_encoder.classes_)))

# ----------------------------------------------------------------
# STEP 4: Train/test split
# ----------------------------------------------------------------
X = df.drop(columns=['status'])
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

feature_order = list(X.columns)  # remember exact column order for the web app
print("\nFeature order:", feature_order)

# ----------------------------------------------------------------
# STEP 5: Train models & compare
# ----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=target_encoder.classes_))
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\n>>> Best model: {best_name} with accuracy {best_acc:.4f}")

# ----------------------------------------------------------------
# STEP 6: Save model + encoders + feature order
# ----------------------------------------------------------------
with open('/home/claude/placement_prediction/model/model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('/home/claude/placement_prediction/model/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('/home/claude/placement_prediction/model/feature_order.pkl', 'wb') as f:
    pickle.dump(feature_order, f)

print("\nSaved model.pkl, encoders.pkl, feature_order.pkl to model/ folder.")
