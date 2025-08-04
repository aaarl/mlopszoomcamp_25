import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import os

# Paths
MODEL_PATH = "model.pkl"
DATA_PATH = "data/breast_cancer.csv"
OUTPUT_PATH = "evaluation/classification_report.csv"

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Predict
preds = model.predict(X)

# Evaluation
report = classification_report(y, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save report
os.makedirs("evaluation", exist_ok=True)
report_df.to_csv(OUTPUT_PATH)

print("✅ Evaluation complete. Report saved to:", OUTPUT_PATH)
