import pandas as pd
import pickle
from sklearn.metrics import classification_report

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load test data
df = pd.read_csv("data/breast_cancer.csv")
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

preds = model.predict(X)

# Save report
report = classification_report(y, preds, output_dict=True)
pd.DataFrame(report).T.to_csv("evaluation/classification_report.csv")

print("✅ Evaluation report saved.")
